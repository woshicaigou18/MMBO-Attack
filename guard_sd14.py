# guard_sd14.py
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union, List

import torch
import numpy as np
from PIL import Image

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor


# -----------------------------
# Utilities: normalization
# -----------------------------
_ZERO_WIDTH = [
    "\u200b", "\u200c", "\u200d", "\ufeff", "\u2060",  # ZWSP, ZWNJ, ZWJ, BOM, WJ
]
_ZERO_WIDTH_RE = re.compile("|".join(map(re.escape, _ZERO_WIDTH)))

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    # Unicode normalization (compat), remove zero-width, lowercase, collapse whitespace
    s = unicodedata.normalize("NFKC", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# -----------------------------
# Text Guard
# -----------------------------
@dataclass
class TextGuardConfig:
    hard_block_patterns: Tuple[str, ...] = (
        r"\bchild porn\b",
        r"\bunderage\b",
        r"\bloli\b",
        r"\bincest\b",
    )
    soft_block_patterns: Tuple[str, ...] = (
        r"\bnude\b",
        r"\bnudity\b",
        r"\bsex\b",
        r"\bexplicit\b",
        r"\bgore\b",
        r"\bgraphic\b",
    )

class TextGuard:
    def __init__(self, cfg: Optional[TextGuardConfig] = None):
        self.cfg = cfg or TextGuardConfig()
        self._hard = [re.compile(p, flags=re.IGNORECASE) for p in self.cfg.hard_block_patterns]
        self._soft = [re.compile(p, flags=re.IGNORECASE) for p in self.cfg.soft_block_patterns]

    def check(self, prompt: str) -> Dict[str, Any]:
        p = normalize_text(prompt)
        hard_hits = [pat.pattern for pat in self._hard if pat.search(p)]
        soft_hits = [pat.pattern for pat in self._soft if pat.search(p)]
        return {
            "normalized": p,
            "hard_block": len(hard_hits) > 0,
            "soft_block": len(soft_hits) > 0,
            "hard_hits": hard_hits,
            "soft_hits": soft_hits,
        }

    def sanitize(self, prompt: str) -> str:
        # 可选：对 soft_block 做“净化/改写”
        # 这里给出一个非常保守的示例：只做最小替换，你可以换成更复杂的 rewrite
        p = prompt
        for pat in self._soft:
            p = pat.sub("[filtered]", p)
        return p


# -----------------------------
# Image Guard (input image)
# -----------------------------
@dataclass
class ImageGuardConfig:
    max_side: int = 2048
    min_side: int = 64
    block_if_nsfw: bool = True

class ImageGuard:
    """
    利用 diffusers 的 StableDiffusionSafetyChecker 对“输入图像”做 NSFW 检测。
    注意：它本来多用于输出图像，但同样可用于 img2img 的 init_image 入口拦截。
    """
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        cfg: Optional[ImageGuardConfig] = None,
        safety_checker_id: str = "./safety-checker",
    ):
        self.cfg = cfg or ImageGuardConfig()
        self.device = torch.device(device)
        self.feature_extractor = CLIPFeatureExtractor.from_pretrained(safety_checker_id)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_checker_id).to(self.device)
        self.safety_checker.eval()

    @torch.no_grad()
    def check_pil(self, img: Image.Image) -> Dict[str, Any]:
        if not isinstance(img, Image.Image):
            raise TypeError("init_image must be a PIL.Image")

        w, h = img.size
        size_ok = (min(w, h) >= self.cfg.min_side) and (max(w, h) <= self.cfg.max_side)

        # Safety checker expects: images as numpy float in [0, 1], shape (B, H, W, 3)
        img_rgb = img.convert("RGB")
        np_img = np.array(img_rgb).astype(np.float32) / 255.0
        np_img = np_img[None, ...]  # batch

        # Feature extractor returns pixel_values (B, 3, 224, 224)
        fe = self.feature_extractor(images=[img_rgb], return_tensors="pt")
        clip_in = fe["pixel_values"].to(self.device)

        # safety_checker signature: (clip_input, images) -> (images, has_nsfw_concept)
        _, has_nsfw = self.safety_checker(clip_in, np_img)

        has_nsfw_bool = bool(has_nsfw[0]) if isinstance(has_nsfw, (list, tuple)) else bool(has_nsfw)
        return {
            "size_ok": size_ok,
            "width": w,
            "height": h,
            "has_nsfw": has_nsfw_bool,
            "block": (not size_ok) or (self.cfg.block_if_nsfw and has_nsfw_bool),
        }


# -----------------------------
# Guarded wrappers
# -----------------------------
class GuardedSD14:
    """
    统一封装：
    - 文生图：过滤 prompt/negative_prompt
    - 图生图：额外过滤 init_image
    """
    def __init__(
        self,
        pipe: Union[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline],
        text_guard: Optional[TextGuard] = None,
        image_guard: Optional[ImageGuard] = None,
        softblock_add_negative: str = "no nudity, no sex, no gore, no violence, safe content",
        softblock_guidance_scale_cap: float = 7.5,
    ):
        self.pipe = pipe
        self.text_guard = text_guard or TextGuard()
        self.image_guard = image_guard  # 只有 img2img 才需要
        self.softblock_add_negative = softblock_add_negative
        self.softblock_guidance_scale_cap = softblock_guidance_scale_cap

    def _guard_text(
        self,
        prompt: str,
        negative_prompt: Optional[str],
        guidance_scale: Optional[float],
        do_rewrite: bool = False,
    ) -> Tuple[str, Optional[str], Optional[float], Dict[str, Any]]:
        info_p = self.text_guard.check(prompt)
        info_n = self.text_guard.check(negative_prompt or "") if negative_prompt is not None else None

        # hard block
        if info_p["hard_block"] or (info_n and info_n["hard_block"]):
            raise ValueError(
                f"[BLOCKED] hard policy hit. prompt_hits={info_p['hard_hits']} "
                f"neg_hits={(info_n['hard_hits'] if info_n else [])}"
            )

        # soft block: 降级策略（可按需调整）
        soft = info_p["soft_block"] or (info_n["soft_block"] if info_n else False)
        if soft:
            if do_rewrite:
                prompt = self.text_guard.sanitize(prompt)

            # 合并负向提示词
            if negative_prompt is None or len(negative_prompt.strip()) == 0:
                negative_prompt = self.softblock_add_negative
            else:
                negative_prompt = negative_prompt + ", " + self.softblock_add_negative

            # guidance scale 上限
            if guidance_scale is not None:
                guidance_scale = float(min(guidance_scale, self.softblock_guidance_scale_cap))

        meta = {"prompt": info_p, "negative_prompt": info_n, "soft_block_applied": soft}
        return prompt, negative_prompt, guidance_scale, meta

    def __call__(self, *args, **kwargs):
        """
        兼容 diffusers pipe(...) 调用形式。
        关键 kwargs:
          - prompt, negative_prompt, guidance_scale
          - image (img2img 常用键) 或 init_image（你自己的键）
        """
        prompt = kwargs.get("prompt", None)
        if prompt is None:
            # diffusers 也允许 prompt 作为第一个 positional
            if len(args) >= 1:
                prompt = args[0]
            else:
                raise ValueError("Missing prompt")

        negative_prompt = kwargs.get("negative_prompt", None)
        guidance_scale = kwargs.get("guidance_scale", None)

        # 1) 文本输入过滤
        prompt2, neg2, gs2, meta_text = self._guard_text(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            do_rewrite=False,  # 你也可以改 True
        )
        kwargs["prompt"] = prompt2
        kwargs["negative_prompt"] = neg2
        if gs2 is not None:
            kwargs["guidance_scale"] = gs2

        # 2) 图像输入过滤（仅当存在输入图像）
        # diffusers img2img 默认参数名是 image
        init_img = kwargs.get("image", None)
        if init_img is None:
            init_img = kwargs.get("init_image", None)

        meta_img = None
        if init_img is not None:
            if self.image_guard is None:
                raise ValueError("image_guard is required for img2img input filtering")

            # 支持单张 PIL 或 list[PIL]
            if isinstance(init_img, list):
                meta_img = [self.image_guard.check_pil(im) for im in init_img]
                if any(m["block"] for m in meta_img):
                    raise ValueError(f"[BLOCKED] init_image policy hit: {meta_img}")
            else:
                meta_img = self.image_guard.check_pil(init_img)
                if meta_img["block"]:
                    raise ValueError(f"[BLOCKED] init_image policy hit: {meta_img}")

        out = self.pipe(*args, **kwargs)

        # 你如果希望把 meta 返回（便于日志），可以改成 return out, meta_text, meta_img
        return out


# -----------------------------
# Factory helpers
# -----------------------------
def build_sd14_txt2img(
    model_id: str = "./stable_diff",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> GuardedSD14:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    return GuardedSD14(pipe=pipe, text_guard=TextGuard(), image_guard=None)

def build_sd14_img2img(
    model_id: str = "./stable_diff",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> GuardedSD14:
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
    pipe = pipe.to(device)
    img_guard = ImageGuard(device=device)
    return GuardedSD14(pipe=pipe, text_guard=TextGuard(), image_guard=img_guard)
