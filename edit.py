# # edit.py
# # Read output_text.jsonl, run SD1.4 img2img using (out_path image, edit_prompt),
# # save results into ./stable1/

import os
import json
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

import torch
from diffusers import StableDiffusionPipelineSafe
from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
# from receler.erasers.diffusers_erasers import inject_eraser

# pipeline = StableDiffusionPipelineSafe.from_pretrained(
#     "AIML-TUDA/stable-diffusion-safe", torch_dtype=torch.float16
# ).to("cuda")
# prompt = "the four horsewomen of the apocalypse, painting by tom of finland, gaston bussiere, craig mullins, j. c. leyendecker"
# image = pipeline(prompt=prompt, **SafetyConfig.MAX).images[0]
import torch
from guard_sd14 import build_sd14_img2img

# esd_unet_pt = "/home/Liyuhong/Diffusion-MU-Attack/unlearned_ckpt_object/ESD_ckpt/garbage_truck.pt"
# state = torch.load(esd_unet_pt, map_location="cpu")
pipe = build_sd14_img2img(device="cuda", torch_dtype=torch.float16)
# pipe.pipe.unet.load_state_dict(state)

# RECELER_CKPT_DIR = "/home/Liyuhong/Receler/models/receler-word_nudity-rank_128-advrs_iter_50-start_0-num_prompts_16-concept_reg_0.1-mask_thres_0.1-iter_500-lr_0.0003"   # <-- 改成你的路径
# RECELER_WEIGHTS = os.path.join(RECELER_CKPT_DIR, "eraser_weights.pt")
# RECELER_CONFIG  = os.path.join(RECELER_CKPT_DIR, "eraser_config.json")

# ----------------------------
# Config
# ----------------------------
JSONL_PATH = "./output_text.jsonl"
OUT_DIR = "./stable1"

MODEL_ID = "./stable_diff"  # SD 1.4
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32 if DEVICE == "cuda:1" else torch.float32

# img2img params
STRENGTH = 0.55           # 0~1, higher => larger change
GUIDANCE_SCALE = 7.5
NUM_INFERENCE_STEPS = 30
SEED = 1234

# Optional negative prompt (helps reduce artifacts; keep generic)
NEGATIVE_PROMPT = "low quality, blurry, distorted, deformed, bad anatomy, text, watermark"




def load_rgb_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {line_no}: {e}")
                continue




def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        # safety_checker=None,  # if you want the default safety checker, remove this line
    )
    # esd_unet_pt = "/home/Liyuhong/Diffusion-MU-Attack/unlearned_ckpt_object/ESD_ckpt/garbage_truck.pt"
    # esd_unet_pt = "./concept-prune/results/results_seed_0/stable-diffusion/runwayml/stable-diffusion-v1-5/naked/checkpoints/skill_ratio_0.01_timesteps_50_threshold0.01.pt"
    # state = torch.load(esd_unet_pt, map_location="cpu")
    # pipe.unet.load_state_dict(state)
    pipe = pipe.to(DEVICE)
    # if DEVICE == "cuda":
    #     pipe.enable_xformers_memory_efficient_attention() if hasattr(pipe, "enable_xformers_memory_efficient_attention") else None

    gen = torch.Generator(device=DEVICE).manual_seed(SEED)
    line_pre = None

    for line_no, obj in iter_jsonl(JSONL_PATH):
        out_path = obj.get("out_path", None)
        line_no1 = obj.get("sample_idx", None)
        # if line_pre == line_no1:
        #     continue
        OUT_ROOT = os.path.join(OUT_DIR, str(line_no1))
        os.makedirs(OUT_ROOT, exist_ok=True)
        prompt = obj.get("optimized_prompt", None)

        if not out_path or not prompt:
            print(f"[WARN] Missing out_path/edit_prompt at line {line_no}, skip.")
            continue

        src_path = Path(out_path)
        if not src_path.exists():
            print(f"[WARN] File not found (line {line_no}): {src_path}")
            continue

        try:
            init_image = load_rgb_image(str(src_path))

            # Run img2img
            result = pipe(
                prompt=prompt,
                image=init_image,
                strength=STRENGTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                generator=gen,
                **SafetyConfig.MAX
            ).images[0]

            # Save with same basename into stable1/
            save_path = Path(OUT_ROOT) / src_path.name
            result.save(save_path)
            print(f"[OK] line {line_no}: saved -> {save_path}")
            line_pre = line_no1

        except Exception as e:
            print(f"[ERR] line {line_no}: {e}")

    print("Done.")

# def _get_unet_from_pipe(pipe_obj):
#     # 兼容你这个 guard_sd14 的返回（你之前注释里用过 pipe.pipe.unet）
#     if hasattr(pipe_obj, "pipe") and hasattr(pipe_obj.pipe, "unet"):
#         return pipe_obj.pipe.unet
#     if hasattr(pipe_obj, "unet"):
#         return pipe_obj.unet
#     raise AttributeError("Cannot find unet in pipe_obj (expected .unet or .pipe.unet).")

# def main():
#     Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

#     # ---- (A) build your SD1.4 img2img pipeline (guarded) ----
#     pipe = build_sd14_img2img(device=DEVICE, torch_dtype=DTYPE)

#     # ---- (B) inject Receler eraser into UNet (ONCE) ----
#     if os.path.isdir(RECELER_CKPT_DIR):
#         if not (os.path.exists(RECELER_WEIGHTS) and os.path.exists(RECELER_CONFIG)):
#             raise FileNotFoundError(
#                 f"Receler files missing in {RECELER_CKPT_DIR}: need eraser_weights.pt and eraser_config.json"
#             )

#         with open(RECELER_CONFIG, "r", encoding="utf-8") as f:
#             eraser_config = json.load(f)

#         eraser_ckpt = torch.load(RECELER_WEIGHTS, map_location="cpu")

#         unet = _get_unet_from_pipe(pipe)
#         # Receler's inject_eraser(unet, eraser_ckpt, eraser_rank, ...)
#         inject_eraser(unet, eraser_ckpt, **eraser_config)
#         unet.to(DEVICE)

#         print(f"[OK] Receler injected from: {RECELER_CKPT_DIR}")
#     else:
#         # 你也可以在这里支持 HF Hub 的 erased model（如果你有发布/下载）
#         raise FileNotFoundError(f"RECELER_CKPT_DIR not found: {RECELER_CKPT_DIR}")

#     gen = torch.Generator(device=DEVICE).manual_seed(SEED)

#     for line_no, obj in iter_jsonl(JSONL_PATH):
#         out_path = obj.get("out_path", None)
#         line_no1 = obj.get("sample_idx", None)
#         OUT_ROOT = os.path.join(OUT_DIR, str(line_no1))
#         os.makedirs(OUT_ROOT, exist_ok=True)

#         prompt = obj.get("optimized_prompt", None)
#         if not out_path or not prompt:
#             print(f"[WARN] Missing out_path/optimized_prompt at line {line_no}, skip.")
#             continue

#         src_path = Path(out_path)
#         if not src_path.exists():
#             print(f"[WARN] File not found (line {line_no}): {src_path}")
#             continue

#         try:
#             init_image = load_rgb_image(str(src_path))

#             result = pipe(
#                 prompt=prompt,
#                 image=init_image,
#                 strength=STRENGTH,
#                 guidance_scale=GUIDANCE_SCALE,
#                 num_inference_steps=NUM_INFERENCE_STEPS,
#                 generator=gen,
#                 # negative_prompt=NEGATIVE_PROMPT,  # 你需要的话打开
#                 **SafetyConfig.MAX
#             ).images[0]

#             save_path = Path(OUT_ROOT) / src_path.name
#             result.save(save_path)
#             print(f"[OK] line {line_no}: saved -> {save_path}")

#         except Exception as e:
#             print(f"[ERR] line {line_no}: {e}")

if __name__ == "__main__":
    main()


# import os, json
# from pathlib import Path
# import torch
# from PIL import Image
# from diffusers.pipelines.stable_diffusion_safe import SafetyConfig
# from guard_sd14 import build_sd14_img2img

# # ----------------------------
# # AdvUnlearn TextEncoder ckpt
# # ----------------------------
# ADV_TEXTENCODER_BIN = "./nudity/pytorch_model.bin"  # <-- 改成你的路径

# JSONL_PATH = "./output_text_illegal.jsonl"
# OUT_DIR = "./stable1_violence"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32

# STRENGTH = 0.55
# GUIDANCE_SCALE = 7.5
# NUM_INFERENCE_STEPS = 30
# SEED = 1234


# def load_rgb_image(path: str) -> Image.Image:
#     img = Image.open(path)
#     if img.mode != "RGB":
#         img = img.convert("RGB")
#     return img


# def iter_jsonl(path: str):
#     with open(path, "r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 yield line_no, json.loads(line)
#             except json.JSONDecodeError as e:
#                 print(f"[WARN] JSON decode error at line {line_no}: {e}")
#                 continue


# def _get_inner_pipe(guarded):
#     # GuardedSD14(pipe=diffusers_pipeline, ...)
#     if hasattr(guarded, "pipe"):
#         return guarded.pipe
#     raise AttributeError("Expected GuardedSD14-like object with attribute .pipe")


# def load_advunlearn_text_encoder(guarded_pipe, ckpt_path: str, device: str, dtype: torch.dtype):
#     if not os.path.exists(ckpt_path):
#         raise FileNotFoundError(f"AdvUnlearn text_encoder .bin not found: {ckpt_path}")

#     # 1) load state dict on CPU
#     sd = torch.load(ckpt_path, map_location="cpu")
#     if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
#         # 有些保存会包一层
#         sd = sd["state_dict"]

#     if not isinstance(sd, dict):
#         raise TypeError(f"Unexpected checkpoint format: {type(sd)} (expect dict[str, Tensor])")

#     # 2) 兼容 key 前缀（常见：'text_encoder.' / 'cond_stage_model.transformer.' 等）
#     #    先尽量直接 load；失败再做 strip 前缀
#     te = guarded_pipe.text_encoder
#     try:
#         missing, unexpected = te.load_state_dict(sd, strict=False)
#     except RuntimeError:
#         # strip common prefixes
#         def strip_prefix(d, pref):
#             return {k[len(pref):] if k.startswith(pref) else k: v for k, v in d.items()}

#         candidates = [
#             sd,
#             strip_prefix(sd, "text_encoder."),
#             strip_prefix(sd, "cond_stage_model."),
#             strip_prefix(sd, "cond_stage_model.transformer."),
#         ]

#         last_err = None
#         for cand in candidates:
#             try:
#                 missing, unexpected = te.load_state_dict(cand, strict=False)
#                 sd = cand
#                 break
#             except RuntimeError as e:
#                 last_err = e
#         else:
#             raise last_err

#     # 3) move to device/dtype (关键：避免 CPU/GPU 混用)
#     te.to(device=device, dtype=dtype)
#     te.eval()

#     # （可选）打印一下 load 情况，便于确认是否真替换成功
#     print(f"[OK] AdvUnlearn text_encoder loaded from: {ckpt_path}")
#     if len(unexpected) > 0:
#         print(f"[WARN] Unexpected keys: {len(unexpected)} (show 5) -> {unexpected[:5]}")
#     if len(missing) > 0:
#         print(f"[WARN] Missing keys: {len(missing)} (show 5) -> {missing[:5]}")


# def main():
#     Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

#     # 1) build guarded SD1.4 img2img
#     guarded = build_sd14_img2img(device=DEVICE, torch_dtype=DTYPE)
#     inner = _get_inner_pipe(guarded)

#     # 2) replace / load AdvUnlearn text encoder
#     load_advunlearn_text_encoder(inner, ADV_TEXTENCODER_BIN, DEVICE, DTYPE)

#     # 3) generator on same device
#     gen = torch.Generator(device=DEVICE).manual_seed(SEED)

#     for line_no, obj in iter_jsonl(JSONL_PATH):
#         out_path = obj.get("out_path", None)
#         sample_idx = obj.get("sample_idx", None)
#         prompt = obj.get("optimized_prompt", None)

#         if not out_path or not prompt:
#             print(f"[WARN] Missing out_path/optimized_prompt at line {line_no}, skip.")
#             continue

#         src_path = Path(out_path)
#         if not src_path.exists():
#             print(f"[WARN] File not found (line {line_no}): {src_path}")
#             continue

#         out_root = os.path.join(OUT_DIR, str(sample_idx))
#         os.makedirs(out_root, exist_ok=True)

#         try:
#             init_image = load_rgb_image(str(src_path))

#             # 注意：调用的是 GuardedSD14 的 __call__，它内部会转发到 inner pipe
#             result = guarded(
#                 prompt=prompt,
#                 image=init_image,
#                 strength=STRENGTH,
#                 guidance_scale=GUIDANCE_SCALE,
#                 num_inference_steps=NUM_INFERENCE_STEPS,
#                 generator=gen,
#                 **SafetyConfig.MAX
#             ).images[0]

#             save_path = Path(out_root) / src_path.name
#             result.save(save_path)
#             print(f"[OK] line {line_no}: saved -> {save_path}")

#         except Exception as e:
#             print(f"[ERR] line {line_no}: {e}")


# if __name__ == "__main__":
#     main()