import os
import json
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from diffusers import StableDiffusionPipeline
JSONL_PATH = "output_church.jsonl"                  # 你的jsonl文件
OUT_DIR = "sd_outputs_church"                       # 输出目录
# MODEL_ID = "/home/Liyuhong/HIMRD-jailbreak/stable-diffusion_3.5"  # 可替换为本地路径，如 "/path/to/sd15"
MODEL_ID = "/home/Liyuhong/HIMRD-jailbreak/stable_diff"
esd_unet_pt = "/home/Liyuhong/Diffusion-MU-Attack/unlearned_ckpt_object/ESD_ckpt/garbage_truck.pt"

# 关键：把 UNet 权重替换成 ESD 的
state = torch.load(esd_unet_pt, map_location="cpu")


# 生成参数
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
HEIGHT = 512
WIDTH = 512
SEED = 1234

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda:1" else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        safety_checker=None if False else None,  # 如果你有合规需求，可按实际情况接入安全检查器
    ).to(device)

    # pipe.unet.load_state_dict(state)

    # 可选：省显存
    if device == "cuda:1":
        pipe.enable_attention_slicing()
        # pipe.enable_xformers_memory_efficient_attention()  # 安装xformers后可开

    gen = torch.Generator(device=device).manual_seed(SEED)

    for obj in read_jsonl(JSONL_PATH):
        line_no = obj.get("line_no", "NA")
        prompt = obj.get("line_line", "").strip()
        if not prompt:
            print(f"[skip] line_no={line_no}: empty input_line")
            continue

        # 生成
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda:1")):
            image = pipe(
                prompt=prompt,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                height=HEIGHT,
                width=WIDTH,
                generator=gen,
            ).images[0]

        out_path = os.path.join(OUT_DIR, f"{line_no:05d}.png" if isinstance(line_no, int) else f"{line_no}.png")
        image.save(out_path)
        print(f"[ok] line_no={line_no} -> {out_path}")

if __name__ == "__main__":
    main()
