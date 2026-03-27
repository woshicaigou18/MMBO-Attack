import os
import sys
import json
import csv
from pathlib import Path

import torch
from PIL import Image


# =========================
# 路径配置
# =========================
IMAGE_ROOT = Path("./stable1_illegal")
JSONL_PATH = Path("./output_illegal.jsonl")
LOCAL_CLIP_DIR = Path("/home/Liyuhong/HIMRD-jailbreak/Clip")

OUTPUT_JSON = Path("./clip_similarity_results_violence.json")
OUTPUT_CSV = Path("./clip_similarity_results_violence.csv")

START_IDX = 0
END_IDX = 756

# 可按需修改
MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# 加载 CLIP
# =========================
def load_clip_model():
    """
    兼容两种情况：
    1) 环境里可直接 import clip
    2) 需要从本地工程目录导入 clip

    返回:
        model, preprocess, clip_module
    """
    try:
        import clip
    except ImportError:
        sys.path.insert(0, str(LOCAL_CLIP_DIR))
        import clip

    model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
    model.eval()
    return model, preprocess, clip


# =========================
# 读取 jsonl
# =========================
def load_optimize_prompts(jsonl_path):
    prompts = []
    bad_lines = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                bad_lines.append({
                    "line_index": line_idx,
                    "reason": "empty line"
                })
                prompts.append(None)
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                bad_lines.append({
                    "line_index": line_idx,
                    "reason": f"json parse error: {repr(e)}"
                })
                prompts.append(None)
                continue

            prompt = obj.get("input_line", None)
            if prompt is None or not isinstance(prompt, str) or not prompt.strip():
                bad_lines.append({
                    "line_index": line_idx,
                    "reason": "missing or empty optimize_prompt"
                })
                prompts.append(None)
            else:
                prompts.append(prompt.strip())

    return prompts, bad_lines


# =========================
# 图片相似度计算
# =========================
@torch.no_grad()
def compute_image_text_similarity(model, preprocess, clip_module, image_path, text):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(DEVICE)
    text_input = clip_module.tokenize([text]).to(DEVICE)

    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_input)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    sim = (image_features @ text_features.T).item()
    return float(sim)


def list_images(folder):
    if not folder.exists() or not folder.is_dir():
        return []

    imgs = []
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    return imgs


# =========================
# 主流程
# =========================
def main():
    model, preprocess, clip_module = load_clip_model()
    prompts, prompt_issues = load_optimize_prompts(JSONL_PATH)

    results = []
    skipped = {
        "prompt_issues": prompt_issues,
        "directories": [],
        "images": []
    }

    valid_dir_means = []

    for idx in range(START_IDX, END_IDX + 1):
        dir_path = IMAGE_ROOT / str(idx)

        # 检查 prompt
        if idx >= len(prompts):
            skipped["directories"].append({
                "dir_index": idx,
                "dir_path": str(dir_path),
                "reason": "jsonl has fewer lines than expected"
            })
            continue

        prompt = prompts[idx]
        if prompt is None:
            skipped["directories"].append({
                "dir_index": idx,
                "dir_path": str(dir_path),
                "reason": "invalid optimize_prompt"
            })
            continue

        # 检查目录和图片
        if not dir_path.exists():
            skipped["directories"].append({
                "dir_index": idx,
                "dir_path": str(dir_path),
                "reason": "directory does not exist"
            })
            continue

        if not dir_path.is_dir():
            skipped["directories"].append({
                "dir_index": idx,
                "dir_path": str(dir_path),
                "reason": "path is not a directory"
            })
            continue

        image_paths = list_images(dir_path)
        if len(image_paths) == 0:
            skipped["directories"].append({
                "dir_index": idx,
                "dir_path": str(dir_path),
                "reason": "no valid image files found"
            })
            continue

        image_sims = []
        valid_image_count = 0

        for image_path in image_paths:
            try:
                sim = compute_image_text_similarity(
                    model, preprocess, clip_module, image_path, prompt
                )
                image_sims.append(sim)
                valid_image_count += 1
            except Exception as e:
                skipped["images"].append({
                    "dir_index": idx,
                    "image_path": str(image_path),
                    "reason": f"failed to process image: {repr(e)}"
                })

        if len(image_sims) == 0:
            skipped["directories"].append({
                "dir_index": idx,
                "dir_path": str(dir_path),
                "reason": "all images failed to process"
            })
            continue

        # dir_mean = sum(image_sims) / len(image_sims)
        dir_mean = max(image_sims)
        valid_dir_means.append(dir_mean)

        results.append({
            "dir_index": idx,
            "dir_path": str(dir_path),
            "prompt": prompt,
            "num_total_images_found": len(image_paths),
            "num_valid_images": valid_image_count,
            "num_failed_images": len(image_paths) - valid_image_count,
            "dir_mean_clip_similarity": dir_mean
        })

        print(
            f"[{idx}] valid_images={valid_image_count}/{len(image_paths)}, "
            f"dir_mean={dir_mean:.6f}"
        )

    # 总体平均：按目录均值再取平均，每个目录权重相同
    overall_mean_equal_dir_weight = (
        sum(valid_dir_means) / len(valid_dir_means)
        if len(valid_dir_means) > 0 else None
    )

    summary = {
        "image_root": str(IMAGE_ROOT),
        "jsonl_path": str(JSONL_PATH),
        "clip_dir": str(LOCAL_CLIP_DIR),
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "index_range": [START_IDX, END_IDX],
        "total_expected_directories": END_IDX - START_IDX + 1,
        "num_valid_directories": len(valid_dir_means),
        "num_skipped_directories": (END_IDX - START_IDX + 1) - len(valid_dir_means),
        "overall_mean_equal_dir_weight": overall_mean_equal_dir_weight
    }

    output_obj = {
        "summary": summary,
        "per_directory_results": results,
        "skipped": skipped
    }

    # 保存 JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output_obj, f, ensure_ascii=False, indent=2)

    # 保存 CSV
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dir_index",
                "dir_path",
                "num_total_images_found",
                "num_valid_images",
                "num_failed_images",
                "dir_mean_clip_similarity",
                "prompt",
            ]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # 终端打印总结果
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    print("\nSkipped directory count:", len(skipped["directories"]))
    print("Skipped image count:", len(skipped["images"]))
    print("Prompt issue count:", len(skipped["prompt_issues"]))

    if skipped["directories"]:
        print("\nSkipped directories:")
        for item in skipped["directories"]:
            print(item)

    if skipped["images"]:
        print("\nSkipped images:")
        for item in skipped["images"]:
            print(item)

    if skipped["prompt_issues"]:
        print("\nPrompt issues:")
        for item in skipped["prompt_issues"]:
            print(item)

    print(f"\nJSON saved to: {OUTPUT_JSON.resolve()}")
    print(f"CSV saved to:  {OUTPUT_CSV.resolve()}")


if __name__ == "__main__":
    main()


# import os
# import sys
# import json
# import csv
# from pathlib import Path

# import torch
# from PIL import Image


# # =========================
# # 路径配置
# # =========================
# IMAGE_ROOT = Path("./sd_outputs_violence_no_attack")
# JSONL_PATH = Path("./output_violence copy.jsonl")
# LOCAL_CLIP_DIR = Path("/home/Liyuhong/HIMRD-jailbreak/Clip")

# OUTPUT_JSON = Path("./clip_similarity_results_violence.json")
# OUTPUT_CSV = Path("./clip_similarity_results_violence.csv")

# START_IDX = 1
# END_IDX = 756

# # 可按需修改
# MODEL_NAME = "ViT-B/32"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# # =========================
# # 加载 CLIP
# # =========================
# def load_clip_model():
#     """
#     兼容两种情况：
#     1) 环境里可直接 import clip
#     2) 需要从本地工程目录导入 clip

#     返回:
#         model, preprocess, clip_module
#     """
#     try:
#         import clip
#     except ImportError:
#         sys.path.insert(0, str(LOCAL_CLIP_DIR))
#         import clip

#     model, preprocess = clip.load(MODEL_NAME, device=DEVICE)
#     model.eval()
#     return model, preprocess, clip


# # =========================
# # 读取 jsonl
# # =========================
# def load_optimize_prompts(jsonl_path):
#     prompts = []
#     bad_lines = []

#     with open(jsonl_path, "r", encoding="utf-8") as f:
#         for line_idx, line in enumerate(f):
#             line = line.strip()
#             if not line:
#                 bad_lines.append({
#                     "line_index": line_idx,
#                     "reason": "empty line"
#                 })
#                 prompts.append(None)
#                 continue

#             try:
#                 obj = json.loads(line)
#             except Exception as e:
#                 bad_lines.append({
#                     "line_index": line_idx,
#                     "reason": f"json parse error: {repr(e)}"
#                 })
#                 prompts.append(None)
#                 continue

#             prompt = obj.get("input_line", None)
#             if prompt is None or not isinstance(prompt, str) or not prompt.strip():
#                 bad_lines.append({
#                     "line_index": line_idx,
#                     "reason": "missing or empty optimize_prompt"
#                 })
#                 prompts.append(None)
#             else:
#                 prompts.append(prompt.strip())

#     return prompts, bad_lines


# # =========================
# # 图片相似度计算
# # =========================
# @torch.no_grad()
# def compute_image_text_similarity(model, preprocess, clip_module, image_path, text):
#     image = Image.open(image_path).convert("RGB")
#     image_input = preprocess(image).unsqueeze(0).to(DEVICE)
#     text_input = clip_module.tokenize([text]).to(DEVICE)

#     image_features = model.encode_image(image_input)
#     text_features = model.encode_text(text_input)

#     image_features = image_features / image_features.norm(dim=-1, keepdim=True)
#     text_features = text_features / text_features.norm(dim=-1, keepdim=True)

#     sim = (image_features @ text_features.T).item()
#     return float(sim)


# def find_image_by_index(image_root, idx):
#     """
#     在 IMAGE_ROOT 下查找 000001.jpg / 000001.png / ...
#     返回找到的图片路径，找不到则返回 None
#     """
#     stem = f"{idx:05d}"
#     for ext in sorted(IMAGE_EXTS):
#         candidate = image_root / f"{stem}{ext}"
#         if candidate.exists() and candidate.is_file():
#             return candidate
#     return None


# # =========================
# # 主流程
# # =========================
# def main():
#     model, preprocess, clip_module = load_clip_model()
#     prompts, prompt_issues = load_optimize_prompts(JSONL_PATH)

#     results = []
#     skipped = {
#         "prompt_issues": prompt_issues,
#         "images": []
#     }

#     valid_sims = []

#     for idx in range(START_IDX, END_IDX + 1):
#         image_path = find_image_by_index(IMAGE_ROOT, idx)

#         # 文本读取方式不变：第1张图对应 jsonl 第1行，即 prompts[0]
#         prompt_idx = idx - 1

#         if prompt_idx >= len(prompts):
#             skipped["images"].append({
#                 "image_index": idx,
#                 "image_path": None,
#                 "reason": "jsonl has fewer lines than expected"
#             })
#             continue

#         prompt = prompts[prompt_idx]
#         if prompt is None:
#             skipped["images"].append({
#                 "image_index": idx,
#                 "image_path": None,
#                 "reason": "invalid optimize_prompt"
#             })
#             continue

#         if image_path is None:
#             skipped["images"].append({
#                 "image_index": idx,
#                 "image_path": str(IMAGE_ROOT / f"{idx:06d}.*"),
#                 "reason": "image file does not exist"
#             })
#             continue

#         try:
#             sim = compute_image_text_similarity(
#                 model, preprocess, clip_module, image_path, prompt
#             )
#             valid_sims.append(sim)

#             results.append({
#                 "image_index": idx,
#                 "image_name": image_path.name,
#                 "image_path": str(image_path),
#                 "prompt": prompt,
#                 "clip_similarity": sim
#             })

#             print(f"[{idx:05d}] image={image_path.name}, clip_similarity={sim:.6f}")

#         except Exception as e:
#             skipped["images"].append({
#                 "image_index": idx,
#                 "image_path": str(image_path),
#                 "reason": f"failed to process image: {repr(e)}"
#             })

#     overall_mean = (
#         sum(valid_sims) / len(valid_sims)
#         if len(valid_sims) > 0 else None
#     )

#     summary = {
#         "image_root": str(IMAGE_ROOT),
#         "jsonl_path": str(JSONL_PATH),
#         "clip_dir": str(LOCAL_CLIP_DIR),
#         "model_name": MODEL_NAME,
#         "device": DEVICE,
#         "index_range": [START_IDX, END_IDX],
#         "total_expected_images": END_IDX - START_IDX + 1,
#         "num_valid_images": len(valid_sims),
#         "num_skipped_images": (END_IDX - START_IDX + 1) - len(valid_sims),
#         "overall_mean_clip_similarity": overall_mean
#     }

#     output_obj = {
#         "summary": summary,
#         "per_image_results": results,
#         "skipped": skipped
#     }

#     # 保存 JSON
#     with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
#         json.dump(output_obj, f, ensure_ascii=False, indent=2)

#     # 保存 CSV
#     with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
#         writer = csv.DictWriter(
#             f,
#             fieldnames=[
#                 "image_index",
#                 "image_name",
#                 "image_path",
#                 "clip_similarity",
#                 "prompt",
#             ]
#         )
#         writer.writeheader()
#         for row in results:
#             writer.writerow(row)

#     # 终端打印总结果
#     print("\n" + "=" * 80)
#     print("Summary")
#     print("=" * 80)
#     print(json.dumps(summary, ensure_ascii=False, indent=2))

#     print("\nSkipped image count:", len(skipped["images"]))
#     print("Prompt issue count:", len(skipped["prompt_issues"]))

#     if skipped["images"]:
#         print("\nSkipped images:")
#         for item in skipped["images"]:
#             print(item)

#     if skipped["prompt_issues"]:
#         print("\nPrompt issues:")
#         for item in skipped["prompt_issues"]:
#             print(item)

#     print(f"\nJSON saved to: {OUTPUT_JSON.resolve()}")
#     print(f"CSV saved to:  {OUTPUT_CSV.resolve()}")


# if __name__ == "__main__":
#     main()