import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "/home/Liyuhong/HIMRD-jailbreak/Qwen"      # 或者改成本地路径 "/path/to/Qwen3-7B"
INPUT_JSONL = "/home/Liyuhong/HIMRD-jailbreak/output_text_illegal.jsonl"    # 改成你的输入文件
OUTPUT_JSONL = "PPL.jsonl"

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda:1" and torch.cuda.is_bf16_supported() else torch.float16
ADD_SPECIAL_TOKENS = True

# ----------------------------
# Load model/tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map="auto" if DEVICE == "cuda" else None,
    trust_remote_code=True,
)
model.eval()


@torch.no_grad()
def ppl_causal_lm(text: str) -> float:
    enc = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=ADD_SPECIAL_TOKENS,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    labels = input_ids.clone()
    if attention_mask is not None:
        labels[attention_mask == 0] = -100

    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss
    return math.exp(loss.item())


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 兼容缺少最外层 { } 的行
                obj = json.loads("{" + line + "}")
            yield line_no, obj


class OnlineStats:
    """Welford online mean/variance."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def add(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def var_population(self) -> float:
        return self.M2 / self.n if self.n > 0 else float("nan")

    @property
    def var_sample(self) -> float:
        return self.M2 / (self.n - 1) if self.n > 1 else float("nan")


def main():
    in_path = Path(INPUT_JSONL)
    out_path = Path(OUTPUT_JSONL)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = OnlineStats()
    n_total, n_written, n_skipped = 0, 0, 0

    with open(out_path, "w", encoding="utf-8") as wf:
        for line_no, obj in iter_jsonl(str(in_path)):
            n_total += 1

            text = obj.get("optimized_prompt", None)
            if not isinstance(text, str) or len(text.strip()) == 0:
                n_skipped += 1
                continue

            ppl = ppl_causal_lm(text.strip())
            stats.add(ppl)

            out_obj = {
                "sample_idx": obj.get("sample_idx", line_no),
                "optimized_prompt": text,
                "ppl": ppl,
            }
            wf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 50 == 0:
                print(f"[progress] written={n_written}, skipped={n_skipped}, total_seen={n_total}")

    # 汇总输出（均值、方差、标准差）
    mean = stats.mean
    var_pop = stats.var_population
    var_samp = stats.var_sample
    std_pop = math.sqrt(var_pop) if var_pop == var_pop else float("nan")
    std_samp = math.sqrt(var_samp) if var_samp == var_samp else float("nan")

    print(f"[done] written={n_written}, skipped={n_skipped}, total_seen={n_total}")
    print(f"Output: {out_path.resolve()}")
    print("PPL summary:")
    print(f"  count = {stats.n}")
    print(f"  mean  = {mean}")
    print(f"  var_population (ddof=0) = {var_pop}")
    print(f"  std_population (ddof=0) = {std_pop}")
    print(f"  var_sample (ddof=1)     = {var_samp}")
    print(f"  std_sample (ddof=1)     = {std_samp}")

    # 可选：把汇总也写成一个 json 文件（不影响你要求的 PPL.jsonl）
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "count": stats.n,
        "mean": mean,
        "var_population_ddof0": var_pop,
        "std_population_ddof0": std_pop,
        "var_sample_ddof1": var_samp,
        "std_sample_ddof1": std_samp,
        "skipped": n_skipped,
        "input": str(in_path),
        "output": str(out_path),
        "model_id": MODEL_ID,
        "add_special_tokens": ADD_SPECIAL_TOKENS,
        "dtype": str(DTYPE),
        "device": DEVICE,
    }
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary JSON: {summary_path.resolve()}")


if __name__ == "__main__":
    main()