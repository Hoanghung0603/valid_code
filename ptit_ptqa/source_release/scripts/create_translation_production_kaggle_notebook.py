"""Generate the resumable two-T4 Kaggle notebook for production query translation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/16_translation_production_2xt4_kaggle.ipynb"


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


QWEN_WORKER = r'''
import argparse
import gc
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?!\w)")
ASCII_MEDICAL_RE = re.compile(r"\b(?=[A-Za-z0-9./+-]*[A-Za-z])(?=[A-Za-z0-9./+-]*(?:\d|[A-Z]))[A-Za-z0-9]+(?:[./+-][A-Za-z0-9]+)*\b")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")

SYSTEM_PROMPTS = {
    "zh": """You are a meticulous medical translator for information retrieval. Translate the Vietnamese patient query literally into Simplified Chinese. Preserve every symptom, body part, disease or condition, medicine or product name, test, Arabic numeral, unit, age, duration, pregnancy detail, uncertainty, comparison, relationship, and negation. Do not diagnose, answer, explain, summarize, normalize, expand, or omit information. If wording is ambiguous, translate conservatively instead of guessing a diagnosis. Keep product names and clinical abbreviations unchanged when no safe translation is known. Return only the Simplified Chinese translation, without a label, markdown, quotation marks, or commentary.""",
    "en": """You are a meticulous medical translator for information retrieval. Translate the Vietnamese patient query literally into English. Preserve every symptom, body part, disease or condition, medicine or product name, test, Arabic numeral, unit, age, duration, pregnancy detail, uncertainty, comparison, relationship, and negation. Do not diagnose, answer, explain, summarize, normalize, expand, or omit information. If wording is ambiguous, translate conservatively instead of guessing a diagnosis. Keep product names and clinical abbreviations unchanged when no safe translation is known. Return only the English translation, without a label, markdown, quotation marks, or commentary.""",
}


def clean(value):
    return SPACE_RE.sub(" ", str(value)).strip()


def normalized_numbers(value):
    return Counter(match.group(0).replace(",", ".").lstrip("+") for match in NUMBER_RE.finditer(str(value)))


def medical_tokens(value):
    return {token.casefold() for token in ASCII_MEDICAL_RE.findall(str(value)) if len(token) >= 2}


def validate(source, translation, target, reached_limit=False):
    source, translation = clean(source), clean(translation)
    errors, warnings = [], []
    if not translation:
        errors.append("empty")
    ratio = len(translation) / max(len(source), 1)
    if translation and ratio < 0.16:
        errors.append("too_short")
    if ratio > 5.0:
        errors.append("too_long")
    source_numbers = normalized_numbers(source)
    target_numbers = normalized_numbers(translation)
    missing_numbers = source_numbers - target_numbers
    if missing_numbers:
        errors.append("missing_numbers:" + ",".join(sorted(missing_numbers.elements())))
    source_tokens = medical_tokens(source)
    target_tokens = medical_tokens(translation)
    missing_tokens = source_tokens - target_tokens
    if missing_tokens:
        warnings.append("missing_ascii_medical:" + ",".join(sorted(missing_tokens)))
    if target == "en" and translation and not LATIN_RE.search(translation):
        errors.append("missing_latin_script")
    if target == "zh" and translation and not CJK_RE.search(translation):
        errors.append("missing_cjk_script")
    if reached_limit:
        errors.append("generation_reached_token_limit")
    return errors, warnings, {
        "number_recall": 1.0 if not source_numbers else 1.0 - sum(missing_numbers.values()) / sum(source_numbers.values()),
        "medical_token_recall": 1.0 if not source_tokens else len(source_tokens & target_tokens) / len(source_tokens),
        "length_ratio": ratio,
    }


def clean_generation(value):
    value = str(value).strip()
    value = re.sub(r"<think>.*?</think>", "", value, flags=re.S | re.I).strip()
    value = re.sub(r"^```(?:text|markdown|json)?\s*", "", value, flags=re.I)
    value = re.sub(r"\s*```$", "", value).strip()
    value = value.strip('"').strip("'").strip()
    value = re.sub(r"^(?:Translation|English translation|Simplified Chinese translation|Chinese translation)\s*:\s*", "", value, flags=re.I)
    return clean(value)


def build_prompt(tokenizer, source, target, retry_errors=None):
    system = SYSTEM_PROMPTS[target]
    if retry_errors:
        system += " Previous output failed these structural checks: " + "; ".join(retry_errors) + ". Correct them and preserve all Arabic numerals exactly."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "<vietnamese_query>\n" + source + "\n</vietnamese_query>"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def token_budget(tokenizer, source, target, retry=False):
    source_tokens = len(tokenizer(source, add_special_tokens=False)["input_ids"])
    factor = 2.1 if target == "en" else 1.7
    budget = max(192, int(source_tokens * factor) + 128)
    if retry:
        budget = max(512, budget * 2)
    return min(1536 if retry else 1024, budget)


def generate_batch(model, tokenizer, records, target, retry_errors=None):
    prompts = [build_prompt(tokenizer, str(row.query), target, retry_errors) for row in records]
    encoded = tokenizer(prompts, padding=True, truncation=False, return_tensors="pt").to("cuda:0")
    if encoded["input_ids"].shape[1] > 2048:
        raise RuntimeError(f"Prompt exceeds 2048 tokens for IDs {[int(row.id) for row in records]}")
    max_new = max(token_budget(tokenizer, str(row.query), target, bool(retry_errors)) for row in records)
    input_width = int(encoded["input_ids"].shape[1])
    started = time.time()
    try:
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new,
                do_sample=False,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        del encoded
        gc.collect()
        torch.cuda.empty_cache()
        if len(records) == 1:
            raise
        middle = len(records) // 2
        return generate_batch(model, tokenizer, records[:middle], target, retry_errors) + generate_batch(model, tokenizer, records[middle:], target, retry_errors)
    elapsed = time.time() - started
    results = []
    for row, output in zip(records, generated):
        new_ids = output[input_width:].detach().cpu().tolist()
        eos_position = next((index for index, token in enumerate(new_ids) if token == tokenizer.eos_token_id), None)
        reached_limit = eos_position is None and len(new_ids) >= max_new
        effective_ids = new_ids if eos_position is None else new_ids[:eos_position]
        raw = tokenizer.decode(effective_ids, skip_special_tokens=True).strip()
        translation = clean_generation(raw)
        errors, warnings, metrics = validate(str(row.query), translation, target, reached_limit)
        results.append({
            "id": int(row.id), "query": str(row.query), f"query_{target}": translation,
            f"status_{target}": "ok" if not errors else "needs_review",
            f"validation_errors_{target}": json.dumps(errors, ensure_ascii=False),
            f"validation_warnings_{target}": json.dumps(warnings, ensure_ascii=False),
            f"number_recall_{target}": metrics["number_recall"],
            f"medical_token_recall_{target}": metrics["medical_token_recall"],
            f"length_ratio_{target}": metrics["length_ratio"],
            f"raw_output_{target}": raw,
            f"generated_tokens_{target}": len(effective_ids),
            f"max_new_tokens_{target}": max_new,
            f"reached_token_limit_{target}": bool(reached_limit),
            f"elapsed_seconds_{target}": elapsed / max(len(records), 1),
            f"attempt_{target}": 2 if retry_errors else 1,
        })
    del encoded, generated
    return results


def save_checkpoint(path, rows, target):
    frame = pd.DataFrame(rows).sort_values("id").drop_duplicates("id", keep="last")
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False, compression="snappy")
    os.replace(temp, path)
    return frame


def run_target(model, tokenizer, source_path, output_path, target, batch_size, checkpoint_every):
    source = pd.read_parquet(source_path, columns=["id", "query"]).sort_values("id").reset_index(drop=True)
    if source.id.duplicated().any():
        raise RuntimeError(f"Duplicate IDs in {source_path}")
    existing = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame()
    if not existing.empty:
        if existing.id.duplicated().any():
            raise RuntimeError(f"Duplicate IDs in checkpoint {output_path}")
        check = source.merge(existing[["id", "query"]], on="id", suffixes=("_source", "_checkpoint"))
        if not check.query_source.eq(check.query_checkpoint).all():
            raise RuntimeError(f"Source/checkpoint mismatch in {output_path}")
    completed = set() if existing.empty else set(map(int, existing.id))
    pending = source.loc[~source.id.isin(completed)].copy()
    pending["length"] = pending["query"].astype(str).str.len()
    pending = pending.sort_values(["length", "id"]).drop(columns="length")
    rows = [] if existing.empty else existing.to_dict("records")
    records = list(pending.itertuples(index=False))
    started = time.time()
    for offset in range(0, len(records), batch_size):
        batch = records[offset:offset + batch_size]
        first_pass = generate_batch(model, tokenizer, batch, target)
        for row, result in zip(batch, first_pass):
            errors = json.loads(result[f"validation_errors_{target}"])
            if errors:
                result = generate_batch(model, tokenizer, [row], target, retry_errors=errors)[0]
            rows.append(result)
        done_now = offset + len(batch)
        if done_now % checkpoint_every < batch_size or done_now == len(records):
            frame = save_checkpoint(output_path, rows, target)
            elapsed = time.time() - started
            total_done = len(frame)
            speed = done_now / max(elapsed, 1e-6)
            eta = (len(records) - done_now) / max(speed, 1e-6) / 60
            print(f"{target} {total_done}/{len(source)} | session {done_now}/{len(records)} | {speed:.3f} q/s | ETA {eta:.1f} min", flush=True)
    if not output_path.exists():
        save_checkpoint(output_path, rows, target)
    final = pd.read_parquet(output_path)
    if set(map(int, final.id)) != set(map(int, source.id)):
        raise RuntimeError(f"Incomplete {target} checkpoint: {len(final)}/{len(source)}")
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", required=True)
    parser.add_argument("--rescue", required=True)
    parser.add_argument("--zh-output", required=True)
    parser.add_argument("--en-output", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    args = parser.parse_args()

    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    ).eval()
    model.config.use_cache = True
    print({
        "gpu": torch.cuda.get_device_name(0),
        "allocated_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        "shard": args.shard,
        "batch_size": args.batch_size,
    }, flush=True)
    run_target(model, tokenizer, Path(args.shard), Path(args.zh_output), "zh", args.batch_size, args.checkpoint_every)
    rescue = pd.read_parquet(args.rescue)
    if len(rescue):
        run_target(model, tokenizer, Path(args.rescue), Path(args.en_output), "en", args.batch_size, args.checkpoint_every)
    else:
        pd.DataFrame(columns=["id", "query", "query_en"]).to_parquet(args.en_output, index=False)
    print("WORKER_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
'''.strip()


CELLS = [
    markdown(r"""
# Production query translation — VinAI EN + Qwen3-14B ZH on 2×T4

Notebook này dịch **đủ 1.200 official queries** và tạo artifact versioned; không ghi đè dữ liệu BTC và không tạo submission trực tiếp.

- EN mặc định: `vinai/vinai-translate-vi2en-v2`, model thắng benchmark MedEV/validator.
- ZH mặc định: `Qwen/Qwen3-14B-AWQ`, model 14,8B đã qua review định tính tốt nhất cho query y khoa khó.
- Qwen EN chỉ rescue những query mà VinAI fail structural validator.
- Hai tiến trình Qwen độc lập, mỗi tiến trình giữ một bản AWQ trên một T4 và xử lý shard riêng. Đây là data parallel thật, không phải một model bị chia layer khiến hai GPU chờ nhau.
- Output Qwen là plain text một ngôn ngữ/lần, có dynamic token budget và retry. Thiết kế này loại lỗi JSON bị cắt ở query 219 trong benchmark.

## Kaggle setup

1. Attach private dataset chứa đúng `data/raw/btc_provided/query.parquet` hoặc một file `query.parquet` cùng SHA-256.
2. Chọn accelerator **GPU T4 ×2**, bật Internet, thêm Kaggle Secret `HF_TOKEN`.
3. Run All. Notebook checkpoint sau mỗi 20 query/worker; nếu session chưa bị xóa, chạy lại sẽ resume.
4. Tải duy nhất `/kaggle/working/translation_production_v1.zip`.

Notebook pin đúng model revisions đã benchmark, kiểm SHA input, đủ ID, duplicate, empty/truncation, ZIP CRC và hashes. Model cache bị xóa ở cuối để Kaggle output không chứa 10+ GB weights.
"""),
    code(r"""
%pip install -q --no-cache-dir "transformers==4.51.3" "accelerate==1.6.0" "autoawq==0.2.9" "sentencepiece>=0.2,<0.3" "huggingface-hub==0.30.2" "pyarrow>=18,<21" "pandas>=2.2,<3"
"""),
    code(r"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/kaggle/working/hf_cache")

import numpy as np
import pandas as pd
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except Exception as exc:
    print(f"HF_TOKEN unavailable; public anonymous download will be used ({type(exc).__name__}).")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    print("Hugging Face authentication enabled via Kaggle Secret HF_TOKEN.")

WORK = Path("/kaggle/working/translation_production_v1")
WORK.mkdir(parents=True, exist_ok=True)
HF_CACHE = Path(os.environ["HF_HOME"])
HF_CACHE.mkdir(parents=True, exist_ok=True)

QUERY_PATH_OVERRIDE = ""
EXPECTED_QUERY_SHA256 = "ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08"
VINAI_REPO = "vinai/vinai-translate-vi2en-v2"
VINAI_REVISION = "ae7baa85da07dbe8e23ac26a9f5ef560c17e2138"
QWEN_REPO = "Qwen/Qwen3-14B-AWQ"
QWEN_REVISION = "31c69efc29464b6bb0aee1398b5a7b50a99340c3"
QWEN_PARAMETERS_B = 14.8
QWEN_WORKERS = 2
QWEN_BATCH_SIZE = 2
CHECKPOINT_EVERY = 20
DELETE_MODEL_CACHE_AT_END = True

if QWEN_PARAMETERS_B > 15:
    raise RuntimeError("Competition rule violation: Qwen model exceeds 15B")
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(f"Select Kaggle GPU T4 x2; detected {torch.cuda.device_count()} CUDA devices")
gpu_info = [{"name": torch.cuda.get_device_name(i), "memory_gb": round(torch.cuda.get_device_properties(i).total_memory/1e9, 2)} for i in range(2)]
if any(item["memory_gb"] < 14 for item in gpu_info):
    raise RuntimeError(f"Each Qwen worker needs a >=14 GB GPU; detected {gpu_info}")

if QUERY_PATH_OVERRIDE:
    QUERY_PATH = Path(QUERY_PATH_OVERRIDE)
else:
    matches = sorted(Path("/kaggle/input").rglob("query.parquet"))
    if len(matches) != 1:
        raise RuntimeError(f"Attach exactly one query.parquet or set QUERY_PATH_OVERRIDE; found {matches}")
    QUERY_PATH = matches[0]

print({
    "query_path": str(QUERY_PATH), "gpus": gpu_info,
    "disk_free_gb": round(shutil.disk_usage("/kaggle/working").free/1e9, 1),
    "qwen_workers": QWEN_WORKERS, "qwen_batch_size_each": QWEN_BATCH_SIZE,
})
"""),
    code(r"""
NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)(?!\w)")
ASCII_MEDICAL_RE = re.compile(r"\b(?=[A-Za-z0-9./+-]*[A-Za-z])(?=[A-Za-z0-9./+-]*(?:\d|[A-Z]))[A-Za-z0-9]+(?:[./+-][A-Za-z0-9]+)*\b")
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")
SPACE_RE = re.compile(r"\s+")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()

def clean(value: str) -> str:
    return SPACE_RE.sub(" ", str(value)).strip()

def normalized_numbers(value: str) -> Counter[str]:
    return Counter(match.group(0).replace(",", ".").lstrip("+") for match in NUMBER_RE.finditer(str(value)))

def medical_tokens(value: str) -> set[str]:
    return {token.casefold() for token in ASCII_MEDICAL_RE.findall(str(value)) if len(token) >= 2}

def validate_translation(source: str, target: str, language: str) -> tuple[list[str], list[str], dict]:
    source, target = clean(source), clean(target)
    errors, warnings = [], []
    if not target:
        errors.append("empty")
    ratio = len(target) / max(len(source), 1)
    if target and ratio < 0.16:
        errors.append("too_short")
    if ratio > 5.0:
        errors.append("too_long")
    source_numbers, target_numbers = normalized_numbers(source), normalized_numbers(target)
    missing_numbers = source_numbers - target_numbers
    if missing_numbers:
        errors.append("missing_numbers:" + ",".join(sorted(missing_numbers.elements())))
    source_tokens, target_tokens = medical_tokens(source), medical_tokens(target)
    missing_tokens = source_tokens - target_tokens
    if missing_tokens:
        warnings.append("missing_ascii_medical:" + ",".join(sorted(missing_tokens)))
    if language == "en" and target and not LATIN_RE.search(target):
        errors.append("missing_latin_script")
    if language == "zh" and target and not CJK_RE.search(target):
        errors.append("missing_cjk_script")
    return errors, warnings, {
        "number_recall": 1.0 if not source_numbers else 1.0 - sum(missing_numbers.values()) / sum(source_numbers.values()),
        "medical_token_recall": 1.0 if not source_tokens else len(source_tokens & target_tokens) / len(source_tokens),
        "length_ratio": ratio,
    }

queries = pd.read_parquet(QUERY_PATH, columns=["id", "query"]).sort_values("id").reset_index(drop=True)
if len(queries) != 1200 or queries.id.nunique() != 1200 or queries["query"].fillna("").str.strip().eq("").any():
    raise RuntimeError("Expected exactly 1,200 unique non-empty official queries")
query_sha256 = sha256_file(QUERY_PATH)
if query_sha256 != EXPECTED_QUERY_SHA256:
    raise RuntimeError(f"Official query SHA mismatch: {query_sha256} != {EXPECTED_QUERY_SHA256}")
queries.to_parquet(WORK/"official_queries.parquet", index=False, compression="snappy")
print({"query_rows": len(queries), "id_min": int(queries.id.min()), "id_max": int(queries.id.max()), "query_sha256": query_sha256})
"""),
    code(r"""
# Stage 1: VinAI v2 translates all 1,200 queries to English on GPU 0.
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

EN_PATH = WORK/"vinai_en_checkpoint.parquet"
existing_en = pd.read_parquet(EN_PATH) if EN_PATH.exists() else pd.DataFrame()
if not existing_en.empty:
    if existing_en.id.duplicated().any():
        raise RuntimeError("Duplicate IDs in VinAI checkpoint")
    joined = queries.merge(existing_en[["id", "query"]], on="id", suffixes=("_source", "_checkpoint"))
    if not joined.query_source.eq(joined.query_checkpoint).all():
        raise RuntimeError("VinAI checkpoint source mismatch")

completed_en = set() if existing_en.empty else set(map(int, existing_en.id))
pending_en = queries.loc[~queries.id.isin(completed_en)].copy()
en_rows = [] if existing_en.empty else existing_en.to_dict("records")

if len(pending_en):
    tokenizer = AutoTokenizer.from_pretrained(VINAI_REPO, revision=VINAI_REVISION, src_lang="vi_VN", cache_dir=HF_CACHE, token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        VINAI_REPO, revision=VINAI_REVISION, torch_dtype=torch.float16,
        low_cpu_mem_usage=True, cache_dir=HF_CACHE, token=HF_TOKEN,
    ).to("cuda:0").eval()
    input_lengths = [len(x) for x in tokenizer(pending_en["query"].astype(str).tolist(), add_special_tokens=True, truncation=False)["input_ids"]]
    if max(input_lengths, default=0) > 1024:
        raise RuntimeError(f"VinAI input would be truncated; maximum is {max(input_lengths)} tokens")
    start_id = tokenizer.lang_code_to_id["en_XX"] if hasattr(tokenizer, "lang_code_to_id") else tokenizer.convert_tokens_to_ids("en_XX")
    batch_size = 16
    values = list(pending_en.itertuples(index=False))
    started = time.time()
    for offset in range(0, len(values), batch_size):
        batch = values[offset:offset+batch_size]
        encoded = tokenizer([str(row.query) for row in batch], padding=True, truncation=False, return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            generated = model.generate(
                **encoded, decoder_start_token_id=start_id, num_beams=5,
                max_new_tokens=512, early_stopping=True,
            )
        translations = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for row, translation in zip(batch, translations):
            translation = clean(translation)
            errors, warnings, metrics = validate_translation(str(row.query), translation, "en")
            en_rows.append({
                "id": int(row.id), "query": str(row.query), "query_en_vinai": translation,
                "status_en_vinai": "ok" if not errors else "needs_review",
                "validation_errors_en_vinai": json.dumps(errors, ensure_ascii=False),
                "validation_warnings_en_vinai": json.dumps(warnings, ensure_ascii=False),
                **{f"{key}_en_vinai": value for key, value in metrics.items()},
            })
        if (offset + len(batch)) % 80 == 0 or offset + len(batch) == len(values):
            frame = pd.DataFrame(en_rows).sort_values("id").drop_duplicates("id", keep="last")
            temp = EN_PATH.with_suffix(".tmp.parquet")
            frame.to_parquet(temp, index=False, compression="snappy")
            os.replace(temp, EN_PATH)
            elapsed = time.time() - started
            done = offset + len(batch)
            print(f"VinAI session {done}/{len(values)} | {done/max(elapsed,1e-6):.2f} q/s", flush=True)
    del model, tokenizer, encoded, generated
    gc.collect(); torch.cuda.empty_cache()

vinai = pd.read_parquet(EN_PATH).sort_values("id").reset_index(drop=True)
if set(map(int, vinai.id)) != set(map(int, queries.id)):
    raise RuntimeError(f"VinAI stage incomplete: {len(vinai)}/1200")
rescue_ids = set(map(int, vinai.loc[vinai.status_en_vinai.ne("ok"), "id"]))
print({
    "vinai_rows": len(vinai), "vinai_ok": int(vinai.status_en_vinai.eq("ok").sum()),
    "qwen_en_rescue_rows": len(rescue_ids), "gpu0_allocated_after_release_gb": round(torch.cuda.memory_allocated(0)/1e9, 3),
})
"""),
    code(r"""
# Stage 2 preparation: deterministic balanced shards and one local Qwen snapshot shared by both workers.
from huggingface_hub import snapshot_download

loads = [0, 0]
shard_ids = [[], []]
for row in queries.assign(length=queries["query"].astype(str).str.len()).sort_values(["length", "id"], ascending=[False, True]).itertuples(index=False):
    worker = int(np.argmin(loads))
    shard_ids[worker].append(int(row.id))
    loads[worker] += int(row.length)

for worker in range(QWEN_WORKERS):
    shard = queries.loc[queries.id.isin(shard_ids[worker])].sort_values("id")
    rescue = shard.loc[shard.id.isin(rescue_ids)]
    shard.to_parquet(WORK/f"qwen_shard_{worker}.parquet", index=False, compression="snappy")
    rescue.to_parquet(WORK/f"qwen_en_rescue_shard_{worker}.parquet", index=False, compression="snappy")
    print({"worker": worker, "zh_rows": len(shard), "en_rescue_rows": len(rescue), "source_chars": loads[worker]})

def complete_checkpoint(path, expected_ids):
    if not path.exists():
        return False
    frame = pd.read_parquet(path, columns=["id"])
    return not frame.id.duplicated().any() and set(map(int, frame.id)) == set(map(int, expected_ids))

all_workers_complete = True
for worker in range(QWEN_WORKERS):
    zh_done = complete_checkpoint(WORK/f"qwen_zh_gpu{worker}.parquet", shard_ids[worker])
    rescue_expected = [qid for qid in shard_ids[worker] if qid in rescue_ids]
    en_done = complete_checkpoint(WORK/f"qwen_en_rescue_gpu{worker}.parquet", rescue_expected)
    all_workers_complete &= zh_done and en_done

if all_workers_complete:
    QWEN_SNAPSHOT = ""
    print("All Qwen checkpoints complete; model download is skipped.")
else:
    QWEN_SNAPSHOT = snapshot_download(
        repo_id=QWEN_REPO, revision=QWEN_REVISION,
        cache_dir=HF_CACHE, token=HF_TOKEN,
    )
    resolved = Path(QWEN_SNAPSHOT).name
    if resolved != QWEN_REVISION:
        raise RuntimeError(f"Resolved Qwen revision {resolved} != pinned {QWEN_REVISION}")
    print({"qwen_snapshot": QWEN_SNAPSHOT, "disk_free_gb": round(shutil.disk_usage('/kaggle/working').free/1e9, 1)})

WORKER_PATH = WORK/"qwen_translation_worker.py"
WORKER_PATH.write_text(QWEN_WORKER_SOURCE, encoding="utf-8")
print({"worker_script": str(WORKER_PATH), "worker_sha256": sha256_file(WORKER_PATH)})
"""),
    code(r"""
# Stage 2 execution: two independent Qwen replicas, one per T4.
processes = []
log_handles = []
if not all_workers_complete:
    for worker in range(QWEN_WORKERS):
        expected_zh = shard_ids[worker]
        expected_en = [qid for qid in shard_ids[worker] if qid in rescue_ids]
        zh_path = WORK/f"qwen_zh_gpu{worker}.parquet"
        en_path = WORK/f"qwen_en_rescue_gpu{worker}.parquet"
        if complete_checkpoint(zh_path, expected_zh) and complete_checkpoint(en_path, expected_en):
            print(f"worker {worker}: checkpoints already complete")
            continue
        log_path = WORK/f"qwen_worker_{worker}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker)
        env["TOKENIZERS_PARALLELISM"] = "false"
        command = [
            sys.executable, str(WORKER_PATH),
            "--shard", str(WORK/f"qwen_shard_{worker}.parquet"),
            "--rescue", str(WORK/f"qwen_en_rescue_shard_{worker}.parquet"),
            "--zh-output", str(zh_path),
            "--en-output", str(en_path),
            "--model-path", QWEN_SNAPSHOT,
            "--batch-size", str(QWEN_BATCH_SIZE),
            "--checkpoint-every", str(CHECKPOINT_EVERY),
        ]
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
        processes.append((worker, process, log_path))
        log_handles.append(log_handle)

    try:
        while any(process.poll() is None for _, process, _ in processes):
            status = []
            for worker, process, log_path in processes:
                last_line = ""
                if log_path.exists():
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    last_line = lines[-1] if lines else ""
                status.append({"gpu": worker, "returncode": process.poll(), "last": last_line[-180:]})
            print(datetime.now().strftime("%H:%M:%S"), status, flush=True)
            time.sleep(30)
    finally:
        for handle in log_handles:
            handle.close()

    failed = [(worker, process.returncode, log_path) for worker, process, log_path in processes if process.returncode != 0]
    if failed:
        for worker, returncode, log_path in failed:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            print(f"--- GPU {worker} failed with {returncode} ---\n" + "\n".join(tail))
        raise RuntimeError(f"Qwen worker failure: {[(worker, code) for worker, code, _ in failed]}")

for worker in range(QWEN_WORKERS):
    expected_en = [qid for qid in shard_ids[worker] if qid in rescue_ids]
    if not complete_checkpoint(WORK/f"qwen_zh_gpu{worker}.parquet", shard_ids[worker]):
        raise RuntimeError(f"Incomplete ZH checkpoint for worker {worker}")
    if not complete_checkpoint(WORK/f"qwen_en_rescue_gpu{worker}.parquet", expected_en):
        raise RuntimeError(f"Incomplete EN rescue checkpoint for worker {worker}")
print("Both Qwen workers completed.")
"""),
    code(r"""
# Merge, gate EN rescue, audit all 1,200 rows, and package only lightweight artifacts.
zh = pd.concat([pd.read_parquet(WORK/f"qwen_zh_gpu{worker}.parquet") for worker in range(QWEN_WORKERS)], ignore_index=True)
en_rescue_parts = [pd.read_parquet(WORK/f"qwen_en_rescue_gpu{worker}.parquet") for worker in range(QWEN_WORKERS)]
en_rescue = pd.concat(en_rescue_parts, ignore_index=True) if any(len(x) for x in en_rescue_parts) else pd.DataFrame(columns=["id", "query", "query_en"])

for name, frame, expected in [("zh", zh, set(map(int, queries.id))), ("en_rescue", en_rescue, rescue_ids)]:
    if frame.id.duplicated().any() or set(map(int, frame.id)) != expected:
        raise RuntimeError(f"{name} IDs incomplete or duplicated: rows={len(frame)}, unique={frame.id.nunique()}")
    if len(frame):
        check = queries.merge(frame[["id", "query"]], on="id", suffixes=("_source", "_translated"))
        if not check.query_source.eq(check.query_translated).all():
            raise RuntimeError(f"{name} source query mismatch")

audit = queries.merge(vinai, on=["id", "query"], validate="one_to_one")
audit = audit.merge(zh, on=["id", "query"], validate="one_to_one")
if len(en_rescue):
    rescue_columns = [column for column in en_rescue.columns if column not in {"query"}]
    audit = audit.merge(en_rescue[rescue_columns], on="id", how="left", validate="one_to_one", suffixes=("", "_qwen_rescue"))
else:
    audit["query_en"] = ""
    audit["status_en"] = ""

use_qwen_en = audit.status_en_vinai.ne("ok") & audit.get("status_en", pd.Series("", index=audit.index)).eq("ok")
audit["query_en_effective"] = audit["query_en_vinai"]
audit.loc[use_qwen_en, "query_en_effective"] = audit.loc[use_qwen_en, "query_en"]
audit["en_selected_model"] = np.where(use_qwen_en, "Qwen3-14B-AWQ-rescue", "VinAI-Translate-v2")
audit["status_en_effective"] = np.where(use_qwen_en, audit.get("status_en", ""), audit.status_en_vinai)

hard_zh = zh.query_zh.fillna("").str.strip().eq("") | ~zh.query_zh.fillna("").str.contains(CJK_RE)
hard_zh |= zh.reached_token_limit_zh.astype(bool)
if hard_zh.any():
    raise RuntimeError(f"Fatal ZH outputs remain for IDs {zh.loc[hard_zh, 'id'].astype(int).tolist()}")
if audit.query_en_effective.fillna("").str.strip().eq("").any():
    raise RuntimeError("Empty effective EN translation remains")

production = audit[["id", "query", "query_zh", "query_en_effective"]].rename(columns={"query_en_effective": "query_en"}).sort_values("id")
if len(production) != 1200 or production.id.nunique() != 1200:
    raise RuntimeError("Production output must contain exactly 1,200 unique IDs")

PRODUCTION_PATH = WORK/"query_translated_production_v1.parquet"
AUDIT_PATH = WORK/"translation_production_audit_v1.parquet"
REVIEW_PATH = WORK/"translation_needs_review_v1.csv"
MANIFEST_PATH = WORK/"translation_production_manifest_v1.json"
production.to_parquet(PRODUCTION_PATH, index=False, compression="snappy")
audit.to_parquet(AUDIT_PATH, index=False, compression="snappy")

needs_review = audit.loc[
    audit.status_en_effective != "ok"
].copy()
needs_review = pd.concat([needs_review, audit.loc[audit.status_zh != "ok"]], ignore_index=True).drop_duplicates("id").sort_values("id")
review_columns = [column for column in [
    "id", "query", "query_en_effective", "query_zh", "en_selected_model",
    "status_en_effective", "status_zh", "validation_errors_en_vinai",
    "validation_errors_en", "validation_errors_zh",
] if column in needs_review.columns]
needs_review[review_columns].to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

manifest = {
    "created_at": datetime.now().astimezone().isoformat(),
    "task": "production_vi_to_en_zh_query_translation",
    "query_rows": len(production), "query_sha256": query_sha256,
    "models": {
        "en_default": {"repo": VINAI_REPO, "revision": VINAI_REVISION, "parameters_b": 0.448, "license": "agpl-3.0"},
        "zh_and_en_rescue": {"repo": QWEN_REPO, "revision": QWEN_REVISION, "parameters_b": QWEN_PARAMETERS_B, "license": "apache-2.0"},
    },
    "execution": {"qwen_workers": QWEN_WORKERS, "qwen_batch_size_each": QWEN_BATCH_SIZE, "gpu_info": gpu_info},
    "qwen_worker_sha256": sha256_file(WORKER_PATH),
    "generation_policy": {
        "vinai_num_beams": 5, "vinai_max_new_tokens": 512,
        "qwen_do_sample": False, "qwen_dynamic_max_new_tokens": [192, 1024],
        "qwen_retry_max_new_tokens": 1536, "qwen_plain_text_single_language": True,
    },
    "counts": {
        "vinai_en_ok": int(audit.status_en_vinai.eq("ok").sum()),
        "qwen_en_rescue_requested": len(rescue_ids),
        "qwen_en_rescue_selected": int(use_qwen_en.sum()),
        "effective_en_needs_review": int((audit.status_en_effective != "ok").sum()),
        "qwen_zh_ok": int(audit.status_zh.eq("ok").sum()),
        "qwen_zh_needs_review": int((audit.status_zh != "ok").sum()),
        "manual_review_unique_ids": len(needs_review),
    },
    "artifacts": {},
}
for path in [PRODUCTION_PATH, AUDIT_PATH, REVIEW_PATH]:
    manifest["artifacts"][path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

ZIP_PATH = Path("/kaggle/working/translation_production_v1.zip")
with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for path in [PRODUCTION_PATH, AUDIT_PATH, REVIEW_PATH, MANIFEST_PATH]:
        archive.write(path, path.name)
    for worker in range(QWEN_WORKERS):
        log_path = WORK/f"qwen_worker_{worker}.log"
        if log_path.exists():
            archive.write(log_path, log_path.name)
with zipfile.ZipFile(ZIP_PATH) as archive:
    if archive.testzip() is not None:
        raise RuntimeError("Output ZIP CRC failed")

print(json.dumps({
    "zip_path": str(ZIP_PATH), "zip_bytes": ZIP_PATH.stat().st_size,
    "zip_sha256": sha256_file(ZIP_PATH), **manifest["counts"],
}, ensure_ascii=False, indent=2))
print(production.loc[production.id.isin([1, 2, 3, 42, 71, 219, 271])].to_string(index=False))
"""),
    code(r"""
# Delete model weights so Kaggle output contains the small production ZIP, not the downloaded checkpoints.
if DELETE_MODEL_CACHE_AT_END and HF_CACHE.exists():
    shutil.rmtree(HF_CACHE)
    print(f"Deleted model cache: {HF_CACHE}")
print({
    "download_this_file": "/kaggle/working/translation_production_v1.zip",
    "zip_exists": Path("/kaggle/working/translation_production_v1.zip").exists(),
    "remaining_working_gb": round(shutil.disk_usage("/kaggle/working").free/1e9, 1),
})
"""),
    markdown(r"""
## Sau khi chạy xong

Tải `translation_production_v1.zip` và gửi lại để local:

1. kiểm tra toàn bộ validator/audit và review các query còn cờ;
2. không ghi đè file dịch cũ;
3. rebuild riêng BM25 EN/ZH cache;
4. đo overlap/churn và số document membership thay đổi so với Exp122;
5. chỉ tạo leaderboard submission khi treatment đủ 1.200 query và thật sự khác control.
"""),
]


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = CELLS
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "kaggle": {"accelerator": "gpu", "internet": True},
    }
    notebook["nbformat"] = 4
    notebook["nbformat_minor"] = 5
    # Inject the worker as a Python string without relying on notebook magics.
    worker_literal = repr(QWEN_WORKER)
    notebook["cells"][5]["source"] = f"QWEN_WORKER_SOURCE = {worker_literal}\n\n" + notebook["cells"][5]["source"]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print({
        "notebook": str(OUTPUT),
        "cells": len(notebook["cells"]),
        "worker_sha256": hashlib.sha256(QWEN_WORKER.encode("utf-8")).hexdigest(),
    })


if __name__ == "__main__":
    main()
