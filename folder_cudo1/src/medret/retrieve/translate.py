"""Vietnamese query -> Simplified Chinese / English, for the cross-lingual channels (BM25 zh/en, dense zh/en).

Any OpenAI-compatible chat endpoint (FPT AI Marketplace now); the model is configuration, not code.
Competition rule: only open-weight models <= 15B params may be used anywhere in the final pipeline.
DEV_ONLY models (e.g. Qwen3.6-27B) may be used to measure the potential of the step during development,
but their outputs are tagged and must never reach a leaderboard submission (`assert_submittable`).
Translations are cached by sha1(model, prompt version, query).
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from medret.cache import KVCache, cache_key
from medret.rerank.fpt_api import RETRY_CODES, load_env

PROMPT_VERSION = "zh-v1"
DEV_ONLY = {"Qwen3.6-27B", "gemma-4-26B-A4B-it", "gemma-4-31B-it", "Llama-3.3-70B-Instruct", "gpt-oss-120b",
            "DeepSeek-V4-Flash", "FPT.AI-Qwen3.5-35B-A3B", "FPT.AI-gemma-4-26B-A4B-it", "gemma-3-27b-it"}

SYSTEM = (
    "You translate Vietnamese medical questions from patients into Simplified Chinese, phrased the way a patient "
    "would ask on a Chinese health Q&A site (有问必答, 39健康, 家庭医生). Keep every number, unit, dose, lab value "
    "and abbreviation exactly as written (SCC, HbA1c, HPV, AMH, NST, G6PD, mg, ng/mL). Translate disease, symptom, "
    "drug and test names into their standard Chinese medical terms; for a drug brand or Latin name with no common "
    "Chinese name, keep it as is. Do not answer, explain or add anything. Output only the translation."
)


PROMPT_VERSION_EN = "en-v1"
SYSTEM_EN = (
    "You translate Vietnamese medical questions from patients into English, phrased as a precise clinical question "
    "that would match the title or abstract of a relevant PubMed article. Keep every number, unit, dose, lab value "
    "and abbreviation exactly as written (SCC, HbA1c, HPV, AMH, NST, G6PD, mg, ng/mL). Use standard English "
    "medical terminology (MeSH-style names for diseases, symptoms, drugs and tests). Do not answer, explain or add "
    "anything. Output only the translation."
)


def assert_submittable(models: list[str]) -> None:
    bad = [m for m in models if m in DEV_ONLY]
    if bad:
        raise RuntimeError(f"{bad} exceed the competition limit (open weights, <= 15B); dev experiments only")


class Translator:
    def __init__(self, model: str, cache_path: Path | None = None, threads: int = 8, max_retries: int = 6,
                 timeout: float = 120, system: str = SYSTEM, prompt_version: str = PROMPT_VERSION,
                 max_tokens: int = 512):
        env = load_env()
        self.base, self.key = env["FPT_API_BASE"].rstrip("/"), env["FPT_API_KEY"]
        self.model, self.threads, self.max_retries, self.timeout = model, threads, max_retries, timeout
        self.system, self.prompt_version = system, prompt_version  # other prompts (dev queries) reuse the client
        self.max_tokens = max_tokens
        self.cache = KVCache(cache_path) if cache_path else None
        self.tokens = 0

    def _chat(self, text: str) -> str:
        body = json.dumps({
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": text}],
            "chat_template_kwargs": {"enable_thinking": False},  # Qwen3.x: no reasoning trace, ~5x faster
        }).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base}/v1/chat/completions", data=body, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    res = json.loads(r.read().decode("utf-8"))
                self.tokens += int(res.get("usage", {}).get("total_tokens", 0))
                out = (res["choices"][0]["message"].get("content") or "").strip()
                return re.sub(r"(?s)^<think>.*?</think>\s*", "", out)
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_CODES or attempt == self.max_retries:
                    raise RuntimeError(f"chat HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries:
                    raise
            time.sleep(min(60, 2**attempt))
        raise AssertionError("unreachable")

    def translate(self, queries: list[str]) -> list[str]:
        keys = [cache_key("translate", self.model, self.prompt_version, q) for q in queries]
        hit = self.cache.get_many(keys) if self.cache else {}
        miss = [i for i, k in enumerate(keys) if k not in hit]
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            new = list(pool.map(lambda i: self._chat(queries[i]), miss))
        if self.cache and miss:
            self.cache.put_many((keys[i], t.encode("utf-8")) for i, t in zip(miss, new, strict=True))
        got = dict(zip(miss, new, strict=True))
        return [got[i] if i in got else hit[k].decode("utf-8") for i, k in enumerate(keys)]
