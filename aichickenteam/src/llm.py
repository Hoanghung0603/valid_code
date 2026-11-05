"""Thin shared wrapper around a local open-weight chat LLM (HF transformers).

Usage:
    from src.llm import get_llm
    llm = get_llm()                       # loads once (default model below, 4-bit)
    outs = llm.generate(["prompt 1", [{"role": "user", "content": "..."}]],
                        max_new_tokens=512, json=True)
    llm.unload()                          # free VRAM when done

Prompts may be a plain string (wrapped as a single user turn), or a list of chat
messages (``[{"role": "system"|"user"|"assistant", "content": str}, ...]``).
Decoding is greedy (deterministic). Generation is batched; prompts are sorted by
length internally and outputs returned in the original order.

Default model: Qwen3-4B-Instruct-2507 (pre-quantized bnb 4-bit). See docs/MODELS.md.
"""
from __future__ import annotations

import copy
import gc
import json as _json
import logging
import re
import time
from typing import Any, Iterable

log = logging.getLogger(__name__)

DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-bnb-4bit"

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less fragmentation

_SINGLETONS: dict[tuple, "LLM"] = {}


class LLM:
    def __init__(self, model_id: str = DEFAULT_MODEL, quant: str = "4bit",
                 max_input_tokens: int = 3072, dtype: str = "bfloat16",
                 device_map: str | dict = "cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_input_tokens = max_input_tokens
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw: dict[str, Any] = {"device_map": device_map,
                              "dtype": getattr(torch, dtype)}
        prequant = "bnb-4bit" in model_id or "bnb_4bit" in model_id
        if quant == "4bit" and not prequant:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                bnb_4bit_use_double_quant=True)
        elif quant == "8bit" and not prequant:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        except TypeError:  # older transformers: torch_dtype instead of dtype
            kw["torch_dtype"] = kw.pop("dtype")
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.stats = {"batches": 0, "steps": 0, "gen_tokens": 0}
        log.info("Loaded %s in %.1fs", model_id, time.time() - t0)

    # ------------------------------------------------------------------ utils
    def _render(self, p: str | list[dict]) -> str:
        msgs = [{"role": "user", "content": p}] if isinstance(p, str) else p
        try:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)  # no-op for non-hybrid templates
        except TypeError:
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False).input_ids)

    # --------------------------------------------------------------- generate
    def generate(self, prompts: Iterable[str | list[dict]], max_new_tokens: int = 512,
                 json: bool = False, batch_size: int = 8,
                 max_batch_tokens: int = 12000, prefill: str = "",
                 prefix_cache: bool = True) -> list[str]:
        """Greedy batched generation. Returns decoded completions (prompt removed).

        json=True: prefills the assistant turn with ``{`` (plus ``prefill``) so the
        model starts a JSON object directly; the returned text includes that ``{``.
        max_batch_tokens bounds batch_size * (longest prompt + max_new_tokens) to
        cap KV-cache memory.
        prefix_cache=True: the longest token prefix shared by all prompts (e.g. a
        system prompt + few-shot example) is prefilled once and its KV cache reused
        for every batch (bnb prefill is compute-bound, so this saves a lot).
        """
        import torch

        prompts = list(prompts)
        if not prompts:
            return []
        start = (prefill or "") + ("{" if json else "")
        ids = [self.tok(self._render(p) + start, add_special_tokens=False).input_ids
               for p in prompts]
        ids = [x[-self.max_input_tokens:] for x in ids]
        P = 0
        if prefix_cache and len(ids) > 1:
            P = _common_prefix_len(ids)
            P = max(0, min(P, min(len(x) for x in ids) - 1))  # keep >=1 suffix token
        pcache = self._prefix_kv(ids[0][:P]) if P >= 16 else None
        if pcache is None:
            P = 0
        order = sorted(range(len(ids)), key=lambda i: -len(ids[i]))
        outs: list[str] = [""] * len(ids)
        pad = self.tok.pad_token_id
        i = 0
        while i < len(order):
            longest = len(ids[order[i]])
            bs = max(1, min(batch_size, max_batch_tokens // (longest + max_new_tokens)))
            idx = order[i:i + bs]
            i += bs
            L = max(len(ids[j]) for j in idx) - P
            inp = torch.full((len(idx), P + L), pad, dtype=torch.long)
            att = torch.zeros((len(idx), P + L), dtype=torch.long)
            for r, j in enumerate(idx):
                suf = ids[j][P:]
                inp[r, :P] = torch.tensor(ids[j][:P])
                att[r, :P] = 1
                inp[r, P + L - len(suf):] = torch.tensor(suf)
                att[r, P + L - len(suf):] = 1
            inp, att = inp.to(self.device), att.to(self.device)
            kw: dict[str, Any] = {}
            if pcache is not None:
                c = copy.deepcopy(pcache)
                if len(idx) > 1:
                    c.batch_repeat_interleave(len(idx))
                kw["past_key_values"] = c
            try:
                with torch.inference_mode():
                    gen = self.model.generate(
                        input_ids=inp, attention_mask=att, max_new_tokens=max_new_tokens,
                        do_sample=False, temperature=None, top_p=None, top_k=None,
                        pad_token_id=pad, **kw)
            except torch.cuda.OutOfMemoryError:
                kw.clear()
                torch.cuda.empty_cache()
                if len(idx) == 1:
                    raise
                log.warning("OOM with batch %d, retrying one by one", len(idx))
                for j in idx:
                    outs[j] = self.generate([prompts[j]], max_new_tokens, json,
                                            1, max_batch_tokens, prefill, False)[0]
                continue
            new = gen[:, P + L:]
            self.stats["batches"] += 1
            self.stats["steps"] += new.shape[1]
            self.stats["gen_tokens"] += int((new != pad).sum())
            for j, row in zip(idx, new):
                outs[j] = start + self.tok.decode(row, skip_special_tokens=True)
            del inp, att, gen, new, kw
        return outs

    def _prefix_kv(self, prefix_ids: list[int]):
        """KV cache for a token prefix (memoized for the last prefix)."""
        import torch
        from transformers import DynamicCache
        key = tuple(prefix_ids)
        if getattr(self, "_pkey", None) == key:
            return self._pval
        try:
            c = DynamicCache()
            x = torch.tensor([prefix_ids], device=self.device)
            with torch.inference_mode():
                self.model(input_ids=x, attention_mask=torch.ones_like(x),
                           past_key_values=c, use_cache=True)
        except Exception as e:  # pragma: no cover
            log.warning("prefix cache disabled: %s", e)
            return None
        self._pkey, self._pval = key, c
        return c

    def unload(self) -> None:
        import torch
        for k, v in list(_SINGLETONS.items()):
            if v is self:
                del _SINGLETONS[k]
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def _common_prefix_len(seqs: list[list[int]]) -> int:
    first = seqs[0]
    n = min(len(x) for x in seqs)
    k = 0
    while k < n and all(x[k] == first[k] for x in seqs):
        k += 1
    return k


def get_llm(model_id: str = DEFAULT_MODEL, quant: str = "4bit", **kw) -> LLM:
    """Process-wide singleton per (model_id, quant)."""
    key = (model_id, quant)
    if key not in _SINGLETONS:
        _SINGLETONS[key] = LLM(model_id, quant=quant, **kw)
    return _SINGLETONS[key]


def generate(prompts, max_new_tokens: int = 512, json: bool = True, **kw) -> list[str]:
    """Module-level convenience: get_llm() default model, then LLM.generate."""
    return get_llm().generate(prompts, max_new_tokens=max_new_tokens, json=json, **kw)


# ----------------------------------------------------------------- JSON parse
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def parse_json(text: str) -> Any | None:
    """Robustly extract the first JSON object from LLM output. None if impossible."""
    if not text:
        return None
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    s = text.find("{")
    if s < 0:
        return None
    text = text[s:]
    # 1) balanced-brace scan (string-aware)
    depth, in_str, esc = 0, False, False
    for k, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                cand = text[:k + 1]
                for c in (cand, re.sub(r",\s*([}\]])", r"\1", cand)):
                    try:
                        return _json.loads(c)
                    except Exception:
                        pass
                break
    # 2) truncated output: try to close open structures
    return _repair_truncated(text)


def _repair_truncated(text: str) -> Any | None:
    stack, in_str, esc, last_ok = [], False, False, 0
    for k, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack:
                stack.pop()
        if not in_str and ch in ",[{" :
            last_ok = k
    # cut at last structural boundary, drop dangling comma, close everything
    body = text[:last_ok + 1].rstrip().rstrip(",")
    stack2, in_str, esc = [], False, False
    for ch in body:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack2.append("}" if ch == "{" else "]")
        elif ch in "}]" and stack2:
            stack2.pop()
    cand = body + "".join(reversed(stack2))
    cand = re.sub(r",\s*([}\]])", r"\1", cand)
    cand = re.sub(r":\s*([}\]])", r": null\1", cand)
    try:
        return _json.loads(cand)
    except Exception:
        return None
