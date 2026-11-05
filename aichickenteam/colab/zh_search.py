"""
colab/zh_search.py - Search Chinese docs on Colab
"""
import argparse
import json
import logging
import math
import os
import re
import sys
import time
from glob import glob
from collections import defaultdict
import heapq
import psutil

try:
    import jieba
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "jieba"])
    import jieba

import numpy as np
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# Copied from chunk_v2.py
RE_ZH_ANSWER = re.compile(
    r"(医生回复|病情分析|指导意见|回答|总结|建议|治疗方案|临床表现|诊断|病因)",
    re.IGNORECASE,
)

ZH_STOP = {"的", "了", "和", "是", "就", "都", "而", "及", "与", "着", "或", "一个", "没有", "我们", "你们", "他们", "她", "它", "这", "那", "有", "在", "等", "也", "、", "，", "。", "！", "？", "：", "“", "”", "（", "）", "《", "》"}

def tokenize(text):
    return [w for w in jieba.cut(text) if w.strip() and w not in ZH_STOP]

def paragraphs(text, min_chars=50):
    spans = []
    pos = 0
    for part in re.split(r"\n\s*\n", text):
        s = text.find(part, pos)
        if s == -1: continue
        e = s + len(part)
        pos = e
        if part.strip(): spans.append([s, e])
    if len(spans) <= 1 and text.count("\n") > 2:
        spans = []
        pos = 0
        for part in text.split("\n"):
            s = text.find(part, pos)
            if s == -1: continue
            e = s + len(part)
            pos = e
            if part.strip(): spans.append([s, e])
    merged = []
    cur = None
    for s, e in spans:
        if cur is None: cur = [s, e]
        else: cur[1] = e
        if cur[1] - cur[0] >= min_chars:
            merged.append((cur[0], cur[1]))
            cur = None
    if cur is not None:
        if merged and cur[1] - cur[0] < min_chars:
            merged[-1] = (merged[-1][0], cur[1])
        else:
            merged.append((cur[0], cur[1]))
    return merged

def get_query_terms(query_str, plan, plan_w=0.6):
    weights = {}
    if plan:
        zh_rewrites = plan.get("zh", [])
        zh_kw = plan.get("keywords", {}).get("zh", [])
        for k in zh_kw:
            for t in tokenize(k):
                weights[t] = max(weights.get(t, 0.0), 1.5 + plan_w)
        for r in zh_rewrites[:4]:
            for t in tokenize(r):
                weights[t] = max(weights.get(t, 0.0), 1.0)
    if not weights:
        for t in tokenize(query_str):
            weights[t] = 1.0
    return weights

def select_chunks(query_str, plan, doc_text, target_chars=750):
    if not doc_text or not doc_text.strip(): return []
    clean_text = doc_text.strip()
    s_clean = doc_text.find(clean_text)
    if s_clean == -1: s_clean = 0; clean_text = doc_text
    if len(clean_text) <= target_chars:
        return [doc_text[s_clean: s_clean + len(clean_text)]]
    paras = paragraphs(doc_text, min_chars=50)
    if not paras:
        chunk_end = min(s_clean + target_chars, len(doc_text))
        return [doc_text[s_clean:chunk_end]]
    
    n_p = len(paras)
    q_weights = get_query_terms(query_str, plan)
    
    scores = []
    for i, (s, e) in enumerate(paras):
        p_toks = tokenize(doc_text[s:e])
        if not p_toks:
            scores.append(0.0)
            continue
        m_val = sum(q_weights.get(t, 0.0) for t in set(p_toks))
        base = m_val / (max(len(p_toks), 1) ** 0.3)
        bonus = 0.0
        if i == 0: bonus += 0.25
        elif i == 1 and n_p > 2: bonus += 0.125
        if RE_ZH_ANSWER.search(doc_text[s:e]): bonus += 0.25
        scores.append(base + bonus)
        
    def expand_window(center_idx):
        lo = hi = center_idx
        while paras[hi][1] - paras[lo][0] < target_chars and (lo > 0 or hi < n_p - 1):
            left_sc = scores[lo - 1] if lo > 0 else -1.0
            right_sc = scores[hi + 1] if hi < n_p - 1 else -1.0
            if lo == 1 and paras[lo - 1][0] == 0: left_sc += 0.1
            if right_sc >= left_sc: hi += 1
            else: lo -= 1
        return lo, hi
        
    best1 = max(range(n_p), key=lambda i: scores[i])
    lo1, hi1 = expand_window(best1)
    c_span = (paras[lo1][0], paras[hi1][1])
    return [doc_text[c_span[0]:c_span[1]]]

def hash_word(word, vocab_size):
    return hash(word) % vocab_size

def print_mem():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    logging.info(f"RAM Usage: {mem_info.rss / 1024 / 1024:.2f} MB")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", required=True, help="Directory containing parquet files")
    parser.add_argument("--plans", required=True, help="Path to query_plans.jsonl")
    parser.add_argument("--out", required=True, help="Path to output jsonl")
    parser.add_argument("--k", type=int, default=100, help="Top k to keep per query")
    parser.add_argument("--vocab-bits", type=int, default=24, help="Bits for hashed vocab (24 = 16M)")
    args = parser.parse_args()

    start_time = time.time()
    
    # Load queries
    queries = []
    with open(args.plans, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            plan = json.loads(line)
            # Truy vấn = plan["zh"] + plan["keywords"]["zh"] (+ title boost).
            q_terms = set()
            if "zh" in plan:
                for rewrite in plan["zh"]:
                    q_terms.update(tokenize(rewrite))
            if "keywords" in plan and "zh" in plan["keywords"]:
                for kw in plan["keywords"]["zh"]:
                    q_terms.update(tokenize(kw))
            queries.append({
                "id": plan.get("id", len(queries)),
                "raw_plan": plan,
                "terms": list(q_terms)
            })
            
    logging.info(f"Loaded {len(queries)} queries.")
    
    vocab_size = 1 << args.vocab_bits
    df = np.zeros(vocab_size, dtype=np.uint32)
    
    parquet_files = sorted(glob(os.path.join(args.docs, "*.parquet")))
    if not parquet_files:
        logging.error(f"No parquet files found in {args.docs}")
        return
        
    logging.info(f"Found {len(parquet_files)} parquet files. Starting Pass 1 (DF computation).")
    
    total_docs = 0
    # Pass 1: compute DF
    for p_idx, p_file in enumerate(parquet_files):
        logging.info(f"Pass 1: Processing {p_file} ({p_idx+1}/{len(parquet_files)})")
        table = pq.read_table(p_file)
        schema_names = table.schema.names
        has_title = "title" in schema_names
        has_lang = "lang" in schema_names
        
        for batch in table.to_batches():
            texts = batch.column("text").to_pylist()
            titles = batch.column("title").to_pylist() if has_title else [""] * len(texts)
            langs = batch.column("lang").to_pylist() if has_lang else ["zh"] * len(texts)
            
            for text, title, lang in zip(texts, titles, langs):
                if has_lang and lang != "zh":
                    continue
                total_docs += 1
                doc_toks = set(tokenize(text or ""))
                if title:
                    doc_toks.update(tokenize(title))
                for t in doc_toks:
                    df[hash_word(t, vocab_size)] += 1
        print_mem()
        
    logging.info(f"Pass 1 complete. Total docs: {total_docs}. Time: {time.time() - start_time:.2f}s")
    
    if total_docs == 0:
        return
        
    # Precompute query term IDF and hashes
    for q in queries:
        q["term_info"] = []
        for t in q["terms"]:
            h = hash_word(t, vocab_size)
            doc_freq = df[h]
            idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1.0)
            if idf < 0: idf = 0
            q["term_info"].append((t, h, idf))
            
    del df
    import gc
    gc.collect()

    logging.info("Starting Pass 2 (BM25 Scoring).")
    
    top_k = {q["id"]: [] for q in queries}
    k1 = 1.5
    b = 0.75
    avgdl = 500
    
    for p_idx, p_file in enumerate(parquet_files):
        logging.info(f"Pass 2: Processing {p_file} ({p_idx+1}/{len(parquet_files)})")
        table = pq.read_table(p_file)
        schema_names = table.schema.names
        has_title = "title" in schema_names
        has_doc_id = "doc_id" in schema_names
        has_lang = "lang" in schema_names
        
        for batch in table.to_batches():
            texts = batch.column("text").to_pylist()
            doc_ids = batch.column("doc_id").to_pylist() if has_doc_id else [str(i) for i in range(len(texts))]
            titles = batch.column("title").to_pylist() if has_title else [""] * len(texts)
            langs = batch.column("lang").to_pylist() if has_lang else ["zh"] * len(texts)
            
            for text, doc_id, title, lang in zip(texts, doc_ids, titles, langs):
                if has_lang and lang != "zh":
                    continue
                text_str = text or ""
                title_str = title or ""
                toks = tokenize(text_str)
                title_toks = tokenize(title_str)
                
                dl = len(toks)
                if dl == 0: continue
                
                tf = defaultdict(int)
                for t in toks:
                    tf[hash_word(t, vocab_size)] += 1
                for t in title_toks:
                    tf[hash_word(t, vocab_size)] += 2
                    
                for q in queries:
                    score = 0.0
                    for t, h, idf in q["term_info"]:
                        if h in tf:
                            freq = tf[h]
                            score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avgdl))
                    
                    if score > 0:
                        heapq.heappush(top_k[q["id"]], (score, doc_id, text_str))
                        if len(top_k[q["id"]]) > args.k:
                            heapq.heappop(top_k[q["id"]])
                            
        print_mem()
        
    logging.info(f"Pass 2 complete. Time: {time.time() - start_time:.2f}s. Starting Chunking.")
    
    out_records = []
    
    for q in queries:
        qid = q["id"]
        top_docs = sorted(top_k[qid], key=lambda x: x[0], reverse=True)
        
        doc_list = []
        chunks = []
        
        for score, doc_id, doc_text in top_docs:
            doc_list.append([score, doc_id])
            sel = select_chunks("", q["raw_plan"], doc_text, target_chars=750)
            if sel:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_text": sel[0],
                    "score": score
                })
                
        out_records.append({
            "id": qid,
            "docs": doc_list,
            "chunks": chunks
        })
        
    with open(args.out, "w", encoding="utf-8") as f:
        for rec in out_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    logging.info(f"Done. Wrote to {args.out}. Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
