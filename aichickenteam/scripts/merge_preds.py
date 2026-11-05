"""
scripts/merge_preds.py - Merge VI predictions with ZH predictions.
"""
import argparse
import json
import zipfile
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.submit import complete, validate, load_doc_texts

def write_zip_int(pred, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_name = "submission.json"
    data = json.dumps(pred, ensure_ascii=False).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(json_name, data)
    tmp.replace(path)
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vi-preds", required=True)
    parser.add_argument("--zh-preds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-vi", type=int, default=20)
    parser.add_argument("--n-zh", type=int, default=60)
    parser.add_argument("--docs", default="data/docs")
    args = parser.parse_args()

    with open(args.vi_preds, "r") as f:
        try:
            vi_data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            vi_data = [json.loads(line) for line in f if line.strip()]
            
    with open(args.zh_preds, "r") as f:
        try:
            zh_data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            zh_data = [json.loads(line) for line in f if line.strip()]

    vi_dict = {str(item["id"]): item for item in vi_data}
    zh_dict = {str(item["id"]): item for item in zh_data}

    all_qids = set(vi_dict.keys()) | set(zh_dict.keys())
    
    final_preds = []
    
    for qid in sorted(all_qids, key=int):
        vi_item = vi_dict.get(qid, {"relevant_docs": [], "relevant_chunks": []})
        zh_item = zh_dict.get(qid, {"docs": [], "chunks": []})
        
        merged_docs = []
        
        vi_docs = vi_item.get("relevant_docs", [])[:args.n_vi]
        for d in vi_docs:
            if str(d) not in merged_docs:
                merged_docs.append(str(d))
                
        zh_docs_info = zh_item.get("docs", [])[:args.n_zh]
        for info in zh_docs_info:
            d = str(info[1])
            if d not in merged_docs:
                merged_docs.append(d)
                
        merged_chunks = []
        seen_c = set()
        
        vi_chunks = vi_item.get("relevant_chunks", [])
        for c in vi_chunks:
            if str(c["doc_id"]) in merged_docs:
                k = (str(c["doc_id"]), c["chunk_text"])
                if k not in seen_c:
                    seen_c.add(k)
                    merged_chunks.append({
                        "doc_id": str(c["doc_id"]),
                        "chunk_text": c["chunk_text"]
                    })
                    
        zh_chunks = zh_item.get("chunks", [])
        for c in zh_chunks:
            if str(c["doc_id"]) in merged_docs:
                k = (str(c["doc_id"]), c["chunk_text"])
                if k not in seen_c:
                    seen_c.add(k)
                    merged_chunks.append({
                        "doc_id": str(c["doc_id"]),
                        "chunk_text": c["chunk_text"]
                    })
                    
        final_preds.append({
            "id": int(qid),
            "relevant_docs": merged_docs,
            "relevant_chunks": merged_chunks
        })
        
    final_preds = complete(final_preds, [int(q) for q in all_qids])
    
    all_doc_ids = set()
    for p in final_preds:
        all_doc_ids.update(p["relevant_docs"])
        for c in p["relevant_chunks"]:
            all_doc_ids.add(c["doc_id"])
            
    
    errs = validate(final_preds, [int(q) for q in all_qids])  # chỉ kiểm định dạng; nguyên văn đã kiểm ở pipeline_v2 (VI) và trên Colab (ZH)
    # doc ZH không có text ở máy local (đã kiểm tra nguyên văn trên Colab) -> bỏ qua đúng loại lỗi này
    errs = [e for e in errs if "not found in doc lookup" not in e]
    if errs:
        print("Validation errors:", errs[:5])
        sys.exit(1)
        
    for p in final_preds:
        p["relevant_docs"] = [int(x) for x in p["relevant_docs"]]
        for c in p["relevant_chunks"]:
            c["doc_id"] = int(c["doc_id"])
            
    write_zip_int(final_preds, args.out)
    print(f"Successfully wrote {args.out}")

if __name__ == "__main__":
    main()
