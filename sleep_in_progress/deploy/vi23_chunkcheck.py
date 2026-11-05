"""Fail fast if a reused/new candidate lacks text or scored windows."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def coverage(cands, wins, docs):
    missing_text = missing_wins = pairs = empty_text = 0
    for query, hits in cands.items():
        for doc, *_ in hits:
            pairs += 1
            missing_text += doc not in docs
            empty_text += doc in docs and not docs[doc].strip()
            missing_wins += bool(docs.get(doc, "").strip()) and not bool(wins.get(query, {}).get(str(doc)))
    return {"pairs": pairs, "missing_text": missing_text, "missing_wins": missing_wins,
            "empty_text_pairs": empty_text,
            "ok": pairs > 0 and missing_text == missing_wins == 0}


def main():
    from rerank_run import load_docs
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["coverage", "verbatim"])
    parser.add_argument("--report", required=True)
    parser.add_argument("--cands")
    parser.add_argument("--docwin")
    parser.add_argument("--docs", nargs="+")
    args = parser.parse_args()
    report = Path(args.report)
    if args.mode == "verbatim":
        data = json.loads(report.read_text(encoding="utf-8"))
        return 0 if data["nguyên văn mới sai"] == data["nguyên văn cũ sai"] == 0 else 2
    cands = json.loads(Path(args.cands).read_text(encoding="utf-8"))
    wins = json.loads(Path(args.docwin).read_text(encoding="utf-8"))
    docs = load_docs(args.docs, wanted={hit[0] for hits in cands.values() for hit in hits})
    data = coverage(cands, wins, docs)
    data["missing_examples"] = [
        {"query": q, "doc": hit[0], "in_docs": hit[0] in docs,
         "text_chars": len(docs.get(hit[0], "")),
         "windows": len(wins.get(q, {}).get(str(hit[0]), []))}
        for q, hits in cands.items() for hit in hits
        if hit[0] not in docs or (docs[hit[0]].strip() and not wins.get(q, {}).get(str(hit[0])))][:20]
    data["empty_text_doc_ids"] = sorted({hit[0] for hits in cands.values() for hit in hits
                                         if hit[0] in docs and not docs[hit[0]].strip()})
    report.write_text(json.dumps(data), encoding="utf-8")
    print(json.dumps(data), flush=True)
    return 0 if data["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
