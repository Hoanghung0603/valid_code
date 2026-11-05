"""Score text extractors against the hand-made gold set (PHAN_CONG.md section 8.2).

Gold set layout (made by NT):
    <gold_dir>/manifest.csv          columns: id,lang,domain,url
    <gold_dir>/<id>.txt              main content copied by hand, verbatim.
        - blocks are separated by one blank line
        - a heading block starts with "# " (any number of '#')
        - a block that must never be lost starts with "!! "
          (prefixes are markers only and are stripped before scoring)

Extractor output layout (produced by TV, one folder per extractor):
    <pred_dir>/<extractor>/<id>.json  {"id": int, "text": str,
                                       "blocks": [{"type", "level", "text",
                                                   "char_start", "char_end"}]}

Usage:
    python eval/score_extraction.py --gold D:/r2ai_data/gold \
        --pred D:/r2ai_data/reports/extract_gold [--baseline prev_summary.csv] \
        [--out D:/r2ai_data/reports/extract_scores]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2ai_eval import lcs_len, normalize, tokenize  # noqa: E402

TOKENIZER = "cjk"
# Starting thresholds (PHAN_CONG.md 8.2); NT may change them.
GATES = {
    "recall_mean_min": 0.95,
    "recall_page_min": 0.85,
    "junk_mean_max": 0.15,
    "required_min": 1.0,
    "heading_min": 0.90,
}
BLOCK_HIT = 0.90  # a gold block counts as present if >= 90% of its tokens survive


def parse_gold(path: Path) -> list[dict]:
    blocks = []
    for raw in path.read_text(encoding="utf-8").split("\n\n"):
        raw = raw.strip()
        if not raw:
            continue
        required = raw.startswith("!! ")
        if required:
            raw = raw[3:].lstrip()
        heading = raw.startswith("#")
        if heading:
            raw = raw.lstrip("#").strip()
        blocks.append({"text": raw, "required": required, "heading": heading})
    return blocks


def check_blocks(doc: dict) -> int:
    """Count blocks whose char offsets do not reproduce their text."""
    text = doc.get("text", "")
    return sum(
        1
        for b in doc.get("blocks", [])
        if text[b.get("char_start", 0):b.get("char_end", 0)] != b.get("text", "")
    )


def score_page(gold_blocks: list[dict], doc: dict | None) -> dict:
    gold_tokens = tokenize(" ".join(b["text"] for b in gold_blocks), TOKENIZER)
    if doc is None:
        return {"missing": 1, "recall": 0.0, "junk": 1.0, "required": 0.0,
                "heading": 0.0, "bad_offsets": 0, "n_gold": len(gold_tokens),
                "n_pred": 0}
    pred_tokens = tokenize(doc.get("text", ""), TOKENIZER)
    common = lcs_len(pred_tokens, gold_tokens)
    recall = common / len(gold_tokens) if gold_tokens else 0.0
    junk = 1 - common / len(pred_tokens) if pred_tokens else 1.0

    def present(block_text: str) -> bool:
        bt = tokenize(block_text, TOKENIZER)
        return bool(bt) and lcs_len(pred_tokens, bt) / len(bt) >= BLOCK_HIT

    required = [b for b in gold_blocks if b["required"]]
    req_score = (sum(present(b["text"]) for b in required) / len(required)
                 if required else 1.0)

    pred_headings = {normalize(b.get("text", "")) for b in doc.get("blocks", [])
                     if b.get("type") == "heading"}
    gold_headings = [b for b in gold_blocks if b["heading"]]
    head_score = (sum(normalize(b["text"]) in pred_headings for b in gold_headings)
                  / len(gold_headings) if gold_headings else 1.0)

    return {"missing": 0, "recall": recall, "junk": junk, "required": req_score,
            "heading": head_score, "bad_offsets": check_blocks(doc),
            "n_gold": len(gold_tokens), "n_pred": len(pred_tokens)}


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def summarize(rows: list[dict], key: str | None) -> dict:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r[key] if key else "ALL", []).append(r)
    out = {}
    for g, rs in sorted(groups.items()):
        out[g] = {
            "pages": len(rs),
            "missing": sum(r["missing"] for r in rs),
            "recall_mean": mean([r["recall"] for r in rs]),
            "recall_min": min(r["recall"] for r in rs),
            "junk_mean": mean([r["junk"] for r in rs]),
            "required_mean": mean([r["required"] for r in rs]),
            "heading_mean": mean([r["heading"] for r in rs]),
            "bad_offsets": sum(r["bad_offsets"] for r in rs),
        }
    return out


def gate(s: dict) -> list[str]:
    fails = []
    if s["recall_mean"] < GATES["recall_mean_min"]:
        fails.append(f"recall_mean {s['recall_mean']:.3f} < {GATES['recall_mean_min']}")
    if s["recall_min"] < GATES["recall_page_min"]:
        fails.append(f"recall_min {s['recall_min']:.3f} < {GATES['recall_page_min']}")
    if s["junk_mean"] > GATES["junk_mean_max"]:
        fails.append(f"junk_mean {s['junk_mean']:.3f} > {GATES['junk_mean_max']}")
    if s["required_mean"] < GATES["required_min"]:
        fails.append(f"required {s['required_mean']:.3f} < {GATES['required_min']}")
    if s["heading_mean"] < GATES["heading_min"]:
        fails.append(f"heading {s['heading_mean']:.3f} < {GATES['heading_min']}")
    if s["missing"]:
        fails.append(f"{s['missing']} page(s) without output")
    if s["bad_offsets"]:
        fails.append(f"{s['bad_offsets']} block(s) with wrong char offsets")
    return fails


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--baseline", type=Path,
                    help="summary.csv of a previous run, for regression check")
    args = ap.parse_args()

    with open(args.gold / "manifest.csv", encoding="utf-8") as f:
        manifest = [m for m in csv.DictReader(f)
                    if (args.gold / f"{m['id']}.txt").exists()]
    if not manifest:
        sys.exit("no gold pages found (manifest rows need a matching <id>.txt)")
    gold = {m["id"]: parse_gold(args.gold / f"{m['id']}.txt") for m in manifest}

    extractors = sorted(p.name for p in args.pred.iterdir() if p.is_dir())
    all_rows, summary_rows = [], []
    for ex in extractors:
        rows = []
        for m in manifest:
            path = args.pred / ex / f"{m['id']}.json"
            doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            rows.append({"extractor": ex, "id": m["id"], "lang": m["lang"],
                         "domain": m["domain"], **score_page(gold[m["id"]], doc)})
        all_rows += rows

        print(f"\n=== {ex} ===")
        for key in (None, "lang", "domain"):
            for g, s in summarize(rows, key).items():
                label = g if key else "ALL"
                print(f"{label:38s} n={s['pages']:3d}  recall={s['recall_mean']:.3f} "
                      f"(min {s['recall_min']:.3f})  junk={s['junk_mean']:.3f}  "
                      f"required={s['required_mean']:.3f}  heading={s['heading_mean']:.3f}")
                summary_rows.append({"extractor": ex, "group": f"{key or 'all'}:{g}", **s})
        overall = summarize(rows, None)["ALL"]
        fails = gate(overall)
        print("GATE:", "PASS" if not fails else "FAIL -> " + "; ".join(fails))
        worst = sorted(rows, key=lambda r: r["recall"])[:5]
        print("worst recall pages:", ", ".join(f"{r['id']}({r['recall']:.2f})" for r in worst))

    if args.baseline and args.baseline.exists():
        with open(args.baseline, encoding="utf-8") as f:
            base = {(r["extractor"], r["group"]): r for r in csv.DictReader(f)}
        print("\n=== regression vs baseline (overall) ===")
        for s in summary_rows:
            b = base.get((s["extractor"], s["group"]))
            if b and s["group"] == "all:ALL":
                d = s["recall_mean"] - float(b["recall_mean"])
                flag = "  <-- WORSE" if d < -0.005 else ""
                print(f"{s['extractor']}: recall {float(b['recall_mean']):.3f} -> "
                      f"{s['recall_mean']:.3f} ({d:+.3f}){flag}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, data in (("pages.csv", all_rows), ("summary.csv", summary_rows)):
            with open(args.out / name, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader()
                w.writerows(data)
        print(f"\nwrote {args.out / 'pages.csv'} and {args.out / 'summary.csv'}")


if __name__ == "__main__":
    main()
