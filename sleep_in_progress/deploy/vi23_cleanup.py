"""Remove named vi23 staging files only; preserve results, reports and source data."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("machine", choices=["local", "b660", "z87"])
    args = parser.parse_args()
    if args.machine == "local":
        base = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + r"\diag23")
        names = ["docs_vi23_dw_new.jsonl.zst", "docs_vi23_dw_old.part000.jsonl.zst",
                 "docs_vi23_embed.jsonl.zst"]
    elif args.machine == "b660":
        base = Path(r"D:\r2ai\data\sub\diag23")
        names = ["next_delta_vi23.txt", "docwin_vi23/docwin_v23all.json",
                 "docwin_vi23/chunk_dw_u900.json.zst",
                 "docwin_vi23/chunk_dw_u900_v23_new.json.zst",
                 "docwin_vi23/docs_vi23_dw_old.part000.jsonl.zst",
                 "docwin_vi23/docs_vi23_dw_new.jsonl.zst", "docs_vi23_embed.jsonl.zst"]
    else:
        base = Path(r"D:\r2ai\work\vi23")
        names = ["docs_vi23_dw_new.jsonl.zst", "docs_vi23_dw_old.part000.jsonl.zst",
                 "winsample.json", "winscore.json", "vm.txt"]
        names += ["dw_parts/part_00000.json", "dw_parts/part_00001.json", "dw_parts/part_00002.json",
                  "dw_parts/part_00003.json", "dw_parts/part_00004.json", "dw_parts/part_00005.json",
                  "dw_parts/part_00006.json", "dw_parts/part_00007.json", "dw_parts/part_00008.json",
                  "dw_parts/part_00009.json", "dw_parts/part_00010.json", "dw_parts/part_00011.json",
                  "dw_parts/part_00012.json", "dw_parts/part_00013.json", "dw_parts/part_00014.json",
                  "dw_parts/part_00015.json", "dw_parts/part_00016.json", "dw_parts/part_00017.json",
                  "dw_parts/part_00018.json", "dw_parts/part_00019.json", "dw_parts/part_00020.json",
                  "dw_parts/part_00021.json", "dw_parts/part_00022.json", "dw_parts/part_00023.json"]
        names += [name.replace("dw_parts/", "rr_new_parts/") for name in list(names)
                  if name.startswith("dw_parts/")]
        names += ["rr_sanity_parts/part_00000.json", "docs_embed/docs_vi23_embed.jsonl.zst",
                  "docs_san/san200.jsonl.zst", "docs_vi23_sanity200.jsonl.zst",
                  "dense_sanity_ignore.json", "dense_scan_ignore.json"]
    removed = []
    for name in names:
        path = base / name
        if path.is_file():
            path.unlink()
            removed.append(name)
    if args.machine == "z87":
        for name in ["dw_parts", "rr_new_parts", "rr_sanity_parts", "docs_embed", "docs_san"]:
            parts = base / name
            if parts.exists() and not list(parts.iterdir()):
                parts.rmdir()
    print(json.dumps({"machine": args.machine, "removed": removed}))


if __name__ == "__main__":
    main()
