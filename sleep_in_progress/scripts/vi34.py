r"""vi34 = vi33 voi DUNG MOT thay doi: diem 8B q8b_v32 -> diag33\q8b_v33.json
(bu 63.302 cap thieu 8B trong toan doclist vi33 + 30 hang bien; scripts/q8b33.py, Modal L40S).

Giu y vi33: w_q8b fhyde = 2 (rrf3d2/zd chunk = 1), pool/dense/hyde v30, chi muc vi30, tham so chain,
dedup "doc_text". Tai dung vi33 (va qua do vi31) bang cach gan lai bien module.
  sanity   duong vi34 voi q8b_v32 + chunk_v33 + W8F 2 -> phai tai lap submission_vi33 1200/1200 (delta 56).
  doclist  nhu vi33.stage_doclist voi Q8 = q8b_v33; so voi doclist vi33.
  chunk/build/stats/docwin  nhu vi33 (vi31).
Out b660 D:\r2ai\data\sub\vi34\, z87 D:\r2ai\work\vi34\. Artefact vi33/diag* CHI DOC.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi30
import vi31
import vi33
from vi24 import read, write, owned_lock

ROOT = vi31.ROOT
OUT33 = ROOT / "data/sub/vi33"                       # CHI DOC
Q8_OLD = ROOT / "data/sub/diag32/q8b_v32.json"       # 8B cua vi33 (sanity)
Q8 = ROOT / "data/sub/diag33/q8b_v33.json"           # 8B moi
CHUNK_V33 = OUT33 / "chunk_dw_u900_v33.json.zst"
OUT = ROOT / "data/sub/vi34"
G = ROOT / "work/vi34"
NAME, SANITY_NAME, PROBE = "vi34", "vi34_sanity", "vi34p"
W8F = 2.0
DELTA = 56             # delta cua submission_vi33

BINDINGS31 = {"OUT": OUT, "IN31": OUT / "in1", "IN_S": OUT / "in_sanity", "G": G,
              "Q8_V31": Q8, "CHUNK30": CHUNK_V33,
              "CHUNK31": OUT / "chunk_dw_u900_v34.json.zst",
              "NAME": NAME, "SANITY_NAME": SANITY_NAME, "PROBE": PROBE, "DELTA0": DELTA}
# vi33.stage_doclist doc cac bien nay: OUT32 -> vi33 (doclist_v31.json = doclist cu de so sanh)
BINDINGS33 = {"OUT": OUT, "G": G, "Q8": Q8, "CHUNK_V32": CHUNK_V33, "OUT32": OUT33, "W8F": W8F,
              "NAME": NAME, "SANITY_NAME": SANITY_NAME, "PROBE": PROBE, "DELTA": DELTA}


def bind() -> dict:
    """Gan lai bien module vi31 + vi33. Tra {(mod, k): gia tri cu}."""
    old = {}
    for mod, b in ((vi31, BINDINGS31), (vi33, BINDINGS33)):
        for k, v in b.items():
            old[(mod.__name__, k)] = getattr(mod, k)
            setattr(mod, k, v)
    return old


def unbind(old: dict):
    mods = {"vi31": vi31, "vi33": vi33}
    for (m, k), v in old.items():
        setattr(mods[m], k, v)


def stage_sanity():
    import vi29 as v29
    sr = OUT / "sanity_report.json"
    if sr.exists() and read(sr).get("ok"):
        print("sanity da xong (resume): bo qua", flush=True)
        return
    vi33.set_w8f(W8F)
    try:
        excl, corpus = vi31.load_excl_corpus()
        loc = vi31.write_inputs_v31(vi31.input_spec(Q8_OLD, CHUNK_V33), vi31.IN_S, excl, corpus)
        files = v29.chain_files(vi31.IN_S, vi31.FILE_NAMES)
        v29.run_chain(SANITY_NAME, DELTA, files, "doc_text", OUT)
    finally:
        vi33.set_w8f(None)
    cmp_rep = vi30.core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"),
                                     str(OUT33 / "submission_vi33.json"))
    cmp_rep["loc"] = loc
    write(sr, cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    v29.require_sanity(cmp_rep)
    (OUT / f"submission_{SANITY_NAME}.json").unlink()


STAGES = {"sanity": stage_sanity, "doclist": lambda: vi33.stage_doclist(), "chunk": lambda: vi31.stage_chunk(),
          "build": lambda: vi33.stage_build(), "stats": lambda: vi31.stage_stats(),
          "docwin": lambda: vi31.stage_docwin()}


def main():
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=list(STAGES))
    ap.add_argument("--machine", choices=["b660", "z87"], default="b660")
    a = ap.parse_args()
    bind()
    t0 = time.time()
    if a.machine == "z87":
        G.mkdir(parents=True, exist_ok=True)
        with owned_lock(ROOT / "locks" / "gpu.lock", f"vi34 {a.stage} pid={os.getpid()}"):
            STAGES[a.stage]()
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        STAGES[a.stage]()
    print("DONE", a.stage, round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
