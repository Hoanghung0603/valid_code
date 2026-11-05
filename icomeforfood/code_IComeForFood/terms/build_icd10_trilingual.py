"""Join Vietnamese, English and Chinese ICD-10 tables on the WHO code.

Inputs (in --dir, made by icd10_vn.py, icd10_zh.py, simple-icd-10 export, wikidata_terms.py):
    icd10_vi_ICD10.parquet, icd10_vi_ICD10_TT06.parquet (optional),
    icd10_en_who2019.parquet, icd10_zh_categories.parquet, icd10_zh_terms.parquet,
    ../terms/wikidata_terms.parquet
Output: <dir>/icd10_trilingual.parquet and .csv with
    code, level, chapter, name_vi, name_vi_tt06, name_en, name_zh,
    note_vi, include_vi, syn_vi, syn_en, syn_zh, source
syn_* are alternative names only (never exclusion terms), '; '-joined.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

LEVEL = {"chapter": "chapter", "section": "block", "type": "category", "disease": "subcategory"}
COVID_EN = {"U08": "Personal history of COVID-19", "U08.9": "Personal history of COVID-19, unspecified",
            "U09": "Post COVID-19 condition", "U09.9": "Post COVID-19 condition, unspecified"}


_PRIORITY = {"disease": 0, "type": 1, "section": 2, "subsection": 3, "subsubsection": 4, "chapter": 5}


def vi_table(path: Path, name_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """WHO-level Vietnamese names, plus names of national extension codes grouped by their 4-char parent.

    Codes are cleaned of dagger/asterisk marks (A06.4† -> A06.4). When a section and a
    category share a code (TT06 uses the first code as section code), the disease/type row wins.
    """
    v = pd.read_parquet(path)
    v["code"] = v.code.str.replace(r"[†*\s]", "", regex=True)
    v["prio"] = v.model.map(_PRIORITY).fillna(9)
    is_ext = v.model.eq("disease") & v.code.str.len().gt(5)          # e.g. E11.90 (VN extension)
    ext = v[is_ext].assign(parent=v.code.str[:5]).groupby("parent")[name_col].agg(list)
    # keep extension rows too: some 5-char codes (B18.00 ...) also exist in the WHO list
    keep = v.sort_values("prio").drop_duplicates("code")
    keep = keep.assign(level=keep.model.map(LEVEL))
    return keep, ext


def joined(values) -> str | None:
    seen, out = set(), []
    for v in values:
        if isinstance(v, str) and v.strip() and v.strip().lower() not in seen:
            seen.add(v.strip().lower())
            out.append(v.strip())
    return "; ".join(out) or None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--wikidata", required=True, type=Path)
    args = ap.parse_args()
    d = args.dir

    en = pd.read_parquet(d / "icd10_en_who2019.parquet")
    base = en.rename(columns={"kind": "level"})[["code", "level", "chapter", "name_en"]]

    vi, ext_vi = vi_table(d / "icd10_vi_ICD10.parquet", "name_vi")
    vi = vi[["code", "name_vi"] + [c for c in ("note", "include") if c in vi.columns]]
    vi = vi.rename(columns={"note": "note_vi", "include": "include_vi"})
    # VN chapter names start with "(A00-B99) ..."
    vi["name_vi"] = vi.name_vi.str.replace(r"^\([A-Z]\d{2}-[A-Z]\d{2}\)\s*", "", regex=True)
    base = base.merge(vi, on="code", how="left")

    tt = d / "icd10_vi_ICD10_TT06.parquet"
    if tt.exists():
        t6, ext_t6 = vi_table(tt, "name_vi")
        base = base.merge(t6[["code", "name_vi"]].rename(columns={"name_vi": "name_vi_tt06"}),
                          on="code", how="left")
        ext_vi = pd.concat([ext_vi, ext_t6]).groupby(level=0).sum()

    zc = pd.read_parquet(d / "icd10_zh_categories.parquet")[["code", "name_zh"]].drop_duplicates("code")
    base = base.merge(zc, on="code", how="left")
    zt = pd.read_parquet(d / "icd10_zh_terms.parquet").groupby("code").term_zh.agg(list)
    # codes newer than the Chinese table (U08.9/U09.9 COVID-19, K43.9 after the 2016 split):
    # no category name, but diagnosis terms exist -> use the first one
    gap = base.name_zh.isna() & base.code.isin(zt.index)
    base.loc[gap, "name_zh"] = [zt[c][0] for c in base.loc[gap, "code"]]
    # WHO "Emergency use of U0x.x" rows are empty reserved slots, not diseases (the Chinese
    # table even names them literally 紧急使用U07.3). Real assignments have a Vietnamese name.
    has_vi = base.name_vi.notna() | (base.name_vi_tt06.notna() if "name_vi_tt06" in base else False)
    placeholder = base.name_en.str.startswith("Emergency use of") & ~has_vi
    base = base[~placeholder].copy()
    # simple-icd-10 still shows placeholder titles for the COVID-19 codes WHO assigned in 2021
    for code, title in COVID_EN.items():
        base.loc[base.code == code, "name_en"] = title
    for head in ("U08", "U09"):
        child = base.loc[base.code == f"{head}.9", "name_zh"]
        if len(child) and base.loc[base.code == head, "name_zh"].isna().all():
            base.loc[base.code == head, "name_zh"] = child.iloc[0]

    # Fill remaining gaps only from official sources, and say where each fill came from.
    # Never machine-translate here: closed LLMs are not allowed anywhere in the pipeline.
    notes: dict[int, list[str]] = {}
    if "name_vi_tt06" in base.columns:
        gap = base.name_vi.isna() & base.name_vi_tt06.notna()
        base.loc[gap, "name_vi"] = base.loc[gap, "name_vi_tt06"]
        for i in base.index[gap]:
            notes.setdefault(i, []).append("name_vi from TT06")
    by_code = base.set_index("code")
    for col in ("name_vi", "name_zh"):
        for i in base.index[base[col].isna()]:
            code = base.at[i, "code"]
            parent = code[:-1] if len(code) > 5 else (code[:3] if "." in code else None)
            if parent in by_code.index and isinstance(by_code.at[parent, col], str):
                base.at[i, col] = by_code.at[parent, col]
                notes.setdefault(i, []).append(f"{col} from parent {parent}")
    base["fill_note"] = [("; ".join(notes[i]) if i in notes else None) for i in base.index]

    wd = pd.read_parquet(args.wikidata)
    wd = wd[wd.prop == "icd10"]
    wsyn = wd.groupby(["code", "lang"]).term.agg(list).unstack()

    def syn(code, lang, extra=()):
        vals = list(extra)
        if code in wsyn.index and lang in wsyn.columns and isinstance(wsyn.at[code, lang], list):
            vals += wsyn.at[code, lang]
        return vals

    notes = base["note_vi"] if "note_vi" in base.columns else pd.Series([None] * len(base))
    t6names = base["name_vi_tt06"] if "name_vi_tt06" in base.columns else pd.Series([None] * len(base))
    base["syn_vi"] = [joined(syn(c, "vi", [x for x in (n, t) if isinstance(x, str)] + list(ext_vi.get(c, []))))
                      for c, n, t in zip(base.code, notes, t6names)]
    base["syn_en"] = [joined(syn(c, "en")) for c in base.code]
    base["syn_zh"] = [joined(syn(c, "zh", zt.get(c, []))) for c in base.code]
    base["source"] = "vi: icd.kcb.vn (QĐ 4469/QĐ-BYT; TT 06/2026); en: WHO ICD-10 2019 (simple-icd-10); " \
                     "zh: NHSA ICD-10 v2.0 + GB 2020; syn: Wikidata (CC0)"
    cols = ["code", "level", "chapter", "name_vi", "name_vi_tt06", "name_en", "name_zh",
            "note_vi", "include_vi", "syn_vi", "syn_en", "syn_zh", "fill_note", "source"]
    base = base[[c for c in cols if c in base.columns]].reset_index(drop=True)
    base.to_parquet(d / "icd10_trilingual.parquet", index=False)
    base.to_csv(d / "icd10_trilingual.csv", index=False, encoding="utf-8-sig")

    lv = base.level.isin(["category", "subcategory"])
    print("rows:", len(base), " category+subcategory:", int(lv.sum()))
    for c in ("name_vi", "name_vi_tt06", "name_en", "name_zh", "syn_vi", "syn_en", "syn_zh"):
        if c in base.columns:
            print(f"  {c:13s} filled for {base.loc[lv, c].notna().mean():.1%} of codes")
    print(base[base.code.isin(["A00", "E11", "I10", "K80.2", "J45"])][["code", "name_vi", "name_en", "name_zh"]].to_string())


if __name__ == "__main__":
    main()
