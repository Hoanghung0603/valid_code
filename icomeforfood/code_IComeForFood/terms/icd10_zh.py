"""Chinese ICD-10 names and diagnosis terms from the national tables (xlsx).

Source files (copies of the official NHSA / national tables, from
https://gitee.com/DoctorLi2023/icd-10, downloaded 14/10/2026), in --raw:
    医保ICD10_v2.0_字典库_0122.xlsx          NHSA medical-insurance ICD-10 v2.0 dictionary
    ICD10国标2020版和医保版2.0对照库.xlsx     national standard 2020 <-> NHSA v2.0, with aliases (别名)
    中医疾病分类与代码数据_20210830.xlsx      TCM disease codes (GB/T 15657)
    中医证候分类与代码数据_20210830.xlsx      TCM syndrome codes

Output in --out:
    icd10_zh_categories.parquet  code (WHO level, e.g. A00 / A00.0), name_zh, chapter_no
    icd10_zh_terms.parquet       code (WHO level), detail_code, term_zh, source
    tcm_disease.parquet, tcm_syndrome.parquet

Usage:
    python -I terms/icd10_zh.py --raw D:/r2ai_data/external/icd10/zh_raw --out D:/r2ai_data/external/icd10
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_WHO = re.compile(r"^([A-Z]\d{2})(?:\.(\d))?")


def who_code(detail: object) -> str | None:
    """'A00.000x001' -> 'A00.0', 'A01.001+' -> 'A01.0', 'B20' -> 'B20'."""
    if not isinstance(detail, str):
        return None
    m = _WHO.match(detail.strip())
    if not m:
        return None
    return m.group(1) + (f".{m.group(2)}" if m.group(2) else "")


def clean(s: object) -> str | None:
    if not isinstance(s, str):
        return None
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    yb = pd.read_excel(args.raw / "医保ICD10_v2.0_字典库_0122.xlsx", dtype=str)
    cats = yb[yb["分类码"].notna()][["分类码", "分类名", "章节序号"]].rename(
        columns={"分类码": "code", "分类名": "name_zh", "章节序号": "chapter_no"})
    cats["name_zh"] = cats.name_zh.map(clean)
    cats["code"] = cats.code.str.replace(r"[†‡*+＋\s]", "", regex=True)  # A17† -> A17
    cats = cats.drop_duplicates("code")
    terms = [pd.DataFrame({"detail_code": yb["诊断码"], "term_zh": yb["诊断名"], "source": "nhsa_v2"})]

    gb = pd.read_excel(args.raw / "ICD10国标2020版和医保版2.0对照库.xlsx", dtype=str)
    terms.append(pd.DataFrame({"detail_code": gb["主要编码"], "term_zh": gb["名称"], "source": "gb2020"}))
    terms.append(pd.DataFrame({"detail_code": gb["主要编码"], "term_zh": gb["别名"], "source": "gb2020_alias"}))
    terms.append(pd.DataFrame({"detail_code": gb["医保2.0编码"], "term_zh": gb["医保2.0名称"], "source": "nhsa_v2"}))
    t = pd.concat(terms, ignore_index=True)
    # 别名 cells may hold several aliases separated by punctuation
    t["term_zh"] = t.term_zh.map(clean)
    t = t.assign(term_zh=t.term_zh.str.split(r"[;；、|]")).explode("term_zh")
    t["term_zh"] = t.term_zh.map(clean)
    t["code"] = t.detail_code.map(who_code)
    t = t.dropna(subset=["term_zh", "code"]).drop_duplicates(["code", "term_zh"])

    args.out.mkdir(parents=True, exist_ok=True)
    cats.to_parquet(args.out / "icd10_zh_categories.parquet", index=False)
    t[["code", "detail_code", "term_zh", "source"]].to_parquet(args.out / "icd10_zh_terms.parquet", index=False)
    for src, name in (("中医疾病分类与代码数据_20210830.xlsx", "tcm_disease"),
                      ("中医证候分类与代码数据_20210830.xlsx", "tcm_syndrome")):
        pd.read_excel(args.raw / src, dtype=str).to_parquet(args.out / f"{name}.parquet", index=False)

    print("categories:", len(cats), " 3-char:", cats.code.str.len().eq(3).sum(), " 4-char:", cats.code.str.len().eq(5).sum())
    print("terms:", len(t), " distinct terms:", t.term_zh.nunique(), " codes covered:", t.code.nunique(),
          " by source:", t.source.value_counts().to_dict())
    print(t[t.code == "A00.0"].head(5).to_string())


if __name__ == "__main__":
    main()
