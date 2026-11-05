import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pubmed_fetch import (  # noqa: E402
    CtxpClient,
    EfetchClient,
    extract_pmid,
    fetch_all,
    join_abstract,
    merge_parts,
    parse_pubmed_xml,
    parse_ris,
)

XML_FULL = b"""<?xml version="1.0"?>
<PubmedArticleSet>
 <PubmedArticle>
  <MedlineCitation>
   <PMID Version="1">12345678</PMID>
   <Article>
    <Journal><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
    <ArticleTitle>COVID-19 <i>vaccine</i> efficacy in adults</ArticleTitle>
    <Abstract>
     <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Old text.</AbstractText>
     <AbstractText Label="METHODS">We studied 100 patients.</AbstractText>
     <AbstractText NlmCategory="RESULTS">Efficacy was 95%.</AbstractText>
    </Abstract>
    <Language>eng</Language>
   </Article>
   <MeshHeadingList>
    <MeshHeading><DescriptorName>COVID-19 Vaccines</DescriptorName></MeshHeading>
    <MeshHeading><DescriptorName>Adult</DescriptorName></MeshHeading>
   </MeshHeadingList>
  </MedlineCitation>
 </PubmedArticle>
 <PubmedArticle>
  <MedlineCitation>
   <PMID Version="1">22222222</PMID>
   <Article>
    <Journal><JournalIssue><PubDate><MedlineDate>2019 Jan-Feb</MedlineDate></PubDate></JournalIssue></Journal>
    <ArticleTitle>Short report</ArticleTitle>
    <Language>fre</Language>
   </Article>
  </MedlineCitation>
 </PubmedArticle>
</PubmedArticleSet>
"""


def test_extract_pmid_ok():
    assert extract_pmid("https://pubmed.ncbi.nlm.nih.gov/12345678/") == 12345678
    assert extract_pmid("https://pubmed.ncbi.nlm.nih.gov/11197") == 11197
    assert extract_pmid("http://pubmed.ncbi.nlm.nih.gov/1/?term=x") == 1


def test_extract_pmid_fail():
    assert extract_pmid("https://www.vinmec.com/bai-viet/") is None
    assert extract_pmid("https://pubmed.ncbi.nlm.nih.gov/abc/") is None
    assert extract_pmid("") is None
    assert extract_pmid(None) is None


def test_join_abstract_labels():
    assert join_abstract([("BACKGROUND", "Old text."), ("METHODS", "We studied."),
                          ("", "Efficacy 95%.")]) == "BACKGROUND: Old text.\nMETHODS: We studied.\nEfficacy 95%."
    assert join_abstract([("", "only")]) == "only"
    assert join_abstract([]) == ""


def test_parse_pubmed_xml_full():
    recs = parse_pubmed_xml(XML_FULL)
    assert set(recs) == {12345678, 22222222}
    r = recs[12345678]
    assert r["title"] == "COVID-19 vaccine efficacy in adults"
    assert r["abstract"] == ("BACKGROUND: Old text.\nMETHODS: We studied 100 patients.\nEfficacy was 95%.")
    assert r["lang"] == "eng"
    assert r["year"] == "2021"
    assert r["mesh"] == ["COVID-19 Vaccines", "Adult"]


def test_parse_pubmed_xml_no_abstract_and_medline_date():
    r = parse_pubmed_xml(XML_FULL)[22222222]
    assert r["title"] == "Short report"
    assert r["abstract"] == ""
    assert r["year"] == "2019"
    assert r["lang"] == "fre"
    assert r["mesh"] == []


def test_parse_pubmed_xml_empty_set():
    assert parse_pubmed_xml(b"<PubmedArticleSet></PubmedArticleSet>") == {}


RIS_TWO = b"""TY  - JOUR
T1  - First title
T1  - ignored second
TT  - Translated title
LA  - eng
Y1  - 2021/05/
AB  - Background line.
      continued line.
AB  - Second abstract part.
KW  - COVID-19 Vaccines
KW  - Adult
AN  - 12345678
ER  - 

TY  - JOUR
T1  - Short report
LA  - fre
Y1  - 2019/01/
AN  - 22222222
ER  - 
"""


def test_parse_ris_two_records():
    recs = parse_ris(RIS_TWO)
    assert set(recs) == {12345678, 22222222}
    r = recs[12345678]
    assert r["title"] == "First title"
    assert r["abstract"] == "Background line.\ncontinued line.\nSecond abstract part."
    assert r["lang"] == "eng" and r["year"] == "2021"
    assert r["mesh"] == ["COVID-19 Vaccines", "Adult"]
    r2 = recs[22222222]
    assert r2["title"] == "Short report" and r2["abstract"] == ""
    assert r2["lang"] == "fre" and r2["year"] == "2019" and r2["mesh"] == []


def test_parse_ris_empty():
    assert parse_ris(b"") == {}


def test_ctxp_client_url_and_fetch():
    seen = []

    def opener(url: str) -> bytes:
        seen.append(url)
        return RIS_TWO

    c = CtxpClient(opener=opener, retry_backoff=[])
    data = c.fetch([12345678, 22222222])
    assert parse_ris(data)[12345678]["title"] == "First title"
    assert seen[0].startswith("/lit/ctxp/v1/pubmed/?format=ris&id=12345678,22222222")


def test_ctxp_client_retry_transient():
    calls = []

    def opener(url: str) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            from pubmed_fetch import TransientHTTP
            raise TransientHTTP(503)
        return RIS_TWO

    c = CtxpClient(opener=opener, retry_backoff=[0.0])
    assert parse_ris(c.fetch([1])) == {12345678: parse_ris(RIS_TWO)[12345678],
                                      22222222: parse_ris(RIS_TWO)[22222222]}
    assert len(calls) == 2


def _write_map(outdir: Path, rows):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "pmid_map.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_fetch_all_resume_and_missing(tmp_path):
    _write_map(tmp_path, [{"id": 1, "pmid": 111}, {"id": 2, "pmid": 222},
                          {"id": 3, "pmid": 333}, {"id": 4, "pmid": 444}])
    calls = []

    def opener(body: bytes) -> bytes:
        calls.append(body.decode())
        if b"id=111" in body:  # lô 0: chỉ trả về 111
            return XML_FULL.replace(b"12345678", b"111").replace(b"22222222", b"999")
        return XML_FULL.replace(b"12345678", b"333").replace(b"22222222", b"444")

    client = EfetchClient(opener=opener, retry_backoff=[])
    st = fetch_all(tmp_path, batch=2, client=client)
    assert st["done"] == 2 and st["returned"] == 3 and st["missing"] == 1
    assert len(calls) == 2
    # resume: chạy lại không request thêm
    st2 = fetch_all(tmp_path, batch=2, client=EfetchClient(opener=opener, retry_backoff=[]))
    assert st2["skipped"] == 2 and len(calls) == 2
    stats = merge_parts(tmp_path)
    assert stats["n_docs"] == 4 and stats["returned"] == 3 and stats["missing"] == 1
    from pubmed_docs import iter_texts
    recs = list(iter_texts([tmp_path / "pubmed_text.jsonl.zst"]))
    assert [r["id"] for r in recs] == [1, 2, 3, 4]
    assert recs[1]["title"] == "" and recs[1]["abstract"] == ""  # pmid 222 không trả về
    assert recs[2]["pmid"] == 333 and recs[3]["pmid"] == 444


def test_fetch_all_stops_when_blocked(tmp_path):
    _write_map(tmp_path, [{"id": i, "pmid": i} for i in range(1, 7)])

    def opener(body: bytes) -> bytes:
        raise OSError("connection refused")

    client = EfetchClient(opener=opener, retry_backoff=[])
    st = fetch_all(tmp_path, batch=1, client=client)
    assert st["fail_stopped"] is True
    assert st["done"] == 0
