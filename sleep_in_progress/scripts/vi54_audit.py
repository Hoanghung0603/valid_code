"""Independent fixed-rule document and full ZIP audit, no compression/build."""
import gc
import math
from pathlib import Path
import zipfile

from vi48 import fingerprint,read,write

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
OUT=TEMP/'vi54'


def main():
    import dwmax_diag as dw
    from vi41c import load_zip,digest,indexed,merge_row
    from vi37 import docwin_hits
    from fuse3 import fuse3
    from vi29 import assert_clean,load_corpus_ids
    from vi52 import compare,load_docs
    zp=OUT/'submission_vi54.zip';report=read(OUT/'build_report.json')
    assert fingerprint(zp)=={k:report[k] for k in ('bytes','sha256')}
    assert zp.stat().st_size<=99_000_000
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        assert z.testzip() is None
    final=indexed(load_zip(zp));assert set(final)==set(range(1,1201))
    lists={q:r['relevant_docs'] for q,r in final.items()}
    chunkhash={q:digest(r['relevant_chunks']) for q,r in final.items()}
    source=TEMP/'vi45'
    bad=set(map(int,read(source/'bad_ids.json')))-set(map(int,read(source/'new_ids.json')))
    clean=assert_clean(list(final.values()),bad,load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    assert all(len(r['relevant_docs'])==len(set(r['relevant_docs'])) for r in final.values())
    del final;gc.collect()
    ctx=dw.read(TEMP/'vi50/fusion_context.json.zst');data=ctx['data']
    qwen=read(TEMP/'vi51/cosines_vi51b.json');nem=dw.read(OUT/'nemotron_cosines.json.zst')
    merged={}
    # Separate implementation: complete common full pool; population moments.
    for q,hits in data['pool'].items():
        ids=[str(h[0]) for h in hits];normalized=[]
        for source in (qwen,nem):
            values=[source[q][d] for d in ids];mu=sum(values)/len(values)
            sigma=math.sqrt(sum((v-mu)**2 for v in values)/len(values))
            assert sigma>0
            normalized.append([(v-mu)/sigma for v in values])
        merged[q]={d:(normalized[0][i]+normalized[1][i])/2 for i,d in enumerate(ids)}
    d8w=read(TEMP/'vi50/d8w.json')
    ranks=fuse3(data['pool'],data['q8b'],data['dense'],(1.,2.,2.,1.),60,data['hyde'],
        [(docwin_hits(data['pool'],ctx['bge']),1.),(docwin_hits(data['pool'],d8w),1.),(docwin_hits(data['pool'],merged),1.)])
    base=indexed(load_zip(TEMP/'vi48/submission_vi48.zip'));protected=0
    for q,r in base.items():
        candidate=dict(id=q,relevant_docs=[int(h[0]) for h in ranks[str(q)][:len(r['relevant_docs'])]])
        expected,m,short=merge_row(r,candidate);protected+=m
        assert short==0 and expected['relevant_docs']==lists[q]
        assert digest(r['relevant_chunks'])==chunkhash[q] and len(r['relevant_docs'])==len(lists[q])
    del base,ctx,data,qwen,nem,d8w,merged,ranks;gc.collect()
    offline=read(OUT/'offline.json')
    for label in ('vi52','vi53'):
        p=TEMP/label/f'submission_{label}.zip'
        if p.exists():assert compare(lists,load_docs(p))==offline['vs_'+label]
    assert protected==offline['protected_docs']
    result=dict(zip=fingerprint(zp),crc=0,entry='submission.json',compression='LZMA',ids=1200,
        independent_dense_mean_z_rrf_protection=1200,chunks_equal_vi48=1200,
        doc_counts_equal_vi48=1200,no_duplicate_docs=True,assert_clean=clean,offline_doc_changes_match=True)
    write(OUT/'independent_audit.json',result);print('INDEPENDENT_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
