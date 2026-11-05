"""Write final <=60-line handoff and concise progress after verified phase 1."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55');r=read(T/'phase1_report.json');assert read(T/'final_audit.json')['sha']
m=r['metrics'];c=r['corpus'];b=r['bench']
lines=['# vi55 — PHA1 HOÀN TẤT; không pha2/nộp/browser/commit/crawl',
       f"- Khochuẩn{c['corpus_vi']}VI/text{c['with_text']}/thiếu{c['without_text']};indexvi36/iter_all_texts/BoilerplateFilter.clean/make_bodyunlimited.",
       '- Poollịchsử223065có68ZH;29embeddingVolumevi51bSHA/fp16/L2PASS,táidùng222997VI2048,mới506707VI1024.',
       '- Pin1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/fp16SDPA/left/LAST/L2/instructquery256.',
       f"- Chốt1024:forecast+30%embed{r['decision']['chosen_forecast']['usd']:.6f}USD/tổngreserve{r['decision']['total_reserved_upper']:.6f}<26.",
       f"- Bench{b['tokens_per_second']:.3f}tok/s/min{b['repeat_cosine_min']:.9f}/mean{b['repeat_cosine_mean']:.9f}/sanityPASS/{b['function_seconds']:.3f}GPU-s.",
       f"- TổngGPU{r['total_gpu_seconds']:.3f}s/cậntrên{r['total_upper_usd']:.6f}USD;billingsnapshotchưasettled."]
for p,s in r['costs'].items():
    lines.append(f"- {p}:new{s['new_documents']}/reuse{s['reused_vi_documents']}/{s['bulk_parts']}part/GPU{s['gpu_seconds']:.3f}s/snapshot{s['billing_before']:.8f}→{s['billing_after']:.8f}/upper{s['upper_usd']:.6f}USD/stopped{s['app_stopped']}.")
for k in ('100','200'):
    x=m[k];n=x['outside_query_pool'];g=x['outside_global_pool'];o=x['overlap_bge_top200']
    lines.append(f"- k{k}:ngoàipooltheocâuTB/p50/p90={n['mean']:.3f}/{n['p50']:.3f}/{n['p90']:.3f};ngoàipoolglobal={g['mean']:.3f}/{g['p50']:.3f}/{g['p90']:.3f}.")
    lines.append(f"- k{k}:overlapbge200TB/p50/p90={o['mean']:.3f}/{o['p50']:.3f}/{o['p90']:.3f};vi53covermicro{x['vi53_coverage_micro']:.6f}/macro{x['vi53_coverage_macro']:.6f}.")
x=m['vi53_rank'];y=m['top30_proxy']
lines.extend([f"- Hạngvi53p50/p90={x['p50']:.3f}/{x['p90']:.3f};ngoàiVI/thiếu{x['missing_or_beyond1000']};competitionrankcosine1e-6.",
              f"- Top30mới/cũtheocâu={y['new_fraction']:.6f}/{y['old_fraction']:.6f};mới/cũratio={y['new_to_old_ratio']:.6f};ngoàiglobal={y['outside_global_pool_fraction']:.6f}.",
              '- Bgecomparisonhiệntạivi36/dense_v36_top200.json;diag18/dense_full_top400historicalcũngcótrongreport.',
              '- Output<HIST_TMP>/vi55/q3e8b_full_top1000.json.zst + phase1_report.json;mirrorD:/r2ai/data/sub/vi55/SHAverified.',
              '- Inputaudit51part729704/SHA/exactID/nooverlapPASS;query8workerSHAequal;finalindependentmerge+coverage+finite+budgetPASS.',
              '- TestsfullD1192PASS2skip+10vi55PASS/py_compile;C2crawlerlow_diskkhôngsửacrawl.',
              '- Haiappstopped/giữVolumeweights+embeddingfp16;taskprepareLastResult0đãxoá/lockabsent;khôngđụngvi54.',
              '- ChưakiểmF2/nhãnvàng/billingsettled;KHÔNGpha2/chấmdocmới/buildsubmission.',
              '- Khônglaunch/spawnlạicallcũ;đọcstate_<MODAL_PROFILE_B>,c.json trongTempvi55.'])
assert len(lines)<=60
(ROOT/'docs/VI55_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
progress=ROOT/'docs/PROGRESS.md';text=progress.read_text(encoding='utf-8')
entry=f"- **vi55 PHA1 XONG/CHƯA PHA2:**toànkhoVI{c['with_text']}/{c['corpus_vi']}text,táidùng222997Qwen8B2048+mới5067071024;top1000×1200lưulocal+mirrorb660/SHA/finalauditPASS. k100/200ngoàipooltheocâuTB{m['100']['outside_query_pool']['mean']:.3f}/{m['200']['outside_query_pool']['mean']:.3f};vi53cover{m['100']['vi53_coverage_micro']:.4f}/{m['200']['vi53_coverage_micro']:.4f};GPU{r['total_gpu_seconds']:.1f}s/upper{r['total_upper_usd']:.4f}<26USD.2appstopped/Volumegiữ/taskpreparexoá;1192testPASS2skip+10vi55PASS.Khôngnộp/browser/commit/pha2;`docs/VI55_HANDOFF.md`."
progress.write_text(text.replace('# Tiến độ\n','# Tiến độ\n\n'+entry+'\n',1),encoding='utf-8')
print('\n'.join(lines),flush=True)
