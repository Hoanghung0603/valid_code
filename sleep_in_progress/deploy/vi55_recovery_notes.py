"""Compact final handoff/progress from the audited report."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55');r=read(T/'phase1_report.json');assert r['status']=='DONE'
m=r['metrics'];cost=r['costs'];lines=['# vi55 pha1 DONE — 31/10; KHÔNG pha2/nộp/browser/commit',
 '- Resume idempotent: npz fp16+ID+SHA+inputSHA; exactID/order/shape4096/finite/L2; commit từng part; partial không gộp.',
 '- Preemption dở+rerun=clean/testmarker/SHA PASS; 13vi55/27chuỗi; full1197PASS2skip; py_compilePASS.',
 '- Recovery30d8/8part80000doc:016,013,021,026,032,012,023,010;4call≤4H100; khôngembedlại43partcũ.',
 '- Tổng506707newVI1024+222997reusedVI2048=729704;740598VIofficial/thiếu10894text;68ZHpoollịchsửloại.',
 '- Qwen3Embedding8B pin1d8ad4ca9b3dd8059ad90a75d4983776a23d44af/fp16SDPA/left/LAST/L2/title+clean.',
 '- Gate50cosmin0.999999881/mean1.0/16.196103GPU-s; reporttrướcassertPASS.',
 '- QueryembedMỘTcall/1200exactID/shape4096/finite/L2/SHA; cùngartifact3workspace/searchSHAkiểm.',
 '- top1000file<HIST_TMP>/vi55/q3e8b_full_top1000.json.zst;1200×1000/noDup/finite/sorted/exactmerge.',
 '- Partition269555+380149+80000=729704/disjoint/fullIDs; exactcompetitionranksroundcos1e-6,outputfp32cosunrounded.']
lines.append('- Audit rank:2doc tại vịtrí1000 có roundedrank1001(q458/d224132,q1189/d609245); chênh quantization≤1e-6,khôngmấtID/cos.')
for k in ('100','200'):
 a=m[k];d=a['outside_query_pool'];g=a['outside_global_pool'];o=a['overlap_bge_top200']
 lines.append(f"- k{k}: ngoàipoolcâuTB/p50/p90={d['mean']:.6f}/{d['p50']:.1f}/{d['p90']:.1f};ngoàiunion={g['mean']:.6f}/{g['p50']:.1f}/{g['p90']:.1f};overlapBGE200={o['mean']:.6f}/{o['p50']:.1f}/{o['p90']:.1f};vi53micro={a['vi53_coverage_micro']:.6%}macro={a['vi53_coverage_macro']:.6%}.")
rank=m['vi53_rank'];p=m['top30_proxy'];lines.append(f"- Vi53rankVIp50/p90={rank['p50']:.1f}/{rank['p90']:.1f};nonVImissing={rank['missing_or_beyond1000']};top30new/querypool={p['new_fraction']:.6%};new/global={p['outside_global_pool_fraction']:.6%}.")
for name,d in m['top30_cosine'].items():
 lines.append(f"- Cos30{name}:n={d['n']};TB/p50/p90={d.get('mean',0):.6f}/{d.get('p50',0):.6f}/{d.get('p90',0):.6f}.")
for profile in (__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"),__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")):
 s=read(T/f'recovery_state_{profile}.json')
 lines.append(f"- {profile}:app{s['app_id']}stopped={s['stopped']};calls="+', '.join(f"{job}:{v.get('call_id')}:{v['status']}" for job,v in s['calls'].items()))
lines.extend([f"- GPUknownsuccess={r['known_success_gpu_seconds']:.6f}s/recovery={r['recovery_gpu_seconds']:.6f}s;oldfailedexactwallchưađủ.",
 f"- Snapshotphase1={r['total_snapshot_delta_usd']:.6f}USD/oldupper={r['prior_upper_usd']:.6f};30dsnapshot={cost[__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")]['snapshot_delta']:.6f}/upper={cost[__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")]['upper_usd']:.6f};totalphase1upper={r['total_upper_usd']:.6f}<26USD.",
 '- 3appstopped/tasks0xácnhận/Volumegiữ/khôngb660jobmới/lockabsent;mirrorb660D:/r2ai/data/sub/vi55/SHAauditPASS.',
 '- phase1_report.jsonDONElocal+b660;recovery_part_manifest/shared_query_audit/vi53_exact_ranks/recovery_tests_final.xml.',
 '- Chưakiểm:goldquality/F2,billingsettled,oldfailedpreemptionexactGPUwall;KHÔNGpha2/ZIP/nộp/browser/commit/crawl.'])
assert len(lines)<=60
(ROOT/'docs/VI55_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
p=ROOT/'docs/PROGRESS.md';text=p.read_text(encoding='utf-8')
entry=f"- **31/10 vi55 PHA1DONE/CHƯAPHA2:** resume idempotent/testpreemptionPASS;8/8missing80000doc30d+43partcũgiữ/tổng729704VI. Query1200mộtcall/sharedSHA/top10001200×1000/fullIDs/noDup/finite/exactmergePASS; đủmetrics+cos30 trongphase1_reportDONE. k100/200ngoàipoolTB{m['100']['outside_query_pool']['mean']:.3f}/{m['200']['outside_query_pool']['mean']:.3f};vi53cover{m['100']['vi53_coverage_micro']:.3%}/{m['200']['vi53_coverage_micro']:.3%};rankp50/p90{rank['p50']:.0f}/{rank['p90']:.0f}. GPUknown{r['known_success_gpu_seconds']:.3f}s/upper{r['total_upper_usd']:.6f}<26USD;3appstopped/Volumegiữ/mirrorb660SHAPASS/lockabsent;1197PASS2skip+13vi55PASS. Khôngnộp/browser/commit/crawl. `docs/VI55_HANDOFF.md`."
# Insert only, preserve other sessions' progress.
p.write_text(text.replace('# Tiến độ\n','# Tiến độ\n\n'+entry+'\n',1),encoding='utf-8')
