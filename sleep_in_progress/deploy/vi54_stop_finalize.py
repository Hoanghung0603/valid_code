"""Read-only billing + record explicit user stop; no deploy/spawn/upload."""
import json
from pathlib import Path
from vi54_control import cli

OUT=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi54')
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for p in sorted(OUT.glob('state_*.json')):
    s=json.loads(p.read_text(encoding='utf-8'));apps=cli(s['profile'],'app','list','--json',parse=True)
    own=next(x for x in apps if x['app_id']==s['app_id']);assert own['state']=='stopped' and own['tasks']=='0'
    bill=cli(s['profile'],'billing','summary','--json',parse=True)
    upper=max(s['reserved_upper'],max(0,float(bill['metered_cost'])-float(s['billing_before']['metered_cost']))+.1)
    s.update(stopped=True,stop_reason='USER STOP 31/10 21:46VN; no further vi54 work',apps_after=apps,billing_after=bill,final_upper_usd=upper)
    p.write_text(json.dumps(s,indent=2),encoding='utf-8');rows.append(s)
prior=.02363315;total=prior+sum(s['final_upper_usd'] for s in rows)
lines=['# vi54 — STOPPED theo người phụ trách 31/10 21:46VN',
 '- KHÔNG deploy/spawn/upload/build/nộp/browser/commit thêm; giữ nguyên2Volume/data/weights/rawlocal+b660.',
 '- Raw: b660 SYSTEMprepare128.968s;indexD:/r2ai/data/index/vi36;đúngexport_docs.iter_all_texts/BoilerplateFilter.clean/make_body unlimited nhưvi51.',
 '- docs54.part000–028.jsonl.zst:223065doc/609662pair;queries54.json1200;exacttậpID+thứtự29partvi51/nonempty/nooverlap/SHA31file/vi51unchangedPASS2auditlocal.',
 '- Rawlocal <HIST_TMP>/vi54/modal_input; b660D:/r2ai/data/sub/vi54/modal_input; đủ31file+manifestuploadcả2workspace,CPUremoteSHAtoànbộPASS.',
 '- Weights nvidia/Nemotron-3-Embed-8B-BF16 pin d1f2f25730bbd775b99b29185134bc86653bf2d1:CPUdownload30b54.921s/30c68.608sDONE/Volume r2ai-vi54-weights /model.',
 '- DataVolume r2ai-vi54-data /in,/out;torch2.9.1+cu128/transformers5.2.0;CPUconfig/tokenizerPASS;302262294doc-token/53650query-token.',
 '- Embeddingđịnhnghĩa:query:/passage:,Nemotron256/2048,bf16SDPA/left/meanattentionmask/L2/4096;configis_causalFalse,truyềnFalse/use_cacheFalse;khôngsửarope.',
 '- CPUreturnTorchVersiondeserializationlocal:khôretrycall;recoverđúngping.jsonđãcommit;đãsửastr(torch.__version__)trướcGPU.',
 '- Bench30cDONE115.058977055GPU-function-s;27493.392750tok/s;cosmin/mean/max.999916017/.999973893/1.000000119 PASS;sanity5queryPASS;benchoutputđãcollectlocal.',
 '- Forecast+30%19.453478148USD<25;7jobplan(4H10030b+3H10030c),NHƯNG0bulkcall/0partGPU:launchbịguardappstoppedtrướcsaveintent/spawn;KHÔNGrelaunch.',
 '- Benchcallcuối fc-01M696MCK9XNHHAGWDP4MADJH0 DONE;0bulk/cosineNemotronfull/ZIPbuild/offline;khôngcózipbyte/sha/validate/check_zip.',
 '- Fullsuite1182PASS2skip;10testvi54PASS(24chuỗivi54/52/51);py_compilePASS. Khôngsửavi52/53/crawl/dchunk.',
 '- b660 taskr2ai_vi54_prepaređãxoá/LastResult0theophiênnhườngquyền;lockabsentđãkiểmtrực tiếp;khôngxoálockngườikhác.',
 '- State/callID/rawaudit/bench/forecast/ledgerở<HIST_TMP>/vi54;giữtoànartifact đểphiênkháctáidùng.']
for s in rows:
    lines.append(f"- {s['profile']} app={s['app_id']} STOPPED/tasks0; snapshotmetered={s['billing_after']['metered_cost']}USD;vi54upper={s['final_upper_usd']:.9f}USD (snapshotdelta+CPUreserve vs functionreserve, lấymax).")
    for name,r in s['calls'].items():lines.append(f"- {s['profile']} {name} {r['call_id']} {r['status']}")
lines.append(f'- TổngGPU115.058977055s;upper={total:.9f}USD gồmpriorCPU.02363315USD;billingchưasettled/khôngquykếtlịchsửworkspacecho vi54.')
(ROOT/'docs/VI54_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'stop_report.json').write_text(json.dumps(dict(stopped=True,total_upper_usd=total,prior_cpu_usd=prior,workspaces=[dict(profile=s['profile'],upper_usd=s['final_upper_usd'],metered=s['billing_after']['metered_cost']) for s in rows]),indent=2),encoding='utf-8')
print(json.dumps(dict(total_upper_usd=total,workspaces=[dict(profile=s['profile'],upper_usd=s['final_upper_usd'],metered=s['billing_after']['metered_cost']) for s in rows])),flush=True)
