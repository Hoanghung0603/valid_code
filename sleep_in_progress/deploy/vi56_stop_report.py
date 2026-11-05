"""Record mandatory stop after same packaging error twice; never resumes jobs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vi56 import read,write,TEMP
out=TEMP/'vi56';state=read(out/'state.json');apps=read(out/'apps_final.json')
assert state['stopped'] and all(r['status'] in ('done','failed_terminal') for r in state['calls'].values())
assert any(a['app_id']==state['app_id'] and a['state']=='stopped' and str(a['tasks'])=='0' for a in apps)
before=float(state['baseline']['metered_cost']);after=float(state['after']['metered_cost'])
delta=max(0,after-before)
upper=max(state['upper'],delta+.1)
report=dict(status='STOPPED_SAME_LOGIC_ERROR_TWICE',reason='Both bench subprocesses: ModuleNotFoundError modal before model loading; no retry/bulk/build',
  raw_pairs=32150,eligible_pairs=32088,eligible_unique_docs=25711,q3e_pairs=32088,
  sources_scored_new=['q3e8b from vi55'],sources_not_scored=['bge_doc','q8b_doc','bge_docwin','d8w'],
  fallback=['dense_bge_m3 historical last rank','HyDE historical last rank'],
  gpu_model_seconds=0,gpu_allocation_exact_seconds=None,gpu_seconds_reserved=400,
  workspace='<MODAL_PROFILE_D>/<MODAL_PROFILE>',billing_before=before,billing_after=after,snapshot_delta=delta,usd_upper=upper,
  sanity=read(out/'sanity.json'),zip_build_attempts=0,offline_final_unavailable=True,
  tests_passed=1205,tests_skipped=2,vi56_tests=8,chain_tests=32,app_stopped=True,tasks=0,volumes_kept=True)
write(out/'stop_report.json',report)
lines=['# vi56 — DỪNG BẮT BUỘC: cùng lỗi đóng gói 2 lần; CHƯA ZIP',
 '- Hai bench qwen/bge cùng ModuleNotFoundError modal trong subprocess trước model loading;khôngretry/respawn/bulk.',
 '- Raw top100ngoàipool32150cặp/25757doc;lọc61bad lịch sử+1bodyrỗng→32088cặp/25711doc.',
 '- Đãcó:q3e8B32088cosinevi55;chưachấm:bge doc/8Bdoc/bge docwin/d8w;khôngbuildvớithiếuđiểm.',
 '- Nguồnvi53:bge doc×1/8Bdoc×2/densebge-m3×2/HyDEdense×1/bgeDocwin×1/d8w×1/q3e8B×1;k60.',
 '- Dense/HyDE cosine lịch sử ngoài top200khôngcó→fallbackđúngvi41.append_last_ranks;giữordercũ.',
 '- q3eunioncũ2048token/ngoàiunion1024/title+clean/fp16/LAST/L2;khôngembedlại.',
 '- Prepareb660106.125s/15partSHA/exact281003identityPASS;32149bgedoc+159192win+32149q8bdoc+57513d8wrawsuperset.',
 '- Forecastbulk+30%5.394558580USD<10;CPUweightsPIN99.362767s+ping15partSHA/tokenizerPASS.',
 '- WeightsQwen77d193c791ed757ca307ee72715aa132723da912;bge953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e.',
 '- Sanitytắtdocmới=vi53đúngthứtự1200/1200;changedsets0;chưaofflinefinal/0ZIPbuild/khôngbyteSHA.',
 '- Full1205PASS2skip/8vi56PASS/32chuỗiPASS;testpreemptionrerun=clean/max1rerun/exactID+SHAPASS.',
 f'- App{state["app_id"]}STOPPED/tasks0;giữ2Volumeweights+data;taskpreparexoá/lockabsent.',
 f'- 30dsnapshotbefore={before:.8f}/after={after:.8f}/delta={delta:.8f}USD;upper={upper:.8f}<10;30b/c=0.',
 '- GPUmodel0s/allocationexactchưađủ;benchwatchdogreserved240+160=400s;khôngcóbulkcall.',
 '- Local<HIST_TMP>/vi56;mirrorb660D:/r2ai/data/sub/vi56;khôngsửavi34–55/dchunk/crawl.',
 '- Chưakiểm:gold/F2/offlinefinal/ZIP/billingsettled;chờduyệtrecoveryriêng;KHÔNGnộp/browser/commit.']
for n,r in state['calls'].items():lines.append(f'- {n}:{r["call_id"]}:{r["status"]}')
(Path(__file__).resolve().parents[1]/'docs/VI56_HANDOFF.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(report,flush=True)
