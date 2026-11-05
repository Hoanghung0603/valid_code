"""List exact final mirror filenames and SHA, for explicit SFTP uploads."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import write,fingerprint,read
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55')
assert read(T/'phase1_report.json')['status']=='DONE'
names=['phase1_report.json','q3e8b_full_top1000.json.zst','vi53_exact_ranks.json','shared_query_embeddings.npz',
       'shared_query_audit.json','recovery_part_manifest.json','recovery_plan.json','recovery_tests_final.xml',
       'input_audit.json','recovery_state_<MODAL_PROFILE_B>.json','recovery_state_<MODAL_PROFILE_C>.json','recovery_state_<MODAL_PROFILE_D>.json']
write(T/'recovery_mirror_manifest.json',{n:fingerprint(T/n) for n in names})
print('\n'.join(names),flush=True)
