"""Write local artifact inventory only; no network/job launch."""
import hashlib
import json
from vi41_local import TEMP

names=['rewrites_complete.jsonl','candidate_report.json','input_audit_report.json','transfer_manifest.json',
       'volume_sha_audit.json','text_audit.json','sanity_report.json','sanity_chain_report.json',
       'rr_audit.json','docwin_audit.json','qselect_audit.json','preaudit_report.json',
       'pre_chunks_report.json','preverify_report.json','q8b_audit.json','build_report.json',
       'stats_report.json','local_final_audit.json','local_compare.json','original_inputs_final_check.json',
       'source_overlap_report.json','candidate_logit_report.json','code_audit.json']
result={n:dict(bytes=(TEMP/n).stat().st_size,sha256=hashlib.sha256((TEMP/n).read_bytes()).hexdigest())
        for n in names if (TEMP/n).exists()}
(TEMP/'artifact_inventory.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'artifact_files':len(result),'all_named_inputs_present':len(result)==len(names)}))
