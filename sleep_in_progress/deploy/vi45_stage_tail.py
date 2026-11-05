"""CPU-only upload of the remaining fixed-order parts; does NOT spawn GPU."""
import json
from vi45_bulk_local import TEMP,cli

profile=__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>")
manifest=json.loads((TEMP/'prepare_vi44_report.json').read_text())
names=[f'pairs.part{i:03}.jsonl.zst' for i in range(14,18)]+[f'bridge44.part{i:03}.jsonl.zst' for i in range(2)]
for n in names:
    cli(profile,'volume','put','r2ai-runner',str(TEMP/'modal_input'/n),'/in/vi45/jobs/new014/'+n)
parts=[dict(name=n,sha256=manifest['files'][n]['sha256'],count=manifest['counts'][n]) for n in names]
config=TEMP/'new014.job.json'
config.write_text(json.dumps(dict(parts=parts)))
cli(profile,'volume','put','r2ai-runner',str(config),'/in/vi45/jobs/new014/job.json')
print('STAGED_ONLY '+json.dumps(dict(parts=names,pairs=sum(p['count'] for p in parts))))
