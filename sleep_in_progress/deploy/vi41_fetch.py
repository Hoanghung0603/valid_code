"""Print explicit SFTP transfer list from own manifests; never runs SSH/shell."""
import json
from vi41_local import TEMP

(TEMP/'modal_input').mkdir(exist_ok=True)
manifest=TEMP/'transfer_manifest.json'
if manifest.exists():
    report=json.loads(manifest.read_text())
    print(json.dumps([dict(remote='/D:/r2ai/data/sub/vi41/modal_input/'+r['name'],
        local=str(TEMP/'modal_input'/r['name']),bytes=r['bytes'],sha256=r['sha256'])
        for r in report['files']],indent=2))
else:
    print('Fetch transfer_manifest.json and export_report.json from b660 first.')
