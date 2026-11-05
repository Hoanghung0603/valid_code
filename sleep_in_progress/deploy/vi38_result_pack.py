"""Small final reports pack after finish+independent audit, no submission/texts."""
from pathlib import Path
import zipfile

base = Path('D:/r2ai/data/sub/vi38')
names = ['merge_report.json', 'confirm_report.json', 'cmerge_report.json',
         'build_report_vi38b.json', 'stats_report_vi38b.json', 'gate2_report.json',
         'independent_audit_vi38b.json', 'exchanges_vi38b.json', 'vi38b_status.json',
         'vi38b_prepare.log', 'vi38b_finish.log', 'vi38b_audit.log']
with zipfile.ZipFile(base/'vi38b_reports.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in names:
        assert (base/name).exists(), name
        z.write(base/name, name)
print('REPORT PACK', (base/'vi38b_reports.zip').stat().st_size, flush=True)
