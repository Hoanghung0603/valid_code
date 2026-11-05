"""Read-only billing snapshot and conservative per-turn/vi38 cumulative report."""
import json
import os
from pathlib import Path
import subprocess

base = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
modal = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'
os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
raw = subprocess.check_output([modal, 'billing', 'report', '--for', 'today', '--show-resources', '--json'],
                              text=True, encoding='utf-8')
(base/'billing_vi38b.json').write_text(raw, encoding='utf-8')
data = json.loads(raw)
ids = []
upper = 0
for mode in ('bulk', 'confirm'):
    p = base/(mode+'_out/launch_audit.json')
    if p.exists():
        audit = json.loads(p.read_text(encoding='utf-8'))
        ids.extend(audit.get('app_ids', []))
        upper += audit.get('all_wall_h100_upper_usd', 0)+audit.get('reserve_egress_usd', 0)
costs = {}
for row in data:
    costs[row['object_id']] = costs.get(row['object_id'], 0)+float(row['cost'])
old = ['ap-78J1sEduPeGybQqGEJoO58', 'ap-k7Z57heAZzuZ49CeuSQwKV']
report = dict(app_costs_snapshot={i: costs.get(i, 0) for i in old+ids},
              turn_billing_snapshot=sum(costs.get(i, 0) for i in ids),
              vi38_billing_snapshot=sum(costs.get(i, 0) for i in old+ids),
              turn_upper_usd=upper, vi38_upper_usd=1.140796+upper,
              note='Billing snapshot may lag settlement; upper includes all client cold-start/error/cancel wall')
assert upper < 8
(base/'cost_report_vi38b.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
