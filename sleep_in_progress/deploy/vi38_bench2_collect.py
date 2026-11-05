"""Download only the approved bench2 and audit stopped app/billing; no launch."""
import json
import os
from pathlib import Path
import subprocess

DEST = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38/bench_out2')
MODAL = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'


def main():
    os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
    audit = json.loads((DEST/'launch_audit.json').read_text(encoding='utf-8'))
    assert audit['stops'] and all(r['exit'] == 0 or 'already stopped' in r['stderr'] for r in audit['stops'])
    subprocess.run([MODAL, 'volume', 'get', 'r2ai-runner', '/out/vi38/bench_out2',
                    str(DEST), '--force'], check=True, timeout=120)
    nested = DEST/'bench_out2'
    if nested.exists():
        for p in nested.iterdir():
            assert p.is_file() and not (DEST/p.name).exists()
            p.rename(DEST/p.name)
        nested.rmdir()
    for name, args in [('apps_after.json', ['app', 'list', '--json']),
                       ('containers_after.json', ['container', 'list', '--json']),
                       ('billing_after.json', ['billing', 'report', '--for', 'today',
                                               '--show-resources', '--json'])]:
        result = subprocess.check_output([MODAL, *args], text=True, encoding='utf-8', timeout=120)
        parsed = json.loads(result)
        (DEST/name).write_text(json.dumps(parsed, indent=2), encoding='utf-8')
        if name == 'apps_after.json':
            assert all(r['state'] == 'stopped' and str(r['tasks']) == '0'
                       for r in parsed if r['app_id'] in audit['app_ids'])
        if name == 'containers_after.json':
            assert parsed == []
    print('COLLECTED', [p.name for p in sorted(DEST.iterdir()) if p.is_file()])


if __name__ == '__main__':
    main()
