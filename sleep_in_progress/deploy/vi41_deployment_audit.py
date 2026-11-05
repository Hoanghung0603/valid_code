"""Snapshot local deployment/test sources and immutable root inputs; no job launch."""
import hashlib
import json
from pathlib import Path
from vi41_local import ROOT,TEMP

paths=[ROOT/'scripts/vi41.py',ROOT/'deploy/modal_vi41_search.py',ROOT/'deploy/modal_vi41_score.py',
       ROOT/'deploy/vi41_resume.py',ROOT/'tests/test_vi41.py',ROOT/'tests/test_vi41_build.py',
       ROOT/'tests/test_vi41_packaging.py',ROOT/'tests/test_vi41_stats.py']
report={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths}
(TEMP/'code_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'files':len(report),'sha256_vi41':report['scripts\\vi41.py']['sha256']}))
