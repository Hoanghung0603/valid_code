"""Static deployment guards; remote CPU pings provide the actual import check."""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def test_modal_files_self_contained_no_sibling_imports():
    for name in ('modal_vi41_search.py','modal_vi41_score.py'):
        tree=ast.parse((ROOT/'deploy'/name).read_text())
        imported={node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
        assert 'modal_dchunk' not in imported
        assert 'modal_runner' not in imported
        funcs={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
        assert 'ping' in funcs and 'guarded' in funcs
        text=(ROOT/'deploy'/name).read_text()
        assert 'init_failed.json' in text and 'CPU ping required' in text
        for node in funcs.values():
            for decorator in node.decorator_list:
                if isinstance(decorator,ast.Call) and isinstance(decorator.func,ast.Attribute) and decorator.func.attr=='function':
                    keys={k.arg:k.value for k in decorator.keywords}
                    assert isinstance(keys['retries'],ast.Constant) and keys['retries'].value==0
                    if 'gpu' in keys:
                        assert keys['max_containers'].value==1


def test_ping_versions_serialize_plain_strings():
    search=(ROOT/'deploy/modal_vi41_search.py').read_text()
    score=(ROOT/'deploy/modal_vi41_score.py').read_text()
    assert "str(torch.__version__)" in search
    assert "str(getattr(importlib.import_module(n),'__version__','import-ok'))" in score


def test_every_gpu_launch_requires_cpu_ping_and_total_budget():
    text=(ROOT/'deploy/vi41_resume.py').read_text()
    assert 'Fresh CPU import ping before EACH GPU launch' in text
    assert 'upper+reserve+remaining+.12<10' in text
    assert "cmd('app','list','--json')" in text
    assert "status='spawn_intent'" in text
    assert "record.update(status='running',call_id=call.object_id)" in text


def test_all_owned_deployment_scripts_compile():
    for p in (ROOT/'deploy').glob('*vi41*.py'):
        compile(p.read_text(encoding='utf-8'),str(p),'exec')
