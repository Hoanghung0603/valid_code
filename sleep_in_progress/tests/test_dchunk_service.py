import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def service():
    import sys
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'deploy'))
    spec=importlib.util.spec_from_file_location('dchunk_service',root/'deploy/dchunk_service.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_deployed_spawn_persists_intent_and_id(tmp_path,monkeypatch):
    m=service()
    monkeypatch.setattr(m,'STATE',tmp_path/'state.json')
    monkeypatch.setattr(m,'handoff',lambda *a:None)
    seen=[]
    def spawn(*args):
        assert json.loads(m.STATE.read_text())['calls']['embed']['status']=='spawn_intent'
        seen.append(args)
        return SimpleNamespace(object_id='fc-test')
    modal=SimpleNamespace(Function=SimpleNamespace(from_name=lambda app,fn:SimpleNamespace(spawn=spawn)))
    state=dict(prior_upper_usd=2.4,calls={})
    m.durable_spawn(state,'embed','bulk_h100',(800,'full',True,'embed'),1.3,modal)
    assert state['calls']['embed']['call_id']=='fc-test' and seen==[(800,'full',True,'embed')]
    with pytest.raises(AssertionError,match='No automatic relaunch'):
        m.durable_spawn(state,'embed','bulk_h100',(),1,modal)


def test_ambiguous_spawn_never_retries(tmp_path,monkeypatch):
    m=service()
    monkeypatch.setattr(m,'STATE',tmp_path/'state.json')
    def spawn(*args):
        raise ConnectionError('DNS')
    modal=SimpleNamespace(Function=SimpleNamespace(from_name=lambda *a:SimpleNamespace(spawn=spawn)))
    state=dict(prior_upper_usd=2.4,calls={})
    with pytest.raises(ConnectionError):
        m.durable_spawn(state,'embed','bulk_h100',(),1.3,modal)
    state=json.loads(m.STATE.read_text())
    assert state['calls']['embed']['status']=='spawn_intent'
    with pytest.raises(AssertionError):
        m.durable_spawn(state,'embed','bulk_h100',(),1.3,modal)


def test_budget_blocks_before_network(tmp_path,monkeypatch):
    m=service()
    monkeypatch.setattr(m,'STATE',tmp_path/'state.json')
    with pytest.raises(AssertionError):
        m.durable_spawn(dict(prior_upper_usd=4,calls={}), 'search','search_h100',(),1,None)
    assert not m.STATE.exists()


@pytest.mark.parametrize('error',[TimeoutError, type('ModalTimeout',(Exception,),{})])
def test_poll_timeout_is_running_not_failure(tmp_path,monkeypatch,error):
    m=service()
    monkeypatch.setattr(m,'STATE',tmp_path/'state.json')
    def get(**kwargs):
        raise error()
    modal=SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda *a:SimpleNamespace(get=get)),
                          exception=SimpleNamespace(TimeoutError=error if error is not TimeoutError else type('OtherTimeout',(Exception,),{})))
    state=dict(prior_upper_usd=2.4,calls=dict(audit=dict(call_id='fc-test',status='running',upper_usd=.05)))
    m.poll(state,'audit',modal)
    assert state['calls']['audit']['status']=='running'
    assert not m.STATE.exists()


def test_launcher_only_deployed_and_split_phases():
    m=service()
    text=Path(m.__file__).read_text()
    assert 'modal.Function.from_name(APP,function).spawn(*args)' in text
    assert 'modal.FunctionCall.from_id' in text
    assert "cmd('deploy','deploy/modal_dchunk.py')" in text
    assert "'run','--detach'" not in text
    assert "cmd('volume','put','--force'" in text
    server=(Path(m.__file__).parent/'modal_dchunk.py').read_text()
    assert "assert phase!='search', 'Search requires every part valid; never embeds'" in server
    assert "if phase=='embed':" in server
    assert 'Unsafe local-entrypoint resume disabled' in server


@pytest.mark.parametrize('phase',['audit','audit-final'])
def test_poll_result_downloads_with_overwrite_without_spawn(tmp_path,monkeypatch,phase):
    m=service()
    monkeypatch.setattr(m,'STATE',tmp_path/'state.json')
    monkeypatch.setattr(m,'TEMP',tmp_path)
    monkeypatch.setattr(m,'handoff',lambda *a:None)
    commands=[]
    monkeypatch.setattr(m,'poll_cmd',lambda *args:commands.append(args))
    modal=SimpleNamespace(FunctionCall=SimpleNamespace(from_id=lambda *a:SimpleNamespace(get=lambda **kw:dict(good=[]))),
                          exception=SimpleNamespace(TimeoutError=TimeoutError))
    state=dict(prior_upper_usd=2.4,calls={phase:dict(call_id='fc-test',status='running',upper_usd=.05)})
    m.poll(state,phase,modal)
    assert state['calls'][phase]['status']=='done' and state['calls'][phase]['downloaded']
    assert len(commands)==2 and all(c[:3]==('volume','get','--force') for c in commands)
    m.poll(state,phase,modal)
    assert len(commands)==2
