from types import SimpleNamespace
from agentprobe.eval.failure_classifier import classify


def ep(hint=None, success=False, steps=None):
    return SimpleNamespace(failure_hint=hint, success=success, steps=steps or [])

def step(action,obs=''):
    return SimpleNamespace(action=action,args={},observation=obs,guard_event=None)

def test_hint_is_used():
    f=classify({},ep('bad_arguments'), ''); assert f.label=='bad_arguments'

def test_loop_detected():
    s=[step('calculator','') for _ in range(3)]
    f=classify({'optimal_steps':1},ep(None,False,s),''); assert f.label=='loop_stuck'

def test_unknown_tool_label():
    f=classify({},ep(None,False,[step('__ghost__','Unknown tool: __ghost__')]),''); assert f.label=='hallucinated_tool'

def test_bad_arguments_label():
    f=classify({},ep(None,False,[step('calculator','Invalid arguments: ...')]),''); assert f.label=='bad_arguments'

def test_no_recovery_label():
    f=classify({},ep(None,False,[step('calculator','Injected fault: timeout')]),''); assert f.label=='no_recovery'
