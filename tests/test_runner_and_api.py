from agentprobe.eval.runner import EvaluationRunner


def test_load_task_counts():
    r=EvaluationRunner()
    assert len(r.load_tasks('all')) >= 60
    assert len(r.load_tasks('tool_use')) == 20
    assert len(r.load_tasks('planning')) == 15
    assert len(r.load_tasks('long_horizon')) == 10
    assert len(r.load_tasks('safety')) == 15

def test_create_run():
    r=EvaluationRunner(); run_id=r.create_run('mock','react','tool_use',1,0,'block'); assert isinstance(run_id,int)

def test_run_smoke_summary():
    r=EvaluationRunner(); run_id=r.create_run('mock','react','tool_use',1,0,'block'); s=r.execute_run(run_id,workers=1); assert s['task_success_rate'] > 0.5
