from agentprobe.eval.metrics import bootstrap_ci, metric_summary, paired_bootstrap


def test_bootstrap_ci_constant():
    lo,hi=bootstrap_ci([1,1,1],iterations=100); assert lo==1 and hi==1

def test_metric_success():
    s=metric_summary([{'success':True,'steps_used':1,'tokens':10,'latency_ms':5,'actions':['calculator'],'allowed_tools':['calculator'],'optimal_steps':1,'had_fault':False,'is_attack':False,'attack_success':False,'benign_control':False}])
    assert s['task_success_rate']==1

def test_metric_recovery():
    s=metric_summary([{'success':True,'steps_used':2,'tokens':10,'latency_ms':5,'actions':['calculator'],'allowed_tools':['calculator'],'optimal_steps':1,'had_fault':True,'is_attack':False,'attack_success':False,'benign_control':False}])
    assert s['recovery_rate']==1

def test_metric_attack():
    s=metric_summary([{'success':True,'steps_used':1,'tokens':10,'latency_ms':5,'actions':['read_file'],'allowed_tools':['read_file'],'optimal_steps':1,'had_fault':False,'is_attack':True,'attack_success':True,'benign_control':False}])
    assert s['attack_success_rate']==1

def test_paired_bootstrap_significance():
    r=paired_bootstrap([0,0,0,0],[1,1,1,1],iterations=200)
    assert r['difference']==1 and r['significant'] is True

def test_paired_bootstrap_none():
    r=paired_bootstrap([],[]); assert r['difference']==0 and r['significant'] is False
