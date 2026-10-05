from agentprobe.scientist import ScientistLite
from agentprobe.interp import entropy, run_attention_demo


def test_scientist_hypotheses():
    h=ScientistLite().hypotheses('reduce errors'); assert len(h)==3

def test_entropy_positive():
    assert entropy([-0.1,-0.5]) > 0

def test_attention_handles_missing_optional():
    text=run_attention_demo('hello')
    assert isinstance(text,str)
