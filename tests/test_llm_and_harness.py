from agentprobe.llm.mock import MockLLM
from agentprobe.agent.harness import AgentHarness
from agentprobe.safety.guardrails import Guardrails

TASK={'id':'smoke','suite':'tool_use','goal':'Calculate 2+2','allowed_tools':['calculator'],'difficulty':'easy','category':'calculator','optimal_steps':1,'grader_type':'numeric_tolerance','expected':4,'mock_plan':[{'tool':'calculator','args':{'expression':'2+2'},'thought':'Calculate the expression.'}],'mock_final_answer':'4'}

def test_mock_returns_tool_call():
    llm=MockLLM(); r=llm.chat([{'role':'user','content':'TASK_CONTEXT: {"task": '+__import__('json').dumps(TASK)+'}'}],tools=[{'type':'function','function':{'name':'calculator','parameters':{}}}],seed=1)
    assert r.tool_calls and r.tool_calls[0].name=='calculator'

def test_harness_offline_episode():
    result=AgentHarness(MockLLM(),Guardrails('block')).run(TASK,'react',0,fault_rate=0,guard_mode='block')
    assert result.final_answer=='4' and result.steps
