import pytest

from agentprobe.tools.builtin import BuiltinToolState, calculator, unit_convert, build_registry, read_file, sql_query


def test_calculator_addition(): assert calculator('2+3') == '5'

def test_calculator_power(): assert calculator('2**5') == '32'

def test_calculator_sqrt(): assert calculator('sqrt(144)') == '12'

def test_calculator_rejects_calls():
    with pytest.raises(ValueError): calculator('__import__("os")')

def test_unit_km_mi(): assert unit_convert(5,'km','mi').startswith('3.106855')

def test_unit_identity(): assert unit_convert(5,'km','km') == '5 km'

def test_unit_bad_conversion():
    with pytest.raises(ValueError): unit_convert(1,'kg','km')

def test_registry_unknown_tool():
    r=build_registry(BuiltinToolState())
    ok,msg,label=r.execute('does_not_exist',{})
    assert not ok and label=='hallucinated_tool'

def test_registry_bad_args():
    r=build_registry(BuiltinToolState())
    ok,msg,label=r.execute('calculator',{'expression':123})
    assert not ok and label=='bad_arguments'

def test_registry_calculator():
    r=build_registry(BuiltinToolState())
    ok,msg,label=r.execute('calculator',{'expression':'4*5'})
    assert ok and msg=='20' and label is None

def test_kv_state():
    state=BuiltinToolState(); r=build_registry(state)
    ok,_,_=r.execute('kv_store',{'key':'a','value':'b'}); assert ok
    ok,msg,_=r.execute('kv_store',{'key':'a'}); assert ok and msg=='"b"'

def test_read_file_is_sandboxed():
    with pytest.raises(ValueError): read_file('../fake_secrets.txt')

def test_sql_read_only_select():
    payload=sql_query('SELECT COUNT(*) AS n FROM employees')
    assert '"n": 6' in payload

def test_sql_write_blocked():
    with pytest.raises(ValueError): sql_query("DELETE FROM employees")
