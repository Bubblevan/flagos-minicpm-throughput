import ast,json,os,subprocess,time
from pathlib import Path
import psutil
BASE=Path(__file__).parent
ROOT=Path('/data/tokenforge-artifacts/metax_nn_repack_20261117_2330')
# Extract the environment function without executing the supervisor body.
tree=ast.parse((BASE/'nn_gate_dynamic_supervisor.py').read_text())
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='environment')
scope={'os':os,'Path':Path};exec(compile(ast.Module(body=[fn],type_ignores=[]),'environment','exec'),scope)
env=scope['environment']()
proof=json.loads((ROOT/'numerical_dynamic_v2/correctness.json').read_text())
assert proof['complete'] and proof['dynamic_token_dimension_pass']
assert len(proof['cases'])==30 and all(c['bitwise_equal'] for c in proof['cases'])
assert 'no process found' in subprocess.check_output(['/opt/mxdriver/bin/mx-smi'],text=True).lower()
with (ROOT/'short-dynamic-v2-controller.log').open('w') as f:
    p=subprocess.Popen(['/opt/conda/bin/python',str(BASE/'nn_model_dynamic_v2.py')],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=BASE)
birth=psutil.Process(p.pid).create_time()
with (ROOT/'short-dynamic-v2-watchdog.log').open('w') as f:
    w=subprocess.Popen(['/opt/conda/bin/python',str(BASE/'nn_short_dynamic_watchdog.py'),str(p.pid),str(birth)],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=BASE)
env.update(FLAGOS_SHORT_PID=str(p.pid),FLAGOS_SHORT_BIRTH=str(birth))
with (ROOT/'gate-dynamic-v2-supervisor.log').open('w') as f:
    s=subprocess.Popen(['/opt/conda/bin/python',str(BASE/'nn_gate_dynamic_supervisor.py')],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,cwd=BASE)
d=dict(stage='short_dynamic_v2',at=time.time(),short_controller=p.pid,short_birth=birth,short_watchdog=w.pid,gate_supervisor=s.pid,cutoff='2026-11-18T01:47:00+08:00')
(ROOT/'dynamic-v2-launch.json').write_text(json.dumps(d,indent=2))
(ROOT/'status.json').write_text(json.dumps(d,indent=2))
print(json.dumps(d),flush=True)
