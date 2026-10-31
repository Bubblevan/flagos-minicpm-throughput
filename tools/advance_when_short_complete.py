"""One reserved-session advancement, not a recurring chat automation."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
reservation=json.loads((ROOT/'reservation.json').read_text())
latest_start=reservation['deadline_epoch']-4*3600


def decide(data):
    data.update(at=dt.datetime.now(dt.timezone.utc).isoformat(),chat_automation_created=False)
    (ROOT/'advancement-decision.json').write_text(json.dumps(data,indent=2))
    print(json.dumps(data),flush=True)


while time.time()<latest_start:
    try:
        status=json.loads((ROOT/'status.json').read_text())
        if status['stage']=='mixed_short_failed':
            decide(dict(full_validation_started=False,reason='Short-trial execution failed; inspect original error.'));break
        summary_path=ROOT/'short-comparison/summary.json'
        if summary_path.exists():
            summary=json.loads(summary_path.read_text())
            if summary.get('comparison_complete'):
                if not summary.get('go_full_validation'):
                    decide(dict(full_validation_started=False,reason='Preregistered short-trial gain/TTFT gate did not pass.',relative_gain=summary.get('relative_gain')));break
                # Child script rechecks the exact evidence, hashes, idle port, and four-hour budget.
                result=subprocess.run([sys.executable,str(ROOT/'prepare_full_mixed_validation.py'),'--start'],capture_output=True,text=True)
                decide(dict(full_validation_started=result.returncode==0,exit_code=result.returncode,
                            result=result.stdout[-2000:],error=result.stderr[-2000:]));break
    except (OSError,ValueError,KeyError):pass
    time.sleep(10)
else:
    decide(dict(full_validation_started=False,reason='Four-hour full-validation budget no longer available; no forced run.'))
