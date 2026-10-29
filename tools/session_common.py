"""Shared bounded diagnostic workload; timings are NOT competition scores.

The source comparison uses the fixed server arguments without a profiler.
No model, dataset or original installed source is modified.
"""
import asyncio
import datetime as dt
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import threading
import time

import aiohttp
import psutil
import requests

ROOT = Path(os.environ['FLAGOS_SESSION_ROOT'])
URL = 'http://127.0.0.1:9031'
DEADLINE = float(os.environ['FLAGOS_SESSION_DEADLINE'])


def state(stage, **kwargs):
    value = dict(stage=stage, at=dt.datetime.now(dt.timezone.utc).isoformat(), **kwargs)
    p = ROOT/'status.tmp'
    p.write_text(json.dumps(value, indent=2))
    p.replace(ROOT/'status.json')
    print(json.dumps(value), flush=True)


def textfile(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def telemetry(pid, stop):
    with (ROOT/'telemetry.jsonl').open('a') as f:
        while not stop.is_set():
            row = dict(at=time.time(), cpu_stat=textfile('/sys/fs/cgroup/cpu/cpu.stat'),
                       memory_usage=textfile('/sys/fs/cgroup/memory/memory.usage_in_bytes'),
                       host_load=os.getloadavg(), processes=[])
            try:
                parent = psutil.Process(pid)
                for p in [parent] + parent.children(recursive=True):
                    try:
                        c = p.cpu_times()
                        row['processes'].append(dict(pid=p.pid, name=p.name(), threads=p.num_threads(),
                            rss=p.memory_info().rss, cpu_user=c.user, cpu_system=c.system,
                            affinity=p.cpu_affinity()))
                    except psutil.Error:
                        pass
                row['gpu'] = subprocess.check_output(['ixsmi','-q','-d','CLOCK,PERFORMANCE,POWER,UTILIZATION'], text=True, timeout=5)
            except Exception as e:
                row['telemetry_error'] = repr(e)
            f.write(json.dumps(row)+'\n'); f.flush()
            stop.wait(5)


async def workload(n, length, output, seed, profile_mode=None):
    rng = random.Random(seed)
    prompts = [[rng.randrange(1000,100000) for _ in range(length)] for _ in range(n)]
    first = set()
    all_first = asyncio.Event()
    timeout = aiohttp.ClientTimeout(total=600)
    rows = []
    started = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as client:
        async def profile(path):
            async with client.post(URL+path) as r:
                body = await r.text()
                if r.status != 200:
                    raise RuntimeError(f'{path}: {r.status}: {body[:500]}')
        async def one(i, prompt):
            t = time.monotonic(); ttft = None; usage = None; reason = None
            body = dict(model='minicpm', prompt=prompt, max_tokens=output, ignore_eos=True,
                        stream=True, stream_options={'include_usage':True})
            async with client.post(URL+'/v1/completions', json=body) as r:
                if r.status != 200:
                    raise RuntimeError(f'Request failed {r.status}: {(await r.text())[:500]}')
                async for line in r.content:
                    if not line.startswith(b'data: '):
                        continue
                    raw = line[6:].strip()
                    if raw == b'[DONE]':
                        continue
                    chunk = json.loads(raw)
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    for choice in chunk.get('choices',[]):
                        if choice.get('text') and ttft is None:
                            ttft = time.monotonic()-t; first.add(i)
                            if len(first)==n:
                                all_first.set()
                        reason = choice.get('finish_reason') or reason
            if ttft is None:
                raise RuntimeError('No generated content')
            rows.append(dict(id=i,ttft_seconds=ttft,duration_seconds=time.monotonic()-t,
                             usage=usage,finish_reason=reason))
        async def capture_decode():
            await asyncio.wait_for(all_first.wait(), timeout=300)
            await profile('/start_profile')
            state('profiling_decode', requests=n, input_len=length, output_len=output)
        if profile_mode=='prefill':
            await profile('/start_profile')
        tasks = [asyncio.create_task(one(i,p)) for i,p in enumerate(prompts)]
        if profile_mode=='decode':
            tasks.append(asyncio.create_task(capture_decode()))
        try:
            await asyncio.gather(*tasks)
        finally:
            if profile_mode:
                await profile('/stop_profile')
    return dict(requests=n,input_len=length,output_len=output,seed=seed,
                profile_mode=profile_mode,elapsed_seconds=time.monotonic()-started,
                rows=sorted(rows,key=lambda x:x['id']),official_score=False)

