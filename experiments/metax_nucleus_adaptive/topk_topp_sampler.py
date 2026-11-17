# SPDX-License-Identifier: Apache-2.0
"""MetaX guarded nucleus shortlist; sampling settings and RNG remain unchanged."""
import atexit,hashlib,json,os,time
from pathlib import Path
import torch
import vllm.v1.sample.ops.topk_topp_sampler as topk_topp_sampler
from .top_p_shortlist import filter_logits,STATS

_mode=os.environ.get('FLAGOS_METAX_TOPP_MODE','native')
_control=os.environ.get('FLAGOS_METAX_TOPP_DIAGNOSTIC_CONTROL')
_root=os.environ.get('FLAGOS_SESSION_ROOT')
_previous=None;_calls=0;_shadow=0
def record(mode):
    if not _root:return
    data=dict(at=time.time(),pid=os.getpid(),mode=mode,calls=_calls,shadow_equal_calls=_shadow,stats=dict(STATS),
        source_file=__file__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).with_name('top_p_shortlist.py').read_bytes()).hexdigest())
    p=Path(_root)/('sampler-runtime-'+str(os.getpid())+'.json');t=p.with_suffix('.tmp');t.write_text(json.dumps(data,indent=2));t.replace(p)
def _apply(logits,k,p):
    global _previous,_calls,_shadow
    _calls+=1
    mode=Path(_control).read_text().strip() if _control else _mode
    if mode!=_previous or _calls%1024==0:record(mode);_previous=mode
    native=topk_topp_sampler.apply_top_k_top_p_pytorch
    if mode=='shadow':
        expected=native(logits.clone(),k,p)
        actual=filter_logits(logits.clone(),k,p,native)
        if not torch.equal(expected,actual):
            record('shadow_mismatch');raise RuntimeError('Nucleus support changed; reject candidate')
        _shadow+=1
        if _shadow%32==0:record(mode)
        return expected
    if mode=='candidate':return filter_logits(logits,k,p,native)
    return logits if k is None and p is None else native(logits,k,p)
atexit.register(lambda:record(_previous))
topk_topp_sampler.apply_top_k_top_p=_apply
