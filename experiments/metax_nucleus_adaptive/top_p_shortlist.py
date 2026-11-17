"""Mass-certified nucleus shortlist; original full sort is the fallback.

No sampling parameter, temperature, RNG draw, or token probability is changed.
For top-p-only rows whose nucleus fits in a bounded shortlist, discarded-tail
probability precedes that shortlist in the original ascending cumulative sum.
Floating-point boundary uncertainty and split ties use the original algorithm.
The output retains the original logits verbatim, masking only rejected tokens.
"""
import torch
try:
    from . import shortlist_kernels as kernels
except ImportError:
    import shortlist_kernels as kernels

CAPACITY=4096
BOUNDARY_MARGIN=1e-5
STATS={'calls':0,'shortlist_rows':0,'fallback_rows':0,'cost_bypass_calls':0}
# Recent certified-row coverage is a cost hint only: bypass always calls the
# original filter, without changing sampling settings or RNG. Re-probe every
# eighth otherwise-eligible call, so changing live distributions recover.
_COVERAGE=0.5
_COOLDOWN=0

def reset_cost_hint():
    global _COVERAGE,_COOLDOWN
    _COVERAGE=0.5;_COOLDOWN=0

def update_cost_hint(count,batch):
    global _COVERAGE,_COOLDOWN
    _COVERAGE=0.75*_COVERAGE+0.25*(count/batch)
    if _COVERAGE<0.5:_COOLDOWN=7


def filter_logits(logits,k,p,native):
    global _COOLDOWN
    STATS['calls']+=1
    if (logits.shape[0]<64 or k is not None or p is None or logits.ndim!=2 or logits.dtype!=torch.float32
        or not logits.is_cuda or logits.stride(-1)!=1 or logits.shape[1]<=CAPACITY):
        STATS['fallback_rows']+=logits.shape[0]
        return native(logits,k,p)
    if _COOLDOWN:
        _COOLDOWN-=1;STATS['cost_bypass_calls']+=1
        STATS['fallback_rows']+=logits.shape[0]
        return native(logits,k,p)
    probs=logits.softmax(dim=-1)
    maximum=logits.amax(dim=-1)
    counts=torch.empty((logits.shape[0],9),device=logits.device,dtype=torch.int32)
    masses=torch.empty_like(counts,dtype=torch.float32)
    values=torch.empty((logits.shape[0],CAPACITY),device=logits.device,dtype=logits.dtype)
    ids=torch.empty((logits.shape[0],CAPACITY),device=logits.device,dtype=torch.int32)
    packed_counts=torch.empty((logits.shape[0],),device=logits.device,dtype=torch.int32)
    kernels.initialize(counts,masses,values,ids,packed_counts)
    kernels.stats(logits,probs,maximum,counts,masses)
    eligible=(counts<=CAPACITY) & (counts>0) & (masses>p.unsqueeze(1)+BOUNDARY_MARGIN)
    exists=eligible.any(dim=-1)
    # Do not compact/sort the bounded scratch if too few rows can benefit.
    possible=int(exists.sum().item())
    if possible*2<logits.shape[0]:
        update_cost_hint(possible,logits.shape[0])
        STATS['fallback_rows']+=logits.shape[0]
        return native(logits,k,p)
    choice=eligible.to(torch.int32).argmax(dim=-1)
    pivot=maximum-(4.+choice*2.)
    pivot=torch.where(exists,pivot,float('inf'))
    kernels.pack(logits,pivot,values,ids,packed_counts)
    asc_values,order=values.sort(dim=-1)
    asc_ids=ids.gather(1,order)
    selected=probs.gather(1,asc_ids.clamp_min(0).long()).masked_fill(asc_ids<0,0.)
    mass=selected.sum(dim=-1)
    cumulative=selected.cumsum(dim=-1)+(1.0-mass).unsqueeze(1)
    boundary=1-p.unsqueeze(1)
    masked=cumulative<=boundary
    masked[:,-1]=False
    split_tie=((asc_values[:,1:]==asc_values[:,:-1]) & (masked[:,1:]!=masked[:,:-1])).any(dim=-1)
    safe=(exists & (packed_counts<=CAPACITY) & (mass>p+BOUNDARY_MARGIN) & torch.isfinite(mass)
          & ((cumulative-boundary).abs().amin(dim=-1)>BOUNDARY_MARGIN) & ~split_tie)
    count=int(safe.sum().item())
    update_cost_hint(count,logits.shape[0])
    STATS['shortlist_rows']+=count
    STATS['fallback_rows']+=logits.shape[0]-count
    if count==0:
        return native(logits,k,p)
    fallback_ids=None;fallback_values=None
    if count!=logits.shape[0]:
        fallback_ids=(~safe).nonzero().flatten()
        fallback_values=native(logits.index_select(0,fallback_ids),None,p.index_select(0,fallback_ids))
    result_values=asc_values.masked_fill(masked,-float('inf'))
    logits.fill_(-float('inf'));kernels.restore(logits,result_values,asc_ids)
    if fallback_ids is not None:logits.index_copy_(0,fallback_ids,fallback_values)
    return logits
