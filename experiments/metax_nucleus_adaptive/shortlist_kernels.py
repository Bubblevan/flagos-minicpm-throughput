"""Monotone-pivot statistics and bounded compaction, native sampling fallback."""
import triton
import triton.language as tl

@triton.jit
def _init(C,M,VALUES,IDS,PACKED_COUNTS,CAP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0)
    j=tl.arange(0,16)
    tl.store(C+row*9+j,0,j<9);tl.store(M+row*9+j,0.,j<9)
    tl.store(PACKED_COUNTS+row,0)
    for tile in tl.static_range(0,(CAP+BLOCK-1)//BLOCK):
        i=tile*BLOCK+tl.arange(0,BLOCK)
        tl.store(VALUES+row*CAP+i,-float('inf'),i<CAP)
        tl.store(IDS+row*CAP+i,-1,i<CAP)

@triton.jit
def _stats(X,P,MAX,C,M,V:tl.constexpr,SX:tl.constexpr,SP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);tile=tl.program_id(1)
    i=tile*BLOCK+tl.arange(0,BLOCK);valid=i<V
    x=tl.load(X+row*SX+i,valid,other=-float('inf'))
    prob=tl.load(P+row*SP+i,valid,other=0.)
    maximum=tl.load(MAX+row)
    for j in tl.static_range(9):
        delta=4.+j*2.
        keep=valid & (x>=maximum-delta)
        count=tl.sum(keep.to(tl.int32),0);mass=tl.sum(tl.where(keep,prob,0.),0)
        tl.atomic_add(C+row*9+j,count,sem='relaxed')
        tl.atomic_add(M+row*9+j,mass,sem='relaxed')

@triton.jit
def _pack(X,PIVOT,VALUES,IDS,COUNTS,V:tl.constexpr,SX:tl.constexpr,CAP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);tile=tl.program_id(1)
    i=tile*BLOCK+tl.arange(0,BLOCK);valid=i<V
    x=tl.load(X+row*SX+i,valid,other=-float('inf'));pivot=tl.load(PIVOT+row)
    take=valid & (x>=pivot) & (x<float('inf'))
    n=tl.sum(take.to(tl.int32),0)
    start=tl.atomic_add(COUNTS+row,n,sem='relaxed')
    rank=tl.cumsum(take.to(tl.int32),0)-1+start
    write=take & (rank<CAP)
    tl.store(VALUES+row*CAP+rank,x,write)
    tl.store(IDS+row*CAP+rank,i,write)

@triton.jit
def _restore(X,VALUES,IDS,SX:tl.constexpr,CAP:tl.constexpr,BLOCK:tl.constexpr):
    row=tl.program_id(0);i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    valid=i<CAP;idx=tl.load(IDS+row*CAP+i,valid,other=-1)
    value=tl.load(VALUES+row*CAP+i,valid,other=-float('inf'))
    tl.store(X+row*SX+idx,value,valid & (idx>=0))

def stats(logits,probs,maximum,counts,masses):
    _stats[(logits.shape[0],triton.cdiv(logits.shape[1],1024))](logits,probs,maximum,counts,masses,
        logits.shape[1],logits.stride(0),probs.stride(0),1024,num_warps=4)
def initialize(counts,masses,values,ids,packed_counts):
    _init[(values.shape[0],)](counts,masses,values,ids,packed_counts,values.shape[1],1024,num_warps=4)
def pack(logits,pivot,values,ids,counts):
    _pack[(logits.shape[0],triton.cdiv(logits.shape[1],1024))](logits,pivot,values,ids,counts,
        logits.shape[1],logits.stride(0),values.shape[1],1024,num_warps=4)
def restore(logits,values,ids):
    _restore[(logits.shape[0],triton.cdiv(values.shape[1],256))](logits,values,ids,logits.stride(0),values.shape[1],256,num_warps=4)
