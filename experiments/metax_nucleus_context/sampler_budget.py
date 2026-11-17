"""Conservative MetaX shortlist cost budget from existing attention metadata.

This chooses between two equivalent filtering implementations, never sampling
parameters. Long-context behavior stays native. No tensor sync or request content.
"""
_MAX_CONTEXT=None
_REQUESTS=None

def set_context(max_seq_len,num_reqs):
    global _MAX_CONTEXT,_REQUESTS
    _MAX_CONTEXT=int(max_seq_len);_REQUESTS=int(num_reqs)

def allow_shortlist(batch):
    return batch>=64 and _MAX_CONTEXT is not None and 0<_MAX_CONTEXT<=8192

def snapshot():
    return dict(max_context=_MAX_CONTEXT,requests=_REQUESTS,context_limit=8192)
