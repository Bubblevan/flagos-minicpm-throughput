"""Bounded Range download from the official index; final index SHA is mandatory."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import time
import urllib.request
import sys

URL='https://resource.flagos.net/repository/flagos-pypi-hosted/packages/flagtree/0.7.0+iluvatar3.6/flagtree-0.7.0+iluvatar3.6-cp312-cp312-linux_x86_64.whl'
SIZE=359492273
SHA='af6a7c9eddf167441fd872c3e6d673e9da572608dee79dba655b0775db740176'


def main():
    dest=Path(sys.argv[1]);dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists()
    deadline=time.time()+1100
    def piece(i):
        first=SIZE*i//8;last=SIZE*(i+1)//8-1
        path=dest.with_suffix('.part'+str(i))
        offset=path.stat().st_size if path.exists() else 0
        assert offset<=last-first+1
        for attempt in range(10):
            if offset==last-first+1:return path
            assert time.time()<deadline
            req=urllib.request.Request(URL,headers={'Range':f'bytes={first+offset}-{last}'})
            try:
                with urllib.request.urlopen(req,timeout=40) as response:
                    assert response.status==206
                    assert response.headers['Content-Range']==f'bytes {first+offset}-{last}/{SIZE}'
                    with path.open('ab') as out:
                        while True:
                            assert time.time()<deadline
                            data=response.read(1024*256)
                            if not data:break
                            out.write(data);offset+=len(data)
                if offset<last-first+1:
                    print(f'part {i} short HTTP body; resume at {offset}',flush=True)
                    continue
                assert offset==last-first+1
                print(f'part {i} complete {offset}',flush=True)
                return path
            except (OSError,TimeoutError):
                if attempt==9:raise
        raise RuntimeError('incomplete download')
    with ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(piece,range(8)))
    temp=dest.with_suffix('.assembling');digest=hashlib.sha256();total=0
    with temp.open('wb') as output:
        for part in parts:
            with part.open('rb') as source:
                while data:=source.read(1024*1024):output.write(data);digest.update(data);total+=len(data)
    assert total==SIZE and digest.hexdigest()==SHA
    os.replace(temp,dest)
    print(f'official index SHA256 verified: {SHA}',flush=True)


if __name__=='__main__':main()
