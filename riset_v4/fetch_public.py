"""Unduh tujuh laga IDSSE publik dan konversi menjadi cache NPZ."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
from pathlib import Path
import shutil
import urllib.request
from .data import DEV,RETEST
from night_tracking import save_external_cache


def download(match,folder,entries):
    target=folder/'cache'/f'{match}.npz'
    if target.exists():return str(target)
    names=[e['path'] for e in entries if e['path'].endswith(f'DFL-MAT-{match}.xml')
           and e['path'].startswith(('DFL_02_01_matchinformation','DFL_04_03_positions_raw_observed'))]
    if len(names)!=2:raise ValueError(f'Expected metadata and positions for {match}: {names}')
    for name in names:
        path=folder/name
        if not path.exists():
            print(f'download {name}',flush=True)
            request=urllib.request.Request('https://huggingface.co/datasets/pysport/idsse-data/resolve/main/'+name)
            with urllib.request.urlopen(request,timeout=120) as src,path.with_suffix('.part').open('wb') as dst:
                shutil.copyfileobj(src,dst,1024*1024)
            path.with_suffix('.part').replace(path)
    print(f'parse {match}',flush=True)
    save_external_cache(target,match,folder)
    print(f'ready {match}',flush=True)
    return str(target)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    (args.output/'cache').mkdir(parents=True,exist_ok=True)
    with urllib.request.urlopen('https://huggingface.co/api/datasets/pysport/idsse-data/tree/main',timeout=60) as r:
        entries=json.load(r)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(download,m,args.output,entries) for m in DEV+RETEST]
        for task in as_completed(pending):print(task.result(),flush=True)


if __name__=='__main__':main()
