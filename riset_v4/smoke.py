"""Uji integrasi seluruh CLI memakai data sintetis, bukan skor eksperimen."""
import argparse
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .data import DEV,RETEST
from .run import benchmark
from .test_v4 import fixture


def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);cache=root/'cache';cache.mkdir()
        for match in DEV+RETEST:
            arrays={}
            for i in range(2):
                d,y=fixture(1200)
                arrays.update({f'ball{i}':d.ball,f'home{i}':d.players[0],f'away{i}':d.players[1],f'truth{i}':y})
            np.savez(cache/f'{match}.npz',pitch=d.pitch,**arrays)
        args=SimpleNamespace(output=root/'out',cache=[cache],folds=[0],seed=331,
            train_scenarios=['original'],iterations=2,epochs=1,steps=1,device='cpu')
        benchmark(args)
        assert (root/'out/result.json').exists()
        # Exercise exact-config resume and the stale-config guard.
        benchmark(args)
        args.seed+=1
        try:benchmark(args)
        except ValueError:pass
        else:raise AssertionError('Stale output was accepted')
        print('INTEGRATION_OK_SYNTHETIC_ONLY')


if __name__=='__main__':main()
