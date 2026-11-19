"""What the frozen models learn (ARCHITECTURE.md §14): theta, alpha, beta, the gate weights averaged over the first 512
validation queries, and delta_l for one of three seeds sharing the candidate's label at the median label size (median
over items of the size of their label), each averaged over seeds 0, 1, 2. Writes paper/figdata/learned.json.

PYTHONPATH=. python paper/figdata/learned.py [datasets ...]
"""
import triad  # pins BLAS threads before numpy loads
import json, math, sys
from pathlib import Path
import numpy as np
import torch
from triad import data
from triad import train as T

ROOT = Path(__file__).resolve().parents[2];OUT = Path(__file__).parent/'learned.json'


@torch.no_grad()
def main(names):
    torch.set_num_threads(4);report = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name in names:
        runs = [ROOT/'runs'/name/f'seed{s}' for s in (0, 1, 2)]
        if not all((r/'model.pt').exists() for r in runs):print(name, 'not all seeds trained');continue
        ds = data.load(name, ROOT/'cache');rows = []
        for run in runs:
            m = T.load(ds, run, ROOT/'cache');a, b = [float(v) for v in m.exponents]
            seeds = [torch.as_tensor(sorted(set(s)), dtype=torch.long) for s, _ in ds.val_pairs[:512]]
            w = torch.cat([m.gate_weights(seeds[i:i + 64], None) for i in range(0, len(seeds), 64)]).mean(0).tolist()
            deltas = []
            for l in range(m.levels):
                sizes = getattr(m, f'label_size{l}')[getattr(m, f'label{l}')]  # log(1 + n_g(c)) per item
                deltas.append(float(m.same_label[l](torch.tensor([[math.log(2), math.log(3), float(sizes.median())]]))))
            rows.append(dict(theta=float(m.theta), alpha=a, beta=b, gate=w, delta=deltas))
        report[name] = {k: np.mean([r[k] for r in rows], 0).tolist() for k in rows[0]}
        print(name, {k: np.round(v, 2).tolist() for k, v in report[name].items()}, flush=True)
    OUT.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:] or ['electronic', 'food', 'clothing', 'pog_dense', 'pog', 'spotify_sparse', 'spotify'])
