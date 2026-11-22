"""The MIMO Shock study's design as code: which problems, which laws, which runs. Writes the problem list and the
jobs file the batch reads; the design and its reasons are in the record's Design.md.

    python shock/plan.py --design study --out <folder>        # writes structures.txt and jobs.json
"""
import argparse
import json
import os

RANDOM_STUDY_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 33, 12, 13, 14, 15, 34, 17, 35, 19]   # the Random study's twenty, in order
AUDIT_SEEDS = [70, 71, 72, 73, 74]     # held back, drawn fresh (not in any Random batch); replacements from 80 up, in order
LAWS = ['inverse', 'shape', 'paired', 'followed', 'followed-hold']   # srs-design did not pass the screen (Design.md)
DESIGNS = {
    # cells: (rig, tier) -> structures (the first n of the Random study's seeds); every cell runs every target kind
    # and every law, except paired, which runs on the first `paired_n` structures of each cell (the screen: it fails
    # at the first level on every problem under every reading, so more of it distinguishes nothing)
    'study': dict(cells={('6x6', 'realistic'): 20, ('6x6', 'strong'): 20, ('12x6', 'realistic'): 20, ('12x6', 'strong'): 20,
                         ('6x6', None): 5, ('12x6', None): 5},
                  kinds=['srs', 'half_sine', 'sawtooth'], laws=LAWS, paired_n=5, noise=1e-3, seed=11),
}


def sid(rig, tier, seed):
    return f'frame-{rig}' + (f'-{tier}' if tier else '') + f'_s{seed:02d}'


def build(name):
    d = DESIGNS[name]
    probs, jobs = [], []
    for (rig, tier), n in d['cells'].items():
        for i, s in enumerate(RANDOM_STUDY_SEEDS[:n]):
            p = sid(rig, tier, s)
            probs.append(p)
            for k in d['kinds']:
                for law in d['laws']:
                    if law == 'paired' and i >= d['paired_n']:
                        continue
                    jobs.append(dict(problem=p, kind=k, law=law, noise=d['noise'], seed=d['seed']))
    return probs, jobs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--design', default='study'); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    probs, jobs = build(a.design)
    open(os.path.join(a.out, 'structures.txt'), 'w').write(','.join(probs) + '\n')
    json.dump(jobs, open(os.path.join(a.out, 'jobs.json'), 'w'), indent=0)
    print(len(probs), 'structures,', len(jobs), 'runs')


if __name__ == '__main__':
    main()
