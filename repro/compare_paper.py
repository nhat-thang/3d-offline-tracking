import json, sys
m = json.load(open(sys.argv[1]))
paper = {'amota': 70.2, 'mota': 60.1, 'ids': 385}   # ImmortalTracker v1, Table 5 (nuScenes val)
print('| Metric | Paper | Chay lai | Lech |\n|---|---|---|---|')
for k, p in paper.items():
    v = m[k] * 100 if k != 'ids' else m[k]
    print(f'| {k.upper()} | {p} | {v:.1f} | {v - p:+.1f} |')
print('\nThem (paper khong bao): ' + ', '.join(f'{k.upper()} {m[k]:.0f}' if k not in ('amotp','recall') else f'{k.upper()} {m[k]:.3f}' for k in ['amotp','recall','fp','fn','frag']))
