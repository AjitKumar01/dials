"""LaTeX for the main results table and its test table (paper/tables/main.tex, paper/tables/tests.tex).

Simple baselines and TRIAD-vs-best-simple tests: bundlefuse/results/fair_prior/comparison.json
(experiments.fair_prior.compare; simple baselines with the frequency prior, weight chosen on validation).
CLHE and RaMen: the reruns and published numbers listed in RESULTS.md sections 3 and 4 (clean test queries; RaMen on
POG and Spotify and CLHE on full Spotify are for all test queries). Their tests against TRIAD are those of RESULTS.md
section 3 (Holm over 20 planned tests), which these runs do not change.

python paper/figdata/tables.py
"""
import json
from pathlib import Path

R = Path(__file__).resolve().parents[3]/'bundlefuse'/'results'/'fair_prior'/'comparison.json'
OUT = Path(__file__).resolve().parents[1]/'tables'
ECOM = ['electronic', 'food', 'clothing'];OTHER = ['pog_dense', 'pog', 'spotify_sparse', 'spotify']
NAME = dict(electronic='Electronic', food='Food', clothing='Clothing', pog_dense='POG\\_DENSE', pog='POG',
            spotify_sparse='Spotify\\_sparse', spotify='Spotify')
SIMPLE = [('itemknn', 'ItemKNN'), ('rp3beta', 'RP3$\\beta$'), ('ease', 'EASE'), ('hybrid', 'Content+ItemKNN'),
          ('label_knn_content', 'Label+ItemKNN+content')]
# (R@20, N@20, mark): rerun on identical splits unless marked; d = published, same split files, all test queries;
# s = one of three seeds finished, all test queries.
CLHE = dict(electronic=(.362, .259, ''), food=(.419, .271, ''), clothing=(.325, .196, ''), pog_dense=(.3486, .2476, ''),
            pog=(.0259, .0172, ''), spotify_sparse=(.0971, .1153, ''), spotify=(.108, .1788, 's'))
RAMEN = dict(electronic=(.345, .256, ''), food=(.345, .265, ''), clothing=(.240, .190, ''), pog=(.0375, .0226, 'd'),
             spotify=(.1091, .1882, 'd'))
NEURAL_TESTS = dict(  # TRIAD minus CLHE / RaMen, NDCG@20 [95% CI], RESULTS.md section 3; all reject after Holm
    clhe=dict(electronic='+.076 [.049, .104]', food='+.203 [.173, .233]', clothing='+.242 [.213, .271]',
              pog_dense='+.031 [.028, .034]', pog='+.013 [.011, .016]', spotify_sparse='+.141 [.131, .151]'),
    ramen=dict(electronic='+.078 [.053, .104]', food='+.209 [.172, .244]', clothing='+.249 [.214, .282]'))


def f3(x):return f'{x:.3f}'.lstrip('0') if x >= 0 else '$-$' + f'{-x:.3f}'.lstrip('0')
def sgn(x):return ('+' if x >= 0 else '$-$') + f'{abs(x):.3f}'.lstrip('0')


def panel(c, names):
    rows = []  # (label, {name: (R, N, mark)})
    for key, label in SIMPLE:
        rows.append((label, {n: (*c['datasets'][n]['methods'][key + '+prior']['test'], '') for n in names if key + '+prior' in c['datasets'][n]['methods']}))
    rows.append(('CLHE', {n: CLHE[n] for n in names if n in CLHE}))
    rows.append(('RaMen', {n: RAMEN[n] for n in names if n in RAMEN}))
    triad = ('TRIAD', {n: (*c['datasets'][n]['triad'], '') for n in names})
    # best and second best per column among the baselines and TRIAD
    rank = {}
    for n in names:
        for j in (0, 1):
            vals = sorted({r[1][n][j] for r in rows + [triad] if n in r[1]}, reverse=True);rank[n, j] = vals[:2]
    def cell(v, n, j, mark):
        s = f3(v[j]);s = ('\\textbf{' + s + '}') if v[j] == rank[n, j][0] else ('\\underline{' + s + '}') if len(rank[n, j]) > 1 and v[j] == rank[n, j][1] else s
        return s + ({'d': '$^\\dagger$', 's': '$^\\ddagger$'}.get(mark, '') if j == 1 else '')
    lines = []
    for i, (label, d) in enumerate(rows + [triad]):
        if i == len(SIMPLE):lines.append('\\midrule')
        if i == len(rows):lines.append('\\midrule')
        cells = []
        for n in names:
            cells += [cell(d[n], n, 0, d[n][2]), cell(d[n], n, 1, d[n][2])] if n in d else ['--', '--']
        lines.append(label + ' & ' + ' & '.join(cells) + '\\\\')
    nb = ['\\emph{TRIAD w/o unseen-item bonus}'];imp = ['Improv.\\ over best baseline']
    for n in names:
        nb += [f3(v) for v in c['datasets'][n]['triad_no_bonus']]
        for j in (0, 1):
            best = max(r[1][n][j] for r in rows if n in r[1]);imp.append(f'{100*(c["datasets"][n]["triad"][j]/best - 1):+.1f}\\%'.replace('-', '$-$'))
    lines += [' & '.join(nb) + '\\\\', '\\midrule', ' & '.join(imp) + '\\\\']
    head = ' & '.join(f'\\multicolumn{{2}}{{c}}{{{NAME[n]}}}' for n in names)
    rules = ''.join(f'\\cmidrule(lr){{{2 + 2*k}-{3 + 2*k}}}' for k in range(len(names)))
    sub = ' & '.join(['R@20 & N@20']*len(names))
    return '\n'.join([f'\\begin{{tabular}}{{l{"cc"*len(names)}}}', '\\toprule', f'Method & {head}\\\\', rules, f' & {sub}\\\\',
                      '\\midrule', *lines, '\\bottomrule', '\\end{tabular}'])


def tests(c):
    t = {(x['dataset'], x['metric']): x for x in c['tests']['triad_vs_best_simple_prior']}
    u = {(x['dataset'], x['metric']): x for x in c['tests']['triad_no_bonus_vs_best_simple_prior']}
    label = dict(itemknn='ItemKNN', rp3beta='RP3$\\beta$', ease='EASE', hybrid='content+kNN', label_knn_content='lab.+kNN+cont.',
                 label_knn='lab.+kNN', samelabel='same-label', transition='transition', cooccurrence='co-occurrence', popularity='popularity')
    def ci(x):return f'{sgn(x["delta"])} [{sgn(x["ci95"][0])}, {sgn(x["ci95"][1])}]'
    def p(x):return ('\\textbf{' if x['p_holm'] < .05 else '') + f'{x["p_holm"]:.3f}'.lstrip('0') + ('}' if x['p_holm'] < .05 else '')
    lines = []
    for n in ECOM + OTHER:
        b = c['datasets'][n]['best_simple'];prior = b.endswith('+prior');b = b.replace('+prior', '')
        lines.append(' & '.join([NAME[n], label.get(b, b) + ('$^{*}$' if prior else ''), ci(t[n, 'ndcg20']), p(t[n, 'ndcg20']), p(t[n, 'recall20']),
                                 ci(u[n, 'ndcg20']), p(u[n, 'ndcg20'])]) + '\\\\')
    return '\n'.join(['\\begin{tabular}{lllrrlr}', '\\toprule',
                      '& & \\multicolumn{3}{c}{TRIAD $-$ best simple} & \\multicolumn{2}{c}{TRIAD w/o bonus $-$ best simple}\\\\',
                      '\\cmidrule(lr){3-5}\\cmidrule(lr){6-7}',
                      'Dataset & Best simple & $\\Delta$ N@20 [95\\% CI] & $p$ (N) & $p$ (R) & $\\Delta$ N@20 [95\\% CI] & $p$ (N)\\\\',
                      '\\midrule', *lines, '\\bottomrule', '\\end{tabular}'])


def main():
    c = json.loads(R.read_text());OUT.mkdir(exist_ok=True)
    (OUT/'main_ecom.tex').write_text(panel(c, ECOM) + '\n');(OUT/'main_other.tex').write_text(panel(c, OTHER) + '\n')
    (OUT/'tests.tex').write_text(tests(c) + '\n');print('tables ok')


if __name__ == '__main__':
    main()
