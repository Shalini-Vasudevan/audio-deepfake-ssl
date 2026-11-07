"""Aggregate results/*.json -> paper/results_tables.tex + paper/results_macros.tex"""
import glob, json, sys, collections, numpy as np
res_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
out = sys.argv[2] if len(sys.argv) > 2 else "../paper"
R = collections.defaultdict(list)
for f in sorted(glob.glob(f"{res_dir}/*.json")):
    r = json.load(open(f)); R[r["name"]].append(r)

NAMES = [("lfcc_lcnn", "LFCC-LCNN (no SSL)"), ("wavlm_last", "WavLM, last layer"),
         ("wavlm_uniform_aug", "WavLM + uniform layer avg + aug"), ("wavlm_wsum_noaug", "WavLM + LAF (no aug)"),
         ("wavlm_wsum_aug", "WavLM + LAF + aug"), ("cola_adv01", "CoLA, $\\lambda_{adv}$=0.1"),
         ("cola_wavlm", "CoLA, $\\lambda_{adv}$=0.3"), ("cola_adv10", "CoLA, $\\lambda_{adv}$=1.0")]
def ms(v, d=2):
    v = np.array(v, float)
    if len(v) == 0 or np.isnan(v).all(): return "--"
    return f"{v.mean():.{d}f}" + (f"$\\pm${v.std(ddof=1):.{d}f}" if len(v) > 1 else "")
def col(rs, k): return [r[k] for r in rs if k in r]

rows = []
for key, lab in NAMES:
    rs = R.get(key, [])
    la, itw = col(rs, "la_eval_eer"), col(rs, "itw_eer")
    gap = [i - l for i, l in zip(itw, la)]
    rows.append(f"{lab} & {ms(la)} & {ms(col(rs,'la_eval_mintdcf'),3)} & {ms(itw)} & {ms(gap,1)} & {len(rs)} \\\\")
tab = ("\\begin{table}[t]\\centering\\caption{Train on ASVspoof 2019 LA; EER (\\%) on LA eval (in-domain) and In-the-Wild "
       "(cross-dataset). Mean$\\pm$std over seeds. Gap = ITW EER $-$ LA EER (points). min t-DCF is defined only for LA.}\\label{tab:main}\n"
       "\\resizebox{\\columnwidth}{!}{\\begin{tabular}{lcccc c}\\toprule Model & LA EER & LA min t-DCF & ITW EER & Gap (pts) & Runs\\\\\\midrule\n" + "\n".join(rows) +
       "\n\\bottomrule\\end{tabular}}\\end{table}\n")
open(f"{out}/results_tables.tex", "w").write(tab)

def best(key, k):
    rs = R.get(key, []); v = col(rs, k)
    return f"{np.mean(v):.2f}" if v else "XX.XX"
M = {"resBaseITW": best("wavlm_last", "itw_eer"), "resBaseLA": best("wavlm_last", "la_eval_eer"),
     "resOursITW": best("cola_wavlm", "itw_eer"), "resOursLA": best("cola_wavlm", "la_eval_eer"),
     "resOursTDCF": (lambda v: f"{np.mean(v):.3f}" if v else "X.XXX")(col(R.get("cola_wavlm", []), "la_eval_mintdcf")),
     "resLfccITW": best("lfcc_lcnn", "itw_eer")}
try:
    rel = 100 * (1 - float(M["resOursITW"]) / float(M["resBaseITW"]))
    M["resRel"] = f"{rel:.0f}"
except ValueError:
    M["resRel"] = "XX"
open(f"{out}/results_macros.tex", "w").write("".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in M.items()))
print(open(f"{out}/results_macros.tex").read()); print(tab)
