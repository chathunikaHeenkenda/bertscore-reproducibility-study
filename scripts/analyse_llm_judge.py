"""Correlate LLM-judge scores with labels/humans next to BERTScore, BLEU, ROUGE-L.
Reads results/llm_judge_perturbations.csv and results/llm_judge_wmt.csv,
writes results/llm_judge_correlations.csv, results/llm_judge_by_type.csv,
results/llm_judge_pairwise_accuracy.csv and figures/llm_judge_by_type.png."""
import pandas as pd, numpy as np
from scipy.stats import pearsonr, kendalltau, spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows = []
p = pd.read_csv("results/llm_judge_perturbations.csv")
p["label"] = (p["expected_quality"] == "high").astype(int)   # 1 = meaning preserved
mets = {"BERTScore_F1": "bertscore_f1", "BLEU": "bleu", "ROUGE-L": "rougeL", "LLM_judge": "judge_expected"}
for name, c in mets.items():
    rows.append(dict(dataset="perturbations", target="meaning_preserved_label(0/1)", metric=name, n=len(p),
                     pearson_r=pearsonr(p[c], p.label)[0], kendall_tau=kendalltau(p[c], p.label)[0],
                     spearman_rho=spearmanr(p[c], p.label)[0]))
w = pd.read_csv("results/llm_judge_wmt.csv")
for name, c in mets.items():
    cc = {"BERTScore_F1": "bertscore_f1"}.get(name, c)
    rows.append(dict(dataset="wmt17_de-en_min3ann", target="human_DA_z", metric=name, n=len(w),
                     pearson_r=pearsonr(w[cc], w.score)[0], kendall_tau=kendalltau(w[cc], w.score)[0],
                     spearman_rho=spearmanr(w[cc], w.score)[0]))
pd.DataFrame(rows).round(4).to_csv("results/llm_judge_correlations.csv", index=False)

bt = p.groupby(["type", "expected_quality"], sort=False).agg(
    judge_mean=("judge_expected", "mean"), judge_argmax_mean=("judge_score", "mean"),
    bertscore_f1=("bertscore_f1", "mean")).round(4).reset_index()
bt.to_csv("results/llm_judge_by_type.csv", index=False)

# pairwise: same reference, high-quality anchor vs each low-quality edit; ties count 0.5
acc = []
for anchor in ["paraphrase", "lexical_variation"]:
    for name, c in mets.items():
        wins = []
        for rid, g in p.groupby("ref_id"):
            a = g[g.type == anchor][c].values[0]
            for v in g[g.expected_quality == "low"][c].values:
                wins.append(1.0 if a > v else 0.5 if a == v else 0.0)
        acc.append(dict(anchor=anchor, metric=name, n_pairs=len(wins), accuracy=round(np.mean(wins), 4)))
pd.DataFrame(acc).to_csv("results/llm_judge_pairwise_accuracy.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 3.2))
cols = ["tab:green" if q == "high" else "tab:red" for q in bt.expected_quality]
ax.bar(bt.type, bt.judge_mean, color=cols)
ax.set_ylim(1, 5); ax.set_ylabel("Mean LLM-judge score (1-5)")
ax.set_title("LLM judge (Qwen2.5-1.5B-Instruct) by edit type; green = meaning preserved")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.savefig("figures/llm_judge_by_type.png", dpi=200)
print(pd.DataFrame(rows).round(4).to_string()); print(bt.to_string()); print(pd.DataFrame(acc).to_string())

# scatter: WMT human DA vs BERTScore F1 and vs LLM judge
fig, axs = plt.subplots(1, 2, figsize=(7, 3))
axs[0].scatter(w.bertscore_f1, w.score, s=6, alpha=.5); axs[0].set_xlabel("BERTScore F1"); axs[0].set_ylabel("Human DA z")
axs[0].set_title(f"r = {pearsonr(w.bertscore_f1, w.score)[0]:.3f}")
axs[1].scatter(w.judge_expected, w.score, s=6, alpha=.5, color="tab:orange"); axs[1].set_xlabel("LLM judge (1-5)")
axs[1].set_title(f"r = {pearsonr(w.judge_expected, w.score)[0]:.3f}")
plt.tight_layout(); plt.savefig("figures/wmt_scatter_bertscore_vs_judge.png", dpi=200)
