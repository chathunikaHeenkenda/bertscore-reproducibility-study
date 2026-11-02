"""Score the constructed perturbation dataset with BERTScore, BLEU and ROUGE-L."""
import os, time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import sentence_bleu, rouge_l, bertscore, MODEL_TYPE

t0 = time.time()
df = pd.read_csv("data/constructed/perturbations.csv")
cands, refs = df["candidate"].tolist(), df["reference"].tolist()
df["bertscore_p"], df["bertscore_r"], df["bertscore_f1"] = bertscore(cands, refs)
df["bleu"] = sentence_bleu(cands, refs)
df["rougeL"] = rouge_l(cands, refs)
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
df.to_csv("results/perturbation_scores.csv", index=False)

order = ["paraphrase", "lexical_variation", "negation", "numerical_change",
         "entity_substitution", "irrelevant_addition", "hallucination"]
summary = (df.groupby("type")[["bertscore_p", "bertscore_r", "bertscore_f1", "bleu", "rougeL"]]
             .mean().reindex(order).round(4))
summary.insert(0, "expected_quality", ["high", "high"] + ["low"] * 5)
summary.to_csv("results/perturbation_summary.csv")
print(f"Mean scores per perturbation type (BERTScore model: {MODEL_TYPE}):")
print(summary.to_string())

# Box plot of BERTScore F1 by type
fig, ax = plt.subplots(figsize=(10, 5))
data = [df.loc[df.type == t, "bertscore_f1"] for t in order]
bp = ax.boxplot(data, patch_artist=True)
for patch, t in zip(bp["boxes"], order):
    patch.set_facecolor("#7fc97f" if t in ("paraphrase", "lexical_variation") else "#f0a07a")
ax.set_xticks(range(1, len(order) + 1))
ax.set_xticklabels([t.replace("_", "\n") for t in order])
ax.set_ylabel("BERTScore F1")
ax.set_title(f"BERTScore F1 by perturbation type ({MODEL_TYPE}, n=40 per type)\n"
             "green = meaning preserved, orange = meaning changed")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig("figures/bertscore_by_type.png", dpi=150)

# Pairwise ranking accuracy: for each reference, is the paraphrase scored strictly
# higher than the low-quality candidate of each type? (ties count as wrong)
metrics = {"BERTScore_F1": "bertscore_f1", "BLEU": "bleu", "ROUGE-L": "rougeL"}
piv = {m: df.pivot(index="ref_id", columns="type", values=c) for m, c in metrics.items()}
rows = []
for anchor in ["paraphrase", "lexical_variation"]:
    for t in order[2:]:
        row = {"high_quality_anchor": anchor, "low_quality_type": t}
        for m, p in piv.items():
            row[m] = round((p[anchor] > p[t]).mean(), 3)
        rows.append(row)
    sub = [r for r in rows if r["high_quality_anchor"] == anchor]
    overall = {"high_quality_anchor": anchor, "low_quality_type": "ALL"}
    for m in metrics:
        overall[m] = round(sum(r[m] for r in sub) / len(sub), 3)
    rows.append(overall)
acc = pd.DataFrame(rows)
acc.to_csv("results/pairwise_accuracy.csv", index=False)
print("\nPairwise accuracy: share of the 40 references where the high-quality candidate")
print("(paraphrase or lexical_variation) is scored strictly above the low-quality one:")
print(acc.to_string(index=False))
print(f"\nSaved results/perturbation_scores.csv, results/perturbation_summary.csv, "
      f"results/pairwise_accuracy.csv, figures/bertscore_by_type.png\nRuntime: {time.time()-t0:.1f}s")
