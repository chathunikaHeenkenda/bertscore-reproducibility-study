"""Dataset snippet figures: a few real rows from WMT17 de-en (original data) and from the
constructed perturbation set. Writes figures/snippet_wmt17_deen.png and figures/snippet_perturbations.png."""
import textwrap, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

def table_png(df, widths, out, title):
    """Render rows as a simple wrapped-text table (line-based layout, no overlap)."""
    rows = [[textwrap.fill(str(v), w) for v, w in zip(r, widths)] for r in df.values.tolist()]
    head = [textwrap.fill(c, w) for c, w in zip(df.columns, widths)]
    allrows = [head] + rows
    hts = [max(c.count("\n") + 1 for c in r) + 0.4 for r in allrows]
    total = sum(hts); xs = [0]
    for w in widths[:-1]: xs.append(xs[-1] + w + 2)
    width = xs[-1] + widths[-1]
    fig = plt.figure(figsize=(width * 0.075, total * 0.17 + 0.5)); ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-1, width + 1); ax.set_ylim(total + 1.6, 0); ax.axis("off")
    ax.text(width / 2, 0.6, title, ha="center", va="center", fontsize=8.5, weight="bold")
    y = 1.4
    for i, (r, h) in enumerate(zip(allrows, hts)):
        if i == 0: ax.add_patch(plt.Rectangle((-0.5, y), width + 1, h, color="#dde6f0", zorder=0))
        for x, c in zip(xs, r):
            ax.text(x, y + 0.3, c, va="top", ha="left", fontsize=7.5, family="DejaVu Sans",
                    weight="bold" if i == 0 else "normal")
        y += h; ax.plot([-0.5, width + 0.5], [y, y], color="#999999", lw=0.5)
    plt.savefig(out, dpi=220, bbox_inches="tight"); plt.close()

w = pd.read_csv("results/replication_wmt_segments_min3ann.csv").head(4)
w = pd.DataFrame({"Source (de)": w.src, "MT output (en)": w.mt, "Reference (en)": w.ref,
                  "Human DA z": w.score.round(3), "#ann": w.annotators})
table_png(w, [38, 34, 34, 8, 5], "figures/snippet_wmt17_deen.png",
          "WMT17 de-en segments (RicardoRei/wmt-da-human-evaluation, lp=de-en, year=2017, >=3 annotators)")

p = pd.read_csv("results/perturbation_scores.csv")
p = p[p.ref_id == "r01"]
p = pd.DataFrame({"Type": p.type, "Label": p.expected_quality.map({"high": "preserved", "low": "changed"}),
                  "Candidate (reference r01 shown in caption)": p.candidate, "BERTScore F1": p.bertscore_f1.round(3)})
table_png(p, [20, 9, 70, 10], "figures/snippet_perturbations.png",
          "Constructed perturbation set, reference r01: " + pd.read_csv("results/perturbation_scores.csv").reference[0])
