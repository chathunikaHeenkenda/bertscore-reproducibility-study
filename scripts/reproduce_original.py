"""Replication on original-style data: WMT segment-level human judgements (DA).

Data: Hugging Face dataset RicardoRei/wmt-da-human-evaluation (WMT17-22 Direct
Assessment scores). We take one to-English pair and year, sample N segments with
a fixed seed, and correlate metric scores with the human z-score ('score').
"""
import argparse, os, time
import pandas as pd
from datasets import load_dataset
from scipy.stats import pearsonr, kendalltau
from common import sentence_bleu, rouge_l, bertscore, MODEL_TYPE

ap = argparse.ArgumentParser()
ap.add_argument("--lp", default="de-en")
ap.add_argument("--year", type=int, default=2017)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--min_annotators", type=int, default=1,
                help="keep only segments judged by at least this many annotators (less noisy)")
args = ap.parse_args()

t0 = time.time()
ds = load_dataset("RicardoRei/wmt-da-human-evaluation", split="train")
ds = ds.filter(lambda x: x["lp"] == args.lp and x["year"] == args.year)
df = ds.to_pandas().dropna(subset=["mt", "ref", "score"])
df = df[df["annotators"] >= args.min_annotators]
print(f"{args.lp} {args.year} (annotators >= {args.min_annotators}): {len(df)} segments available")
df = df.sample(n=min(args.n, len(df)), random_state=args.seed).reset_index(drop=True)
os.makedirs("data/original", exist_ok=True)
df.to_csv(f"data/original/wmt{args.year}_{args.lp}_sample{len(df)}.csv", index=False)

cands, refs = df["mt"].tolist(), df["ref"].tolist()
_, _, df["bertscore_f1"] = bertscore(cands, refs, verbose=True)
df["bleu"] = sentence_bleu(cands, refs)
df["rougeL"] = rouge_l(cands, refs)

# Paper values: Zhang et al. (2020), Appendix Table 13, WMT17 to-English segment-level,
# absolute Pearson, official WMT17 DAseg set (560 segments/pair, >=15 judgements each).
PAPER = {("de-en", 2017): {"bertscore_f1": 0.745, "bleu": 0.432}}
rows = []
for m in ["bertscore_f1", "bleu", "rougeL"]:
    r, _ = pearsonr(df[m], df["score"])
    tau, _ = kendalltau(df[m], df["score"])
    rows.append({"metric": m, "pearson_r": round(r, 4), "kendall_tau": round(tau, 4),
                 "lp": args.lp, "year": args.year, "n": len(df),
                 "min_annotators": args.min_annotators,
                 "model": MODEL_TYPE if m == "bertscore_f1" else "-",
                 "paper_pearson": PAPER.get((args.lp, args.year), {}).get(m, "n/a")})
res = pd.DataFrame(rows)
os.makedirs("results", exist_ok=True)
tag = "" if args.min_annotators == 1 else f"_min{args.min_annotators}ann"
res.to_csv(f"results/replication_wmt{tag}.csv", index=False)
df.to_csv(f"results/replication_wmt_segments{tag}.csv", index=False)
print("\nCorrelation with human DA scores (segment level):")
print(res.to_string(index=False))
print("(paper_pearson: Zhang et al. 2020 Table 13; BLEU there is SentBLEU; ROUGE-L not reported)")
print(f"\nRuntime: {time.time()-t0:.1f}s")
