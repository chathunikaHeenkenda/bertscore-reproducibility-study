"""Load 200 XSum test examples, describe them, and score a lead-1 baseline."""
import os, time
import pandas as pd
from datasets import load_dataset
import re
from common import sentence_bleu, rouge_l, bertscore, ensure_nltk, MODEL_TYPE

N = 200
try:
    ensure_nltk()
    from nltk.tokenize import sent_tokenize
    sent_tokenize("Test. Sentence.")
except Exception as e:  # e.g. no internet for the NLTK download
    print(f"NLTK punkt unavailable ({type(e).__name__}); using a simple regex sentence splitter")
    def sent_tokenize(text):
        return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
t0 = time.time()
try:
    ds = load_dataset("EdinburghNLP/xsum", split=f"test[:{N}]")
    name, doc_col, sum_col, lead_k = "xsum", "document", "summary", 1
except Exception as e:
    print(f"XSum failed ({e}); falling back to cnn_dailymail 3.0.0")
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{N}]")
    name, doc_col, sum_col, lead_k = "cnn_dailymail", "article", "highlights", 3

df = ds.to_pandas()
df = df.rename(columns={doc_col: "document", sum_col: "summary"})
os.makedirs("data/new_existing", exist_ok=True)
df.to_csv(f"data/new_existing/{name}_{N}.csv", index=False)

df["doc_words"] = df["document"].str.split().str.len()
df["sum_words"] = df["summary"].str.split().str.len()
print(f"Dataset: {name} (test split, first {len(df)} examples)")
print(f"Avg document length: {df.doc_words.mean():.1f} words (median {df.doc_words.median():.0f})")
print(f"Avg summary length:  {df.sum_words.mean():.1f} words (median {df.sum_words.median():.0f})")
print(f"Compression ratio:   {(df.doc_words / df.sum_words).mean():.1f}x")

df["lead"] = [" ".join(sent_tokenize(d)[:lead_k]) for d in df["document"]]
cands, refs = df["lead"].tolist(), df["summary"].tolist()
P, R, F = bertscore(cands, refs)
df["bertscore_p"], df["bertscore_r"], df["bertscore_f1"] = P, R, F
df["bleu"] = sentence_bleu(cands, refs)
df["rougeL"] = rouge_l(cands, refs)
os.makedirs("results", exist_ok=True)
df[["id", "summary", "lead", "doc_words", "sum_words", "bertscore_p", "bertscore_r",
    "bertscore_f1", "bleu", "rougeL"]].to_csv("results/new_dataset_baseline.csv", index=False)
print(f"\nLead-{lead_k} baseline vs reference summary (mean over {len(df)}), model {MODEL_TYPE}:")
print(df[["bertscore_p", "bertscore_r", "bertscore_f1", "bleu", "rougeL"]].mean().round(4).to_string())
print(f"\nRuntime: {time.time()-t0:.1f}s")
