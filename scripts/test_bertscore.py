"""Sanity test: BERTScore on two candidate/reference pairs."""
import time
from bert_score import score
from common import MODEL_TYPE

cands = ["The cat sat on the mat.", "The stock market crashed on Monday."]
refs = ["A cat was sitting on the mat.", "Shares rose sharply at the start of the week."]

t = time.time()
P, R, F = score(cands, refs, model_type=MODEL_TYPE, lang="en", verbose=False)
print(f"Model: {MODEL_TYPE}")
for c, r, p, rr, f in zip(cands, refs, P, R, F):
    print(f"\nCandidate: {c}\nReference: {r}\nP={p:.4f}  R={rr:.4f}  F1={f:.4f}")
print(f"\nTime: {time.time()-t:.1f}s")
