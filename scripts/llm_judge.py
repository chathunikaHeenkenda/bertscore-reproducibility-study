"""RQ4: LLM-as-a-judge with a 1-5 rubric.

Runs a small open instruct model locally on CPU (default Qwen/Qwen2.5-1.5B-Instruct).
For each (reference, candidate) pair the model is asked for a 1-5 score; we read the
next-token probabilities over the tokens "1".."5" (single forward pass, deterministic):
  judge_score    = argmax digit
  judge_expected = sum_k k * p(k)   (probability-weighted, used for correlations)

Usage:
  python scripts/llm_judge.py --task perturbations   # data/constructed/perturbations.csv or results/perturbation_scores.csv
  python scripts/llm_judge.py --task wmt --n 500      # results/replication_wmt_segments_min3ann.csv
"""
import argparse, os, time
import pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

RUBRIC = """You are an expert evaluator. Compare the CANDIDATE sentence with the REFERENCE sentence and rate how well the candidate preserves the meaning of the reference.
5 = same meaning, all facts (numbers, names, polarity) preserved; wording may differ
4 = same meaning with a minor, unimportant difference
3 = partly the same meaning, some important information missing, added or changed
2 = mostly different meaning or an important fact is wrong (e.g. wrong number, wrong name, negated)
1 = completely different or contradictory meaning
Answer with a single digit from 1 to 5."""

def build(tok, ref, cand):
    msgs = [{"role": "system", "content": RUBRIC},
            {"role": "user", "content": f"REFERENCE: {ref}\nCANDIDATE: {cand}\nScore (1-5):"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["perturbations", "wmt"], required=True)
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(os.cpu_count())
    if a.task == "perturbations":
        df = pd.read_csv("results/perturbation_scores.csv")
        refcol, candcol = "reference", "candidate"
    else:
        df = pd.read_csv("results/replication_wmt_segments_min3ann.csv")
        df = df.sample(n=min(a.n, len(df)), random_state=a.seed).reset_index(drop=True)
        refcol, candcol = "ref", "mt"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    digit_ids = [tok.encode(str(k), add_special_tokens=False)[0] for k in range(1, 6)]
    out_s, out_e = [], []
    t0 = time.time()
    for i, r in df.iterrows():
        ids = tok(build(tok, r[refcol], r[candcol]), return_tensors="pt")
        with torch.no_grad():
            logits = model(**ids).logits[0, -1].float()
        p = torch.softmax(logits[digit_ids], dim=0)
        out_s.append(int(p.argmax()) + 1)
        out_e.append(float((p * torch.arange(1, 6)).sum()))
        if i % 25 == 0: print(i, len(df), f"{time.time()-t0:.0f}s", flush=True)
    df["judge_score"], df["judge_expected"] = out_s, out_e
    df["judge_model"] = a.model
    out = f"results/llm_judge_{a.task}.csv"
    df.to_csv(out, index=False); print("wrote", out, f"{time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
