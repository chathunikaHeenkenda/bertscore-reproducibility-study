"""Shared settings and metric helpers for the BERTScore reproducibility study."""
import os
import nltk
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

# Model used by bert_score for lang="en" by default: roberta-large, layer 17.
# Set env var BERTSCORE_MODEL=distilbert-base-uncased to use a smaller, faster model.
MODEL_TYPE = os.environ.get("BERTSCORE_MODEL", "roberta-large")
BATCH_SIZE = int(os.environ.get("BERTSCORE_BATCH", "32"))

_bleu = BLEU(effective_order=True)  # effective_order avoids zero BLEU on short sentences
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def sentence_bleu(cands, refs):
    return [_bleu.sentence_score(c, [r]).score for c, r in zip(cands, refs)]


def rouge_l(cands, refs):
    return [_rouge.score(r, c)["rougeL"].fmeasure for c, r in zip(cands, refs)]


def bertscore(cands, refs, verbose=False):
    from bert_score import score
    P, R, F = score(cands, refs, model_type=MODEL_TYPE, lang="en",
                    batch_size=BATCH_SIZE, verbose=verbose)
    return P.tolist(), R.tolist(), F.tolist()


def ensure_nltk():
    for pkg in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg, quiet=True)
