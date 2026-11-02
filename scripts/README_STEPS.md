# BERTScore Reproducibility Study: step-by-step guide to the preliminary results

COMP8240 project. Paper: Zhang et al. (2020), *BERTScore: Evaluating Text Generation with BERT*, ICLR. Official code: https://github.com/Tiiiger/bert_score (pip package `bert-score`).

Each script was tested end to end. The outputs below are **examples from my machine** (Linux, 8-core CPU, no GPU, Python 3.13.5). Your numbers may differ slightly in the last decimal place.

BERTScore model used everywhere: **`roberta-large`, layer 17**. This is the package default for `lang="en"`, with no idf weighting and no baseline rescaling. On CPU, the whole pipeline took about **3 minutes** after downloads.

---

## 0. What you need
- Python 3.10 or newer (3.11–3.13 recommended). Check with `python --version` (`python3` on Mac/Linux).
- Git.
- About 3 GB of free disk: roughly 1.4 GB for the roberta-large model and about 1 GB for the WMT dataset download. The first run needs internet.

## 1. Clone your repo (with the official code as a submodule)
```bash
git clone --recurse-submodules https://github.com/chathunikaHeenkenda/bertscore-reproducibility-study.git
cd bertscore-reproducibility-study
# if you already cloned it without the submodule:
git submodule update --init --recursive
```

## 2. Create and activate a virtual environment
**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
# if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   (then retry)
```
Your prompt should now start with `(.venv)`.

## 3. Install the packages
Copy `requirements.txt` from this bundle into the repo root, then run:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
No NVIDIA GPU? You can install a much smaller CPU-only torch **first**, then the rest:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
If an exact version will not install on your Python, use the unpinned list instead and record what you got (step 5 does this):
```bash
pip install torch transformers bert-score sacrebleu rouge-score nltk datasets scipy pandas matplotlib
```

NLTK (optional; only used to split XSum documents into sentences):
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```
If this download fails, `load_new_dataset.py` falls back to a built-in splitter automatically.

## 4. Put the files in place and create the folders
Copy the `scripts/` folder from this bundle into the repo root. Also delete the oddly named old folder `scripts/test_bestscore.py/`.
```bash
# Mac/Linux
mkdir -p data/constructed data/new_existing data/original results figures
# Windows PowerShell
mkdir data\constructed, data\new_existing, data\original, results, figures -Force
```
Final layout:
```
bertscore-reproducibility-study/
├── original_bertscore/        (git submodule: official code)
├── scripts/
│   ├── common.py              (shared metric helpers + model choice)
│   ├── test_bertscore.py
│   ├── record_env.py
│   ├── reproduce_original.py
│   ├── create_perturbations.py
│   ├── evaluate_perturbations.py
│   └── load_new_dataset.py
├── data/  results/  figures/
└── requirements.txt
```
**Run every command from the repo root** (the folder that contains `scripts/`).

## 5. Record the environment
```bash
python scripts/record_env.py
```
Example from my machine (saved to `results/environment.txt`):
```
Python: 3.13.5
OS: Linux-6.12.94+-x86_64-with-glibc2.41
Processor: x86_64
CPU cores: 8
torch: 2.14.1+cpu
transformers: 5.19.0
bert-score (pip): 0.3.13
CUDA available: False
GPU: none (CPU only)
BERTScore model: roberta-large (layer 17), idf=False, no baseline rescaling
```

## 6. Sanity test (good to run live in the video)
```bash
python scripts/test_bertscore.py
```
The first run downloads roberta-large (about 1.4 GB). Example output:
```
Model: roberta-large

Candidate: The cat sat on the mat.
Reference: A cat was sitting on the mat.
P=0.9757  R=0.9714  F1=0.9736

Candidate: The stock market crashed on Monday.
Reference: Shares rose sharply at the start of the week.
P=0.9329  R=0.9278  F1=0.9304
```
Talking point: the second pair means the *opposite*, yet it still scores 0.93. Raw BERTScore values are squeezed into a narrow, high range, so only relative differences matter.

## 7. Replication on the original data (WMT17 de-en, human correlation)
```bash
python scripts/reproduce_original.py                      # 500 segments, de-en, 2017, seed 42
python scripts/reproduce_original.py --min_annotators 3   # less noisy subset (optional)
```
- **Data:** the Hugging Face dataset `RicardoRei/wmt-da-human-evaluation`, which holds the official WMT17–22 Direct Assessment human scores. The script keeps `lp == "de-en"` and `year == 2017`, samples 500 segments with seed 42, and correlates each metric with the human z-score (`score`).
- **Saves:** `results/replication_wmt.csv`, plus per-segment scores in `results/replication_wmt_segments.csv` and the sample in `data/original/`.

Example output (about 40 s on CPU, after a one-off dataset download of about 1 minute):
```
de-en 2017 (annotators >= 1): 21704 segments available
      metric  pearson_r  kendall_tau    lp  year   n  min_annotators         model paper_pearson
bertscore_f1     0.4949       0.3527 de-en  2017 500               1 roberta-large         0.745
        bleu     0.2944       0.1925 de-en  2017 500               1             -         0.432
      rougeL     0.3494       0.2346 de-en  2017 500               1             -           n/a

de-en 2017 (annotators >= 3): 1662 segments available
      metric  pearson_r  kendall_tau    lp  year   n  min_annotators         model paper_pearson
bertscore_f1     0.5374       0.3757 de-en  2017 500               3 roberta-large         0.745
        bleu     0.3735       0.2665 de-en  2017 500               3             -         0.432
      rougeL     0.4398       0.3047 de-en  2017 500               3             -           n/a
```
How to explain it in the video:
- **Replicated:** the *ranking* matches the paper. BERTScore correlates with humans better than sentence-BLEU, by about 0.16–0.20 Pearson here; the paper reports 0.745 vs 0.432 in Appendix Table 13.
- **Not replicated (yet):** the *absolute* values are lower. The likely reason is the data. The paper used the official WMT17 "DAseg" set: 560 segments per pair, each averaged over at least 15 human judgements. The Hugging Face data mostly has 1–2 judgements per segment (mean 1.5 in my sample), which makes the human scores noisy. Keeping only segments with ≥3 judgements raises every correlation (0.49 to 0.54 for BERTScore), which supports this explanation.
- **Next step:** run the official script on the official package (below).

### The official reproduce/ scripts (in `original_bertscore/reproduce/`)
The official repo contains `README.md`, `download_wmt17.sh`, `download_wmt18.sh`, `get_wmt17_sys_results.py` and `get_wmt18_seg_results.py`. Its README says to run:
```bash
cd original_bertscore/reproduce
bash download_wmt18.sh                                  # downloads WMT18 metrics package from ufallab.ms.mff.cuni.cz
python get_wmt18_seg_results.py -b 16 --model roberta-large   # WMT18 segment-level (Kendall tau-like, paper Table 4/17)
bash download_wmt17.sh
python get_wmt17_sys_results.py -b 16 --model roberta-large   # WMT17 system-level (paper Table 15)
```
Notes:
- These are bash scripts that use `wget`. On Windows, use Git Bash or WSL, or download the `.tgz` URLs inside the `.sh` files by hand.
- From my machine the `ufallab.ms.mff.cuni.cz` server did not respond (timed out), so I could not run them. Try them on yours. If they also fail for you, say so in the video ("the original data host is unreachable, so I used the same WMT DA judgements from Hugging Face"). The marking rewards honesty about what did and didn't work.
- Running all language pairs with roberta-large on CPU is slow. Start with one pair.

## 8. Build the constructed dataset (your novel contribution)
```bash
python scripts/create_perturbations.py
```
Output:
```
Wrote 280 rows (40 references x 7 types) to data/constructed/perturbations.csv
```
The dataset has 40 hand-written news/finance/science references, each containing numbers and named entities. Each reference has 7 hand-written candidates:

| type | meaning | expected_quality |
|---|---|---|
| paraphrase | same meaning, restructured sentence | high |
| lexical_variation | same meaning, synonyms swapped | high |
| negation | meaning reversed | low |
| numerical_change | a number changed | low |
| entity_substitution | a person/place/company swapped | low |
| irrelevant_addition | an unrelated clause appended | low |
| hallucination | an unsupported but plausible "fact" appended | low |

Columns: `id, ref_id, reference, candidate, type, expected_quality`.

## 9. Score the constructed dataset
```bash
python scripts/evaluate_perturbations.py
```
This takes about 15 s. It saves `results/perturbation_scores.csv`, `results/perturbation_summary.csv`, `results/pairwise_accuracy.csv` and `figures/bertscore_by_type.png`. Example output:
```
Mean scores per perturbation type (BERTScore model: roberta-large):
                    expected_quality  bertscore_p  bertscore_r  bertscore_f1     bleu  rougeL
paraphrase                      high       0.9664       0.9702        0.9683  25.6320  0.4636
lexical_variation               high       0.9899       0.9902        0.9900  61.6345  0.8278
negation                         low       0.9763       0.9745        0.9754  65.8799  0.8420
numerical_change                 low       0.9909       0.9912        0.9910  72.9273  0.8960
entity_substitution              low       0.9879       0.9889        0.9884  84.4159  0.9200
irrelevant_addition              low       0.9472       0.9878        0.9670  63.2753  0.8277
hallucination                    low       0.9531       0.9927        0.9725  64.9107  0.8337

Pairwise accuracy: share of the 40 references where the high-quality candidate
(paraphrase or lexical_variation) is scored strictly above the low-quality one:
high_quality_anchor    low_quality_type  BERTScore_F1  BLEU  ROUGE-L
         paraphrase            negation         0.300 0.075    0.000
         paraphrase    numerical_change         0.000 0.000    0.000
         paraphrase entity_substitution         0.050 0.000    0.000
         paraphrase irrelevant_addition         0.600 0.000    0.000
         paraphrase       hallucination         0.350 0.025    0.000
         paraphrase                 ALL         0.260 0.020    0.000
  lexical_variation            negation         0.825 0.350    0.375
  lexical_variation    numerical_change         0.425 0.125    0.125
  lexical_variation entity_substitution         0.525 0.075    0.075
  lexical_variation irrelevant_addition         0.975 0.400    0.525
  lexical_variation       hallucination         0.950 0.425    0.425
  lexical_variation                 ALL         0.740 0.275    0.305
```
Findings to present:
1. **BERTScore is far better than BLEU/ROUGE-L at rewarding meaning, but it is still fooled.** Against a real paraphrase, BERTScore ranks the paraphrase higher only 26% of the time overall. BLEU manages 2% and ROUGE-L 0%.
2. **Numbers and entities are BERTScore's blind spot.** A wrong number (mean F1 0.991) or a swapped entity (0.988) scores *higher* than a correct paraphrase (0.968), and as high as a correct synonym rewrite (0.990). Pairwise accuracy is only 0.43 and 0.53 even against lexical_variation.
3. **Negation is partly caught** (0.83 vs lexical_variation), but the negation box in the figure is very wide.
4. **Additions and hallucinations show up in precision, not recall.** Recall stays about 0.99 because all of the reference is still covered, while precision drops to about 0.95. F1 therefore punishes them, but only a little.

## 10. Prepare the new existing dataset (XSum)
```bash
python scripts/load_new_dataset.py
```
This loads the first 200 test examples of `EdinburghNLP/xsum` (falling back to `cnn_dailymail 3.0.0` if XSum fails). It saves `data/new_existing/xsum_200.csv` and `results/new_dataset_baseline.csv`, and scores a lead-1 baseline (the first sentence of the article) against the reference summary. Example output (about 25 s):
```
Dataset: xsum (test split, first 200 examples)
Avg document length: 400.3 words (median 282)
Avg summary length:  21.2 words (median 20)
Compression ratio:   19.9x

Lead-1 baseline vs reference summary (mean over 200), model roberta-large:
bertscore_p     0.8585
bertscore_r     0.8655
bertscore_f1    0.8618
bleu            2.3723
rougeL          0.1237
```
Talking points: XSum summaries are single, highly abstractive sentences (BBC articles, about 20x compression). The lead sentence shares few words with the summary (BLEU 2.4, ROUGE-L 0.12), yet BERTScore gives 0.86. XSum is also known for hallucinated content in references, so it is a good stress test for an embedding metric.

## 11. Commit and push
```bash
git add scripts requirements.txt README_STEPS.md data/constructed results figures
git commit -m "Preliminary results: WMT17 replication, constructed perturbation set, XSum baseline"
git push
```
(`data/new_existing/xsum_200.csv` and `data/original/` can be regenerated by the scripts. Add them too if you want.)

## Using a smaller/faster model (if roberta-large is too slow or runs out of memory)
```bash
# Mac/Linux
export BERTSCORE_MODEL=distilbert-base-uncased
# Windows PowerShell
$env:BERTSCORE_MODEL="distilbert-base-uncased"
```
Then re-run the scripts and **state in the video that you used a different model from the paper**. You can also lower the batch size with `BERTSCORE_BATCH=8`.

## Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: common` | Run from the repo root: `python scripts/xxx.py` (not from inside `scripts/`). |
| `pip install -r requirements.txt` fails on a version | Use the unpinned install line in step 3 and record versions with `record_env.py`. |
| torch install is huge or slow | Install the CPU wheel first (step 3). |
| Out of memory / very slow | Set `BERTSCORE_BATCH=8`, or use `distilbert-base-uncased` (see above). Close other apps. roberta-large needs about 3 GB of RAM. |
| HF warning "unauthenticated requests" | Harmless. Optionally run `huggingface-cli login`. |
| "LOAD REPORT ... UNEXPECTED/MISSING lm_head/pooler" | Harmless. BERTScore only uses the encoder layers. |
| XSum fails to load | The script falls back to CNN/DailyMail automatically. Or `pip install -U datasets`. |
| NLTK punkt download fails | Ignore it. A regex sentence splitter is used automatically. |
| `bash download_wmt18.sh` fails | The server may be down. Report it honestly and use the `reproduce_original.py` route. On Windows use Git Bash or WSL. |
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
