"""Cohen's kappa between two annotators.
Usage: python scripts/annotation_agreement.py [annotation/annotation_sheet.csv]
If annotator 2 filled annotation_sheet_blind.csv, merge first with
  python scripts/annotation_agreement.py --blind annotation/annotation_sheet_blind.csv
Writes results/annotation_agreement.csv."""
import sys, pandas as pd
import numpy as np
def cohen_kappa_score(a, b):
    a, b = np.asarray(a), np.asarray(b); labs = np.union1d(a, b)
    po = (a == b).mean(); pe = sum((a == l).mean() * (b == l).mean() for l in labs)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)
def confusion_matrix(a, b, labels):
    return np.array([[int(((np.asarray(a) == i) & (np.asarray(b) == j)).sum()) for j in labels] for i in labels])
args = sys.argv[1:]
df = pd.read_csv("annotation/annotation_sheet.csv")
if args[:1] == ["--blind"]:
    b = pd.read_csv(args[1])[["item", "label_1_preserved_0_changed"]]
    df = df.drop(columns="annotator_2").merge(b.rename(columns={"label_1_preserved_0_changed": "annotator_2"}), on="item")
elif args:
    df = pd.read_csv(args[0])
df = df.dropna(subset=["annotator_1", "annotator_2"])
if len(df) == 0:
    sys.exit("annotator_2 is empty - get a second person to label the sheet first.")
a1, a2 = df.annotator_1.astype(int), df.annotator_2.astype(int)
k = cohen_kappa_score(a1, a2); agree = (a1 == a2).mean()
print(f"n={len(df)}  raw agreement={agree:.3f}  Cohen's kappa={k:.3f}")
print("confusion (rows=annotator_1 0/1, cols=annotator_2 0/1):\n", confusion_matrix(a1, a2, labels=[0, 1]))
print("disagreements:\n", df[a1 != a2][["item", "id", "type"]].to_string(index=False))
pd.DataFrame([dict(n=len(df), raw_agreement=round(agree, 4), cohen_kappa=round(k, 4))]).to_csv(
    "results/annotation_agreement.csv", index=False)
