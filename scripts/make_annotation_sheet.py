"""Draw a stratified 20% sample (8 per edit type = 56 of 280 rows) of the perturbation
pairs for double annotation. annotator_1 is pre-filled from the construction label
(the author designed each edit; 1 = meaning preserved, 0 = meaning changed).
annotator_2 is left BLANK and must be filled by a real second person, blind to
annotator_1 and to the 'type' column (use annotation_sheet_blind.csv)."""
import pandas as pd
p = pd.read_csv("results/perturbation_scores.csv")
s = pd.concat([g.sample(8, random_state=42) for _, g in p.groupby("type", sort=False)])
s = s.sample(frac=1, random_state=42).reset_index(drop=True)
s.insert(0, "item", range(1, len(s) + 1))
s["annotator_1"] = (s.expected_quality == "high").astype(int)
s["annotator_2"] = ""
s[["item", "id", "type", "reference", "candidate", "annotator_1", "annotator_2"]].to_csv(
    "annotation/annotation_sheet.csv", index=False)
s[["item", "reference", "candidate"]].assign(label_1_preserved_0_changed="").to_csv(
    "annotation/annotation_sheet_blind.csv", index=False)
print(len(s), s.type.value_counts().to_dict())
