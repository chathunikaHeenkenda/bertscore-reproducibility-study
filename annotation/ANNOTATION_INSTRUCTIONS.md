# Annotation instructions (second annotator)

Thank you for helping. This takes about 15-20 minutes.

**Open `annotation_sheet_blind.csv`** (in Excel, Google Sheets or Numbers). Do NOT open
`annotation_sheet.csv`. It contains the first annotator's labels and the edit type, and seeing them would bias your labels.

There are 56 rows. Each row has a REFERENCE sentence and a CANDIDATE sentence.
In the column `label_1_preserved_0_changed`, type:

- **1** if the candidate says the same thing as the reference. The wording can be different, but every fact must be unchanged: numbers, names, places, dates, and whether something did or did not happen.
- **0** if the candidate changes the meaning. That includes a wrong number or name, a negation, a contradiction, or added information that the reference does not support or that is off-topic.

Rules:
- Work alone. Don't discuss items with the author until you have finished.
- Label every row with 0 or 1, with no blanks.
- If you are unsure, pick the closer option. You can add a note in an extra column.
- Save the file as CSV with the same name and send it back.

The author then runs:
    python scripts/annotation_agreement.py --blind annotation/annotation_sheet_blind.csv
This prints Cohen's kappa and writes `results/annotation_agreement.csv`.
