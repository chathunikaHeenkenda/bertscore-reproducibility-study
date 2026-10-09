
from bert_score import score
import pandas as pd
import time
import os

references = [
    "The company increased its revenue.",
    "The company increased its revenue.",
    "The company increased its revenue.",
    "Revenue increased by 12%."
]

candidates = [
    "The company's revenue rose.",
    "The company decreased its revenue.",
    "The weather is cold today.",
    "Revenue increased by 72%."
]

print("Running BERTScore...")
start = time.time()

precision, recall, f1 = score(
    candidates,
    references,
    lang="en",
    verbose=True
)

results = pd.DataFrame({
    "reference": references,
    "candidate": candidates,
    "precision": precision.tolist(),
    "recall": recall.tolist(),
    "f1": f1.tolist()
})

os.makedirs("results", exist_ok=True)
results.to_csv("results/preliminary_results.csv", index=False)

print("\nPreliminary results:")
print(results.to_string(index=False))
print(f"\nRuntime: {time.time() - start:.2f} seconds")
print("\nSaved to results/preliminary_results.csv")