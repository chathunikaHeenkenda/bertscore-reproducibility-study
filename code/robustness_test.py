
from bert_score import score
import pandas as pd
import time

examples = [
    # Paraphrases
    ("The company increased its revenue.",
     "The company's revenue rose.", "paraphrase"),
    ("The meeting starts at nine.",
     "The meeting begins at 9 o'clock.", "paraphrase"),
    ("The dog is sleeping on the sofa.",
     "A dog is asleep on the couch.", "paraphrase"),
    ("The team won the match.",
     "The team was victorious in the game.", "paraphrase"),
    ("The temperature decreased overnight.",
     "It became colder during the night.", "paraphrase"),

    # Negation / contradictions
    ("The company increased its revenue.",
     "The company did not increase its revenue.", "negation"),
    ("The patient has a fever.",
     "The patient has no fever.", "negation"),
    ("The team won the match.",
     "The team lost the match.", "contradiction"),
    ("The store is open today.",
     "The store is not open today.", "negation"),
    ("The temperature increased.",
     "The temperature decreased.", "contradiction"),

    # Numerical changes
    ("Revenue increased by 12%.",
     "Revenue increased by 72%.", "number_change"),
    ("The company hired 20 workers.",
     "The company hired 200 workers.", "number_change"),
    ("The journey took 30 minutes.",
     "The journey took 13 minutes.", "number_change"),
    ("The price was $50.",
     "The price was $500.", "number_change"),
    ("The population reached 1 million.",
     "The population reached 10 million.", "number_change"),

    # Entity substitutions
    ("Microsoft announced a new product.",
     "Google announced a new product.", "entity_change"),
    ("Paris is the capital of France.",
     "Rome is the capital of France.", "entity_change"),
    ("Maria visited Sydney last week.",
     "Maria visited Melbourne last week.", "entity_change"),
    ("The novel was written by Jane Austen.",
     "The novel was written by Charles Dickens.", "entity_change"),
    ("The meeting is on Monday.",
     "The meeting is on Friday.", "entity_change"),

    # Irrelevant or unsupported additions
    ("The company reported higher profits.",
     "The company reported higher profits. The sky is blue.",
     "irrelevant_addition"),
    ("The cat slept on the chair.",
     "The cat slept on the chair. The ocean is very deep.",
     "irrelevant_addition"),
    ("The team won the final.",
     "The team won the final and received a $10 million bonus.",
     "unsupported_addition"),
    ("The report describes sales growth.",
     "The report describes sales growth and confirms fraud.",
     "unsupported_addition"),
    ("The train arrived at noon.",
     "The train arrived at noon. It was carrying 500 passengers.",
     "unsupported_addition"),
]

references = [x[0] for x in examples]
candidates = [x[1] for x in examples]
categories = [x[2] for x in examples]

start = time.time()

P, R, F1 = score(
    candidates,
    references,
    lang="en",
    verbose=True
)

results = pd.DataFrame({
    "reference": references,
    "candidate": candidates,
    "category": categories,
    "precision": P.tolist(),
    "recall": R.tolist(),
    "f1": F1.tolist()
})

results.to_csv("results/robustness_results.csv", index=False)

summary = (
    results.groupby("category")["f1"]
    .agg(["count", "mean", "std"])
    .reset_index()
)

summary.to_csv("results/robustness_summary.csv", index=False)

print("\nIndividual results:")
print(results[["category", "f1"]].to_string(index=False))

print("\nAverage F1 by category:")
print(summary.to_string(index=False))

print(f"\nRuntime: {time.time() - start:.2f} seconds")
print("\nSaved results to the results folder.")