from bert_score import score

candidates = [
    "A cat is sitting on a mat.",
    "The weather is sunny today."
]

references = [
    "A cat is resting on the mat.",
    "Today the weather is bright and sunny."
]

P, R, F1 = score(
    candidates,
    references,
    lang="en",
    verbose=True
)

for i in range(len(candidates)):
    print(f"\nExample {i + 1}")
    print(f"Candidate: {candidates[i]}")
    print(f"Reference: {references[i]}")
    print(f"Precision: {P[i].item():.4f}")
    print(f"Recall:    {R[i].item():.4f}")
    print(f"F1:        {F1[i].item():.4f}")