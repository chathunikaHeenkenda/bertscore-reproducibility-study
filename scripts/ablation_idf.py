import pandas as pd
from scipy.stats import pearsonr, kendalltau
from bert_score import score
df = pd.read_csv("results/replication_wmt_segments_min3ann.csv")
c, r, h = df.mt.astype(str).tolist(), df.ref.astype(str).tolist(), df.score
rows=[]
for idf in [False, True]:
    P,R,F = score(c, r, model_type="roberta-large", lang="en", idf=idf, batch_size=16)
    for name,v in [("P",P),("R",R),("F1",F)]:
        v=v.numpy()
        rows.append(dict(idf=idf,component=name,pearson_r=round(pearsonr(v,h)[0],4),kendall_tau=round(kendalltau(v,h)[0],4),n=len(df)))
    print(rows[-3:], flush=True)
pd.DataFrame(rows).to_csv("/workspace/t6/ablation_idf_wmt_min3ann.csv", index=False)
