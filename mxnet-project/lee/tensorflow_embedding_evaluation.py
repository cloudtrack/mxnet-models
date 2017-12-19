import logging
from web.datasets.analogy import fetch_google_analogy
from web.embeddings import fetch_SG_GoogleNews
import numpy as np
import _pickle as pickle
from web.embeddings import load_embedding
w = load_embedding("tf_w2vec_dict.p", format="dict")
data = fetch_google_analogy()
from web.evaluate import evaluate_on_all
out_fname = "tf_results.csv"
results = evaluate_on_all(w)
print("Saving results...")
print(results)
results.to_csv(out_fname)
