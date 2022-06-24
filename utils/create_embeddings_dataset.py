import time

import pandas as pd
from sentence_transformers import SentenceTransformer


df_train = pd.read_csv("./data/go_emotions_test.csv")
model = SentenceTransformer("all-mpnet-base-v2")

start_time = time.time()
df_train["embeddings"] = df_train["text"].apply(lambda x: model.encode(x))
elapsed_time = time.time() - start_time
print(elapsed_time)
df_train.to_csv("./data/go_emotions_test_embeddings.csv", index=False)
