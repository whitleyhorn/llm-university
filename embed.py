import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.cohere_utils import co
import umap

df = pd.read_csv("https://github.com/cohere-ai/notebooks/raw/main/notebooks/data/hello-world-kw.csv", names=["search_term"])
df.head()

def embed_text(texts):
    """
      Turns a piece of text into embeddings
      Arguments:
        text(str): the text to be turned into embeddings
      Returns:
        embedding(list): the embeddings
    """
    output = co.embed(model="embed-english-v2.0", texts=texts)
    embeds = output.embeddings
    return embeds;

df["search_term_embeds"] = embed_text(df["search_term"].tolist())
embeds = np.array(df["search_term_embeds"].tolist())

# Compress the embeddings to 2D
reducer = umap.UMAP(n_neighbors=49)
umap_embeds = reducer.fit_transform(embeds)

# Store the 2D embeddings in the dataframe
df["x"] = umap_embeds[:, 0]
df["y"] = umap_embeds[:, 1]

# Plot the embeddings
plt.figure(figsize=(10, 6))
plt.scatter(df["x"], df["y"], alpha=0.5)
plt.title("UMAP Embeddings")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.show()
