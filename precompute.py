from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# load data
df = pd.read_csv("ipc_training_dataset.csv")

# load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# generate embeddings
embeddings = model.encode(
    df["complaint_text"].astype(str).tolist(),
    convert_to_numpy=True,
    show_progress_bar=True
)

# save embeddings
np.save("case_embeddings.npy", embeddings)

print("✅ Embeddings saved!")