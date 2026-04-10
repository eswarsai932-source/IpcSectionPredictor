import pandas as pd
import random
import re

# -----------------------------
# Simple text augmentation
# -----------------------------
def normalize_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def augment_text(text):
    text = normalize_text(text)

    replacements = {
        "mobile phone": ["phone", "smartphone", "cell phone"],
        "bike": ["motorcycle", "two-wheeler", "vehicle"],
        "scooter": ["two-wheeler", "vehicle"],
        "market": ["shopping area", "local market", "bazaar"],
        "office": ["workplace", "office premises"],
        "railway station": ["station", "railway premises"],
        "bus": ["bus stop", "public bus"],
        "parking area": ["parking", "parking lot"],
        "stole": ["snatched", "took away", "stolen"],
        "robbed": ["mugged", "looted"],
        "threatened": ["warned", "intimidated"],
        "hit": ["assaulted", "beat"],
        "injury": ["hurt", "wound"],
        "damaged": ["broke", "destroyed"]
    }

    for k, vals in replacements.items():
        if k in text.lower():
            # replace preserving original case style roughly
            choice = random.choice(vals)
            text = re.sub(k, choice, text, flags=re.IGNORECASE)

    # random add small phrase sometimes
    add_phrases = [
        "",
        "",
        " It happened suddenly.",
        " The accused escaped immediately.",
        " I request legal action.",
        " Please register a case."
    ]
    text = text + random.choice(add_phrases)

    return normalize_text(text)


# -----------------------------
# MAIN: Expand dataset
# -----------------------------
input_csv = "ipc_training_dataset.csv"
output_csv = "ipc_training_dataset_expanded.csv"

df = pd.read_csv(input_csv)

# Remove empty rows
df = df.dropna(subset=["complaint_text", "ipc_sections"])
df["complaint_text"] = df["complaint_text"].astype(str).apply(normalize_text)
df["ipc_sections"] = df["ipc_sections"].astype(str).str.replace(" ", "")

# How many new samples per row?
AUGMENT_MULTIPLIER = 6   # ✅ 6x bigger (450 -> ~2700)
# Increase to 10 for 4500 rows

new_rows = []

for _, row in df.iterrows():
    base_text = row["complaint_text"]
    sections = row["ipc_sections"]

    # Keep original
    new_rows.append({"complaint_text": base_text, "ipc_sections": sections})

    # Add augmented versions
    for _ in range(AUGMENT_MULTIPLIER - 1):
        aug = augment_text(base_text)
        new_rows.append({"complaint_text": aug, "ipc_sections": sections})

expanded_df = pd.DataFrame(new_rows)

# Shuffle
expanded_df = expanded_df.sample(frac=1).reset_index(drop=True)

expanded_df.to_csv(output_csv, index=False, encoding="utf-8")

print("✅ Expanded dataset saved:", output_csv)
print("Old rows:", len(df))
print("New rows:", len(expanded_df))