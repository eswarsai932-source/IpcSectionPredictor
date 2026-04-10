import pandas as pd

def load_ipc_lookup(csv_path="ipc_sections.csv"):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    ipc_dict = {}

    for _, row in df.iterrows():
        section = str(row.get("Section", "")).strip()

        if section == "" or section.lower() == "nan":
            continue

        # ✅ Convert IPC_379 -> 379
        section = section.replace("IPC_", "").strip()

        ipc_dict[section] = {
            "description": str(row.get("Description", "")).strip(),
            "offense": str(row.get("Offense", "")).strip(),
            "punishment": str(row.get("Punishment", "")).strip()
        }
        break

    return ipc_dict
#print(load_ipc_lookup())