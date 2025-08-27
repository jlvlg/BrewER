import pandas as pd
import pickle as pkl
import pandas as pd
from pathlib import Path

Path("raw_data/datasets").mkdir(parents=True, exist_ok=True)
Path("data").mkdir(parents=True, exist_ok=True)

l_ds_name = "dblp"
r_ds_name = "google_scholar"
id_col = "ID"

raw_ds_path = "raw_data/datasets/"
raw_lab_path = "raw_data/labeled_data.csv"
raw_cand_path = "raw_data/candset.csv"
cand_path = "data/candidates.pkl"
ds_path = "data/dataset.csv"
match_path = "data/matches.csv"
lab_path = "data/labels.csv"

l_ds = pd.read_csv(raw_ds_path + l_ds_name + ".csv", encoding="latin")
l_ds_len = len(l_ds)
l_ds["_id"] = l_ds.index
l_ds["_id"] = l_ds["_id"].apply(
    lambda x: l_ds_name + "_" + str(x).zfill(len(str(l_ds_len)))
)
old_ids = list(l_ds[id_col])
new_ids = list(l_ds["_id"])
l_map = {old_ids[i]: new_ids[i] for i in range(0, len(l_ds))}
l_ds = l_ds.drop(columns=[id_col])
l_ds.columns = l_ds.columns.str.lower()

# Load the second dataset and prepare its identifiers
r_ds = pd.read_csv(raw_ds_path + r_ds_name + ".csv", encoding="latin")
r_ds_len = len(r_ds)
r_ds["_id"] = r_ds.index
r_ds["_id"] = r_ds["_id"].apply(
    lambda x: r_ds_name + "_" + str(x).zfill(len(str(r_ds_len)))
)
old_ids = list(r_ds[id_col])
new_ids = list(r_ds["_id"])
r_map = {old_ids[i]: new_ids[i] for i in range(0, len(r_ds))}
r_ds = r_ds.drop(columns=[id_col])
r_ds.columns = r_ds.columns.str.lower()

# Append the two datasets
ds = pd.concat([l_ds, r_ds], ignore_index=True, join="inner").set_index("_id")

# # Preprocess numeric columns
ds["year"] = pd.to_numeric(ds["year"], errors="coerce")
ds["number"] = pd.to_numeric(ds["number"], errors="coerce")
ds["volume"] = pd.to_numeric(ds["volume"], errors="coerce")

# Save the obtained dataset
ds.to_csv(ds_path)

# Load the labeled data
lab = pd.read_csv(raw_lab_path, comment="#")

# Rename identifier columns
lab = lab.rename(columns={f"ltable.{id_col}": "l_id", f"rtable.{id_col}": "r_id"})

# Filter out useless columns and map the identifiers
l_old_ids = list(lab["l_id"])
r_old_ids = list(lab["r_id"])
labels = list(lab["is_match"])
new_ids = [
    (l_map[l_old_ids[i]], r_map[r_old_ids[i]], labels[i]) for i in range(0, len(lab))
]
lab = pd.DataFrame(new_ids, columns=["l_id", "r_id", "label"])
lab = lab.reset_index(drop=True)

# Keep only the matches and filter out the labels
gt = lab[lab["label"] == 1][["l_id", "r_id"]]
gt = gt.reset_index(drop=True)

# Save the obtained matches
gt.to_csv(match_path, index=False)

candset = pd.read_csv(raw_cand_path, comment="#")

pairs = list(
    zip(candset[f"ltable.{id_col}"].map(l_map), candset[f"rtable.{id_col}"].map(r_map))
)

with open(cand_path, "wb") as file:
    pkl.dump(pairs, file)
