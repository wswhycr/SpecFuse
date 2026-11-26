"""Strain-graph loader for the SLT antigenic-distance benchmarks.

One sample is an ordered strain pair (source, reference) with an observed
haemagglutination-inhibition distance.  Each strain is encoded as a sequence of
100-d ProtVec trigram vectors; the strain graph stores the ordered pair list as
directed edges together with their distances.

Data expected
-------------
data/<dataset>.xlsx
    Columns: strain1, strain2, seq1, seq2, distance (optionally year1, year2).
data/protVec_100d_3grams.csv
    Tab-separated ProtVec table with a ``words`` column of trigrams.
"""
import os

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
from torch_geometric.data import Data

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")

# Dataset keys used on the command line -> file stems shipped in data/.
DATASETS = {
    "h1n1_566": "h1n1_566",
    "h3n2_566_asym": "h3n2_566_asym",
    "bvic_585_asym": "bvic_585_asym",
}


class graphDataset(Dataset):
    """Directed strain-pair graph with ProtVec node features."""

    def __init__(self, subtype, strip=False):
        if subtype not in DATASETS:
            raise KeyError(f"unknown dataset {subtype!r}; "
                           f"choose from {sorted(DATASETS)}")
        path = os.path.join(DATA, DATASETS[subtype] + ".xlsx")
        strain = pd.read_excel(path)

        strain["strain1_original"] = strain["strain1"].copy()
        strain["strain2_original"] = strain["strain2"].copy()
        strain["strain1"] = strain["strain1"].astype(str).str.lower()
        strain["strain2"] = strain["strain2"].astype(str).str.lower()

        unique_names = pd.concat([
            strain[["strain1", "strain1_original"]].rename(
                columns={"strain1": "lower", "strain1_original": "original"}),
            strain[["strain2", "strain2_original"]].rename(
                columns={"strain2": "lower", "strain2_original": "original"})
        ]).drop_duplicates()
        self.lower_to_original = pd.Series(
            unique_names["original"].values,
            index=unique_names["lower"]).to_dict()

        lables = preprocessing.LabelEncoder()
        unique_labels = np.unique(
            list(strain["strain1"]) + list(strain["strain2"]))
        lables.fit(unique_labels)
        strain["id1"] = lables.transform(list(strain["strain1"].values))
        strain["id2"] = lables.transform(list(strain["strain2"].values))
        strain.sort_values(by=["id1", "id2"], inplace=True)

        # Ordered pairs are stored as directed edges; the direction is the
        # source -> reference orientation induced by the SLT target.
        oriN = strain["id1"].values
        endN = strain["id2"].values
        edge_index = torch.tensor(np.array([oriN, endN]), dtype=torch.long)
        edge_attr = torch.tensor(strain["distance"].values, dtype=torch.float)

        provect = pd.read_csv(os.path.join(DATA, "protVec_100d_3grams.csv"),
                              delimiter="\t")
        trigrams = list(provect["words"])
        trigram_to_idx = {trigram: i for i, trigram in enumerate(trigrams)}
        trigram_vecs = provect.loc[:, provect.columns != "words"].values

        x = []
        virus_sequences = []
        virus_names = []
        for strain_name in lables.classes_:
            if len(strain[strain["strain1"] == strain_name]["seq1"]) > 0:
                seq = strain[strain["strain1"] == strain_name]["seq1"].values[0]
            else:
                seq = strain[strain["strain2"] == strain_name]["seq2"].values[0]
            seq = str(seq).upper()
            if strip:
                seq = seq.strip()
            virus_sequences.append(seq)
            strain_embedding = []
            for i in range(0, len(seq) - 2):
                trigram = seq[i:i + 3]
                if "-" in trigram:
                    tri_embedding = trigram_vecs[trigram_to_idx["<unk>"]]
                else:
                    try:
                        tri_embedding = trigram_vecs[trigram_to_idx[trigram]]
                    except KeyError:
                        tri_embedding = trigram_vecs[trigram_to_idx["<unk>"]]
                strain_embedding.append(tri_embedding)
            x.append(strain_embedding)
            original_name = self.lower_to_original.get(strain_name, strain_name)
            virus_names.append(original_name)

        x = np.array(x)
        x = torch.tensor(x.reshape(x.shape[0], -1), dtype=torch.float)
        self.data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                         virus_names=virus_names,
                         virus_sequences=virus_sequences)

        # Year-indexed splits (temporal generalisation study) carry year1/year2.
        if "year1" in strain.columns and "year2" in strain.columns:
            self.data.year1 = torch.tensor(
                pd.to_numeric(strain["year1"], errors="coerce").fillna(-1).values,
                dtype=torch.long)
            self.data.year2 = torch.tensor(
                pd.to_numeric(strain["year2"], errors="coerce").fillna(-1).values,
                dtype=torch.long)