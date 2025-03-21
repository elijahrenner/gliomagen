import torch
import numpy as np
import json
from torch.utils.data import DataLoader
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from dataset import FCD2Dataset  # now you can import FCD2Dataset
from sklearn.cluster import KMeans

# Create the dataset (no augmentation needed for histogram extraction)
dataset = FCD2Dataset(root_dir='../../data/FCD2', test_txt_dir='../../data/FCD2/Pathological/test.txt', augmentation=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

all_hists = []
for batch in loader:
    # 'hist' is computed in __getitem__ as a 16-bin histogram
    hist = batch['hist'].squeeze(0).numpy()  # shape (16,)
    if np.sum(hist) > 0:  # optionally filter out zero histograms
        all_hists.append(hist)
all_hists = np.stack(all_hists, axis=0)

# Cluster the histograms into (say) 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(all_hists)
centers = kmeans.cluster_centers_

# Save as JSON (format matching your inference script)
clusters = [{"n_class": 2, "centers": centers.tolist()}]
with open("clusters.json", "w") as f:
    json.dump(clusters, f, indent=4)