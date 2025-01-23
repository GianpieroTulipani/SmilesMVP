import pandas as pd
#from smilesmvp.config import PROCESSED_DATA_DIR
from dataset import Molecule3DMaskingDataset, collate_fn
from torch.utils.data import DataLoader

root_3d = f'/kaggle/input/geom-3d-nmol50000-nconf5-nupper1000'
#root_3d = f"{PROCESSED_DATA_DIR}/GEOM_3D_nmol50000_nconf5_nupper1000"
csv_path = f"{root_3d}/processed/smiles.csv"

print("Loading 3D dataset...")
dataset_3d = Molecule3DMaskingDataset(root=root_3d, dataset="GEOM_3D_nmol50000_nconf5_nupper1000", mask_ratio=0.15, smiles_path=csv_path)

dataloader = DataLoader(dataset_3d, batch_size=32, shuffle=True, collate_fn=collate_fn)

print("Iterating over the dataloader...")
print(" ")
for smiles_batch, batch_graph in dataloader:
    print("SMILES:", smiles_batch[:5]) 
    print("Graph batch:", batch_graph[:5])
    break