import argparse
import os
from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=0)
parser.add_argument('--device', type=int, default=0)

parser.add_argument('--input_data_dir', type=str, default=Path('/kaggle/input/geom-3d-regression'))#geom-3d-regression
parser.add_argument('--dataset', type=str, default='GEOM_3D_nmol50000_nconf5_nupper1000')#GEOM_3D_nmol50000_nconf5_nupper1000_morefeat
parser.add_argument('--num_workers', type=int, default=os.cpu_count())

parser.add_argument('--batch_size', type=int, default=32) #32 for classification, 16 for regression
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0001) #0.001
parser.add_argument('--decay', type=float, default=0)

parser.add_argument('--emb_dim', type=int, default=384)
parser.add_argument('--model_3d', type=str, default='schnet', choices=['schnet'])

parser.add_argument('--num_filters', type=int, default=128)
parser.add_argument('--num_interactions', type=int, default=6)
parser.add_argument('--num_gaussians', type=int, default=51)
parser.add_argument('--cutoff', type=float, default=10)
parser.add_argument('--readout', type=str, default='mean', choices=['mean', 'add'])
parser.add_argument('--schnet_lr_scale', type=float, default=1)
parser.add_argument('--chemBERTa_lr_scale', type=float, default=1)
parser.add_argument('--lr_scale', type=float, default=1)#0.1
parser.add_argument('--CL_neg_samples', type=int, default=1)
parser.add_argument('--CL_similarity_metric', type=str, default='EBM_dot_prod',
                    choices=['InfoNCE_dot_prod', 'EBM_dot_prod'])

parser.add_argument('--T', type=float, default=0.1)
parser.add_argument('--normalize', dest='normalize', action='store_true')
parser.add_argument('--no_normalize', dest='normalize', action='store_false')
parser.add_argument('--SSL_masking_ratio', type=float, default=0.15)


parser.add_argument('--AE_loss', type=str, default='l2', choices=['l1', 'l2', 'cosine'])
parser.add_argument('--detach_target', dest='detach_target', action='store_true')
parser.add_argument('--no_detach_target', dest='detach_target', action='store_false')
parser.set_defaults(detach_target=True)

parser.add_argument('--beta', type=float, default=1)#2
parser.add_argument('--dropout_ratio', type=float, default=0.5)
parser.add_argument('--alpha_1', type=float, default=1)
parser.add_argument('--alpha_2', type=float, default=1) #0.1

parser.add_argument('--input_model_dir', type=str, default=Path('/kaggle/input/regression-pretrained'))
parser.add_argument('--output_model_dir', type=str, default=Path('/kaggle/working'))

args = parser.parse_args()
#print('arguments\t', args)