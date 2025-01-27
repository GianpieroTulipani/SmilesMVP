from loguru import logger
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.append('/kaggle/working/SmilesMVP/smilesmvp')
sys.path.append('/kaggle/working/SmilesMVP/smilesmvp/modeling')
sys.path.append('/kaggle/working/SmilesMVP/smilesmvp/models')
from config import args
from models.vae import VariationalAutoEncoder
from models.schnet import SchNet
from modeling.contrastive_loss import dual_CL
from dataset import Molecule3DMaskingDataset, collate_fn
from torch.utils.data import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_model(save_best):
    if args.output_model_dir:
        if save_best:
            global optimal_loss
            logger.info(f'Saving best model with loss: {optimal_loss:.5f}')
            torch.save(molecule_model_smiles.state_dict(), args.output_model_dir / '_model.pth')
            saver_dict = {
                'model': molecule_model_smiles.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir / '_model_complete.pth')
        else:
            torch.save(molecule_model_smiles.state_dict(), args.output_model_dir / '_model_final.pth')
            saver_dict = {
                'model': molecule_model_smiles.state_dict(),
                'model_3D': molecule_model_3D.state_dict(),
                'AE_2D_3D_model': AE_2D_3D_model.state_dict(),
                'AE_3D_2D_model': AE_3D_2D_model.state_dict(),
            }
            torch.save(saver_dict, args.output_model_dir / '_model_complete_final.pth')
    return

def train(args, molecule_model_smiles, device, loader, optimizer):
    molecule_model_smiles.train()
    molecule_model_3D.train()
    
    CL_loss_accum, CL_acc_accum, AE_loss_accum = 0, 0, 0

    def tokenizer_fn(x):
        return tokenizer(x, padding=True, truncation=True, return_tensors='pt')

    for smiles_batch, graph_batch in tqdm(loader):
        graph_batch = graph_batch.to(device)
        molecule_smiles_repr = molecule_model_smiles(**tokenizer_fn(smiles_batch).to(device), return_dict=True).last_hidden_state[:, 0]
        molecule_3D_repr = molecule_model_3D(graph_batch.x[:, 0], graph_batch.positions, graph_batch.batch)
        
        CL_loss, CL_acc = dual_CL(molecule_smiles_repr, molecule_3D_repr, args)
        AE_loss_1 = AE_2D_3D_model(molecule_smiles_repr, molecule_3D_repr)
        AE_loss_2 = AE_3D_2D_model(molecule_3D_repr, molecule_smiles_repr)
        AE_loss = (AE_loss_1 + AE_loss_2) / 2

        CL_loss_accum += CL_loss.detach().cpu().item()
        CL_acc_accum += CL_acc
        AE_loss_accum += AE_loss.detach().cpu().item()

        loss = args.alpha_1 * CL_loss + args.alpha_2 * AE_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    global optimal_loss, patience_counter
    CL_loss_accum /= len(loader)
    CL_acc_accum /= len(loader)
    AE_loss_accum /= len(loader)
    temp_loss = args.alpha_1 * CL_loss_accum + args.alpha_2 * AE_loss_accum

    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)
        patience_counter = 0
    else:
        patience_counter += 1
    
    logger.info(f'CL Loss: {CL_loss_accum:.5f}	CL Acc: {CL_acc_accum:.5f}	AE Loss: {AE_loss_accum:.5f}')
    return patience_counter

if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
        torch.cuda.set_device(args.device)

    data_root = args.input_data_dir / args.dataset
    dataset = Molecule3DMaskingDataset(data_root, dataset=args.dataset, mask_ratio=args.SSL_masking_ratio, smiles_path=args.input_data_dir / args.dataset / 'processed' / 'smiles.csv')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    
    tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
    molecule_model_smiles = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)
    
    molecule_model_3D = SchNet(
        hidden_channels=args.emb_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
        num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout).to(device)
    
    AE_2D_3D_model = VariationalAutoEncoder(
        emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
    AE_3D_2D_model = VariationalAutoEncoder(
        emb_dim=args.emb_dim, loss=args.AE_loss, detach_target=args.detach_target, beta=args.beta).to(device)
    
    model_param_group = [
        {'params': molecule_model_smiles.parameters(), 'lr': args.lr * args.chemBERTa_lr_scale},
        {'params': molecule_model_3D.parameters(), 'lr': args.lr * args.schnet_lr_scale},
        {'params': AE_2D_3D_model.parameters(), 'lr': args.lr * args.chemBERTa_lr_scale},
        {'params': AE_3D_2D_model.parameters(), 'lr': args.lr * args.schnet_lr_scale}
    ]
    
    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    optimal_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    logger.info("Starting model training...")
    for epoch in range(1, args.epochs + 1):
        logger.info(f'Epoch: {epoch}')
        patience_counter = train(args, molecule_model_smiles, device, dataloader, optimizer)
        if patience_counter >= max_patience:
            logger.info("Early stopping triggered!")
            break
    
    save_model(save_best=False)