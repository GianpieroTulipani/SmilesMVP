from loguru import logger
from tqdm import tqdm
import sys
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader
from os.path import join
import numpy as np
import pandas as pd
import deepchem as dc
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score

sys.path.append('/kaggle/working/SmilesMVP/smilesmvp')
sys.path.append('/kaggle/working/SmilesMVP/smilesmvp/modeling')
from config import args

class ChemBERTaClassifier(nn.Module):
    def __init__(self, model_path, num_tasks):
        super(ChemBERTaClassifier, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.chemberta = AutoModel.from_pretrained(model_path)
        self.fc = nn.Linear(self.chemberta.config.hidden_size, num_tasks)
    
    def forward(self, smiles_batch):
        tokens = self.tokenizer(smiles_batch, padding=True, truncation=True, return_tensors='pt').to(self.chemberta.device)
        molecule_repr = self.chemberta(**tokens, return_dict=True).last_hidden_state[:, 0]
        return self.fc(molecule_repr)

def train(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for smiles_batch, labels in tqdm(loader):
        labels = labels.to(device)
        pred = model(smiles_batch)

        loss = criterion(pred, labels.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)

def eval(model, device, loader):
    model.eval()
    y_true, y_scores = [], []

    for smiles_batch, labels in loader:
        labels = labels.to(device)
        with torch.no_grad():
            pred = model(smiles_batch)
        
        y_true.append(labels.cpu())
        y_scores.append(pred.cpu())
    
    y_true = torch.cat(y_true, dim=0).numpy()
    y_scores = torch.cat(y_scores, dim=0).numpy()
    
    roc_list = [roc_auc_score(y_true[:, i], y_scores[:, i]) for i in range(y_true.shape[1])]
    
    return sum(roc_list) / len(roc_list), y_true, y_scores

if __name__ == '__main__':
    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device('cuda:' + str(args.device)) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Load dataset using DeepChem
    tasks, datasets, transformers = dc.molnet.load_tox21(featurizer='Raw')
    train_dataset, valid_dataset, test_dataset = datasets

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = ChemBERTaClassifier("DeepChem/ChemBERTa-77M-MLM", len(tasks)).to(device)

    model_param_group = [{'params': model.chemberta.parameters()},
                        {'params': model.fc.parameters(),
                        'lr': args.lr * args.lr_scale}]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    if not args.input_model_dir == '':
        model.load_state_dict(torch.load(join(args.input_model_dir, '_model.pth')))
        logger.info(f"Loaded pretrained model from {args.pretrained_model_path}")

    best_val_roc = -1
    for epoch in range(1, args.epochs + 1):
        loss_acc = train(model, device, train_loader, optimizer)
        logger.info(f'Epoch: {epoch}\nLoss: {loss_acc}')

        val_roc, val_target, val_pred = eval(model, device, val_loader)
        test_roc, test_target, test_pred = eval(model, device, test_loader)
        logger.info(f'val: {val_roc:.6f}\ttest: {test_roc:.6f}\n')

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            if args.output_model_dir:
                torch.save(model.state_dict(), f"{args.output_model_dir}/model_best.pth")
                np.savez(f"{args.output_model_dir}/evaluation_best.npz", val_target=val_target, val_pred=val_pred, test_target=test_target, test_pred=test_pred)

    if args.output_model_dir:
        torch.save(model.state_dict(), f"{args.output_model_dir}/model_final.pth")