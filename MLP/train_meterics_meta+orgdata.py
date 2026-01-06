import os
import random
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from rnd_dataloader import create_dataloaders

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

CACHE_DIR = "./cache"
LOG_DIR = "./logs"

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"train_{timestamp}.log")
    
    logger = logging.getLogger("RND")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

class RNDPredictor(nn.Module):
    def __init__(self, bert_dim=768, returns_dim=1000, hidden_dim=512, output_dim=8, n_proto=6):
        super().__init__()
        self.n_proto = n_proto
        self.hidden_dim = hidden_dim
        
        self.text_encoder = nn.Sequential(
            nn.Linear(bert_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.prototypes = nn.Parameter(torch.randn(n_proto, hidden_dim))
        
        self.returns_encoder = nn.Sequential(
            nn.Linear(returns_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.prototypes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, text_feat, returns):
        text_feat = self.text_encoder(text_feat)
        
        text_norm = nn.functional.normalize(text_feat, dim=-1)
        proto_norm = nn.functional.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(text_norm, proto_norm.t())
        idx = sim.argmax(dim=-1)
        selected_proto = self.prototypes[idx]
        
        returns_feat = self.returns_encoder(returns)
        fused = torch.cat([text_feat, selected_proto, returns_feat], dim=-1)
        return self.fusion(fused)

class Metrics:
    TARGET_NAMES = ['mean', 'variance', 'skewness', 'excess_kurtosis', 'var_5', 'var_10', 'es_5', 'es_10']
    
    @staticmethod
    def mae(pred, target):
        return torch.abs(pred - target).mean(dim=0)
    
    @staticmethod
    def qs(pred, target, alpha=0.05):
        diff = target - pred
        qs = (alpha - (target < pred).float()) * diff
        return torch.nanmean(qs).item()
    
    @staticmethod
    def compute_all(pred, target):
        results = {}
        mae_vals = Metrics.mae(pred, target)
        for i, name in enumerate(Metrics.TARGET_NAMES[:4]):
            results[f'MAE_{name}'] = mae_vals[i].item()
        results['QS@0.05'] = Metrics.qs(pred, target, 0.05)
        results['QS@0.10'] = Metrics.qs(pred, target, 0.10)
        return results

@torch.no_grad()
def extract_bert_features(loader, bert, device):
    bert.eval()
    all_feats, all_returns, all_targets = [], [], []
    for batch in tqdm(loader, desc='Extracting BERT features'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        out = bert(input_ids=input_ids, attention_mask=attention_mask)
        all_feats.append(out.last_hidden_state[:, 0, :].cpu())
        all_returns.append(batch['returns'])
        all_targets.append(batch['targets'])
    return torch.cat(all_feats), torch.cat(all_returns), torch.cat(all_targets)

def load_or_extract_features(loader, bert, device, cache_path, logger):
    if os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        data = torch.load(cache_path)
        return data['feats'], data['returns'], data['targets']
    else:
        feats, returns, targets = extract_bert_features(loader, bert, device)
        torch.save({'feats': feats, 'returns': returns, 'targets': targets}, cache_path)
        logger.info(f"Saved features to {cache_path}")
        return feats, returns, targets

def train_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{epochs} [Train]')
    for text_feat, returns, targets in pbar:
        text_feat, returns, targets = text_feat.to(device), returns.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = model(text_feat, returns)
        loss = nn.L1Loss()(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, device, desc='Eval'):
    model.eval()
    all_pred, all_target = [], []
    total_loss = 0
    pbar = tqdm(loader, desc=desc)
    for text_feat, returns, targets in pbar:
        text_feat, returns, targets = text_feat.to(device), returns.to(device), targets.to(device)
        pred = model(text_feat, returns)
        loss = nn.L1Loss()(pred, targets)
        total_loss += loss.item()
        all_pred.append(pred.cpu())
        all_target.append(targets.cpu())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    metrics = Metrics.compute_all(all_pred, all_target)
    metrics['loss'] = total_loss / len(loader)
    return metrics

def main():
    set_seed(42)
    logger = setup_logger()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    train_cache = os.path.join(CACHE_DIR, "train_features.pt")
    val_cache = os.path.join(CACHE_DIR, "val_features.pt")
    test_cache = os.path.join(CACHE_DIR, "test_features.pt")
    
    all_cached = all(os.path.exists(p) for p in [train_cache, val_cache, test_cache])
    
    if all_cached:
        logger.info("Loading all features from cache...")
        train_data = torch.load(train_cache)
        val_data = torch.load(val_cache)
        test_data = torch.load(test_cache)
        train_feats, train_returns, train_targets = train_data['feats'], train_data['returns'], train_data['targets']
        val_feats, val_returns, val_targets = val_data['feats'], val_data['returns'], val_data['targets']
        test_feats, test_returns, test_targets = test_data['feats'], test_data['returns'], test_data['targets']
    else:
        bert_path = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert = AutoModel.from_pretrained(bert_path).to(device)
        bert.eval()
        
        train_loader, val_loader, test_loader, _ = create_dataloaders(
            data_path="./data/rnd_prediction_dataset.json",
            tokenizer=tokenizer,
            batch_size=64,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            returns_dim=1000,
            normalize_targets=True,
            shuffle_train=False,
            sort_by_date=True,
            num_workers=4
        )
        
        logger.info("Extracting BERT features...")
        train_feats, train_returns, train_targets = load_or_extract_features(train_loader, bert, device, train_cache, logger)
        val_feats, val_returns, val_targets = load_or_extract_features(val_loader, bert, device, val_cache, logger)
        test_feats, test_returns, test_targets = load_or_extract_features(test_loader, bert, device, test_cache, logger)
        
        del bert
        torch.cuda.empty_cache()
    
    logger.info(f"Train: {train_feats.shape[0]}, Val: {val_feats.shape[0]}, Test: {test_feats.shape[0]}")
    
    batch_size = 128
    train_loader = DataLoader(TensorDataset(train_feats, train_returns, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_returns, val_targets), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_feats, test_returns, test_targets), batch_size=batch_size, shuffle=False)
    
    model = RNDPredictor(bert_dim=768, returns_dim=1000, hidden_dim=512, output_dim=8).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    epochs = 100
    
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs)
        val_metrics = evaluate(model, val_loader, device, desc=f'Epoch {epoch}/{epochs} [Val]')
        
        log_msg = (f"Epoch {epoch:02d}/{epochs} | "
                   f"Train: {train_loss:.4f} | Val: {val_metrics['loss']:.4f} | "
                   f"MAE[μ={val_metrics['MAE_mean']:.4f}, σ²={val_metrics['MAE_variance']:.4f}, "
                   f"γ={val_metrics['MAE_skewness']:.4f}, κ={val_metrics['MAE_excess_kurtosis']:.4f}] | "
                   f"QS@5%={val_metrics['QS@0.05']:.4f}, @10%={val_metrics['QS@0.10']:.4f}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save(model.state_dict(), 'best_model.pt')
            log_msg += " *"
            no_improve = 0
        else:
            no_improve += 1
        
        logger.info(log_msg)
        scheduler.step()
        
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("=" * 60)
    logger.info("Test Evaluation")
    logger.info("=" * 60)
    
    model.load_state_dict(torch.load('best_model.pt'))
    test_metrics = evaluate(model, test_loader, device, desc='Test')
    
    logger.info(f"Test Loss: {test_metrics['loss']:.2f}")
    logger.info(f"MAE - μ: {test_metrics['MAE_mean']:.2f}, var: {test_metrics['MAE_variance']:.2f}, "
                f"skew: {test_metrics['MAE_skewness']:.2f}, kurt: {test_metrics['MAE_excess_kurtosis']:.2f}")
    logger.info(f"QS@0.05: {test_metrics['QS@0.05']:.4f}, QS@0.10: {test_metrics['QS@0.10']:.4f}")

if __name__ == "__main__":
    main()
