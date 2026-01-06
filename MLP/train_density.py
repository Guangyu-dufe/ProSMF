import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from scipy import interpolate
import random
import json
from pathlib import Path

CACHE_DIR = "./cache"
LOG_DIR = "./logs"
DATA_PATH = "./data/dataset.json"

DENSITY_DIM = 500

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logger():
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"train_density_{timestamp}.log")
    
    logger = logging.getLogger("RND_D")
    logger.setLevel(logging.INFO)
    logger.handlers = []
    
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def compute_metrics_from_density(returns, density):
    returns = np.asarray(returns)
    density = np.asarray(density)
    
    density = np.clip(density, 0, None)
    total = np.trapz(density, returns)
    if total > 1e-10:
        density = density / total
    else:
        return {k: np.nan for k in ['mean', 'variance', 'skewness', 'excess_kurtosis', 
                                     'var_5', 'var_10', 'es_5', 'es_10']}
    
    mean = np.trapz(returns * density, returns)
    variance = np.trapz((returns - mean)**2 * density, returns)
    std = np.sqrt(max(variance, 0))
    
    if std > 1e-10:
        skewness = np.trapz(((returns - mean) / std)**3 * density, returns)
        kurtosis = np.trapz(((returns - mean) / std)**4 * density, returns)
        excess_kurtosis = kurtosis - 3.0
    else:
        skewness = 0.0
        excess_kurtosis = 0.0
    
    dx = np.diff(returns)
    cdf = np.zeros_like(returns)
    cdf[1:] = np.cumsum(0.5 * (density[:-1] + density[1:]) * dx)
    if cdf[-1] > 1e-10:
        cdf = cdf / cdf[-1]
    
    def compute_var(alpha):
        try:
            f_inv = interpolate.interp1d(cdf, returns, kind='linear', 
                                         bounds_error=False, fill_value='extrapolate')
            return float(f_inv(alpha))
        except:
            idx = np.searchsorted(cdf, alpha)
            return returns[idx] if idx < len(returns) else returns[-1]
    
    def compute_es(alpha):
        var = compute_var(alpha)
        mask = returns <= var
        if not np.any(mask):
            return returns[0]
        tail_returns = returns[mask]
        tail_density = density[mask]
        tail_prob = np.trapz(tail_density, tail_returns)
        if tail_prob > 1e-10:
            return np.trapz(tail_returns * tail_density, tail_returns) / tail_prob
        return var
    
    return {
        'mean': mean,
        'variance': variance,
        'skewness': skewness,
        'excess_kurtosis': excess_kurtosis,
        'var_5': compute_var(0.05),
        'var_10': compute_var(0.10),
        'es_5': compute_es(0.05),
        'es_10': compute_es(0.10)
    }

def compute_batch_metrics(pred_density, returns_grid):
    batch_size = pred_density.shape[0]
    metrics = {k: [] for k in ['mean', 'variance', 'skewness', 'excess_kurtosis',
                                'var_5', 'var_10', 'es_5', 'es_10']}
    
    pred_np = pred_density.cpu().numpy()
    returns_np = returns_grid.cpu().numpy()
    
    for i in range(batch_size):
        m = compute_metrics_from_density(returns_np[i], pred_np[i])
        for k, v in m.items():
            metrics[k].append(v)
    
    return {k: np.array(v) for k, v in metrics.items()}

class RNDDensityPredictor(nn.Module):
    def __init__(self, bert_dim=768, returns_dim=1000, hidden_dim=512, output_dim=DENSITY_DIM):
        super().__init__()
        self.output_dim = output_dim
        
        self.text_encoder = nn.Sequential(
            nn.Linear(bert_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        
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
            nn.Linear(hidden_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, text_feat, returns):
        text_feat = self.text_encoder(text_feat)
        returns_feat = self.returns_encoder(returns)
        fused = torch.cat([text_feat, returns_feat], dim=-1)
        density = self.fusion(fused)
        density = torch.softmax(density, dim=-1)
        return density

class DensityLoss(nn.Module):
    def __init__(self, lambda_smooth=0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth
    
    def forward(self, pred, target, returns_grid):
        pred = pred + 1e-10
        target = target + 1e-10
        
        pred_norm = pred / pred.sum(dim=-1, keepdim=True)
        target_norm = target / target.sum(dim=-1, keepdim=True)
        
        mae_loss = torch.abs(pred_norm - target_norm).mean()
        
        diff = pred_norm[:, 1:] - pred_norm[:, :-1]
        smooth_loss = (diff ** 2).mean()
        
        return mae_loss + self.lambda_smooth * smooth_loss

class Metrics:
    METRIC_NAMES = ['mean', 'variance', 'skewness', 'excess_kurtosis', 'var_5', 'var_10', 'es_5', 'es_10']
    
    @staticmethod
    def qs(pred, target, alpha=0.05):
        diff = target - pred
        qs = (alpha - (target < pred).astype(float)) * diff
        return np.nanmean(qs)
    
    @staticmethod
    def compute_all(pred_density, target_density, returns_grid, target_metrics):
        results = {}
        
        pred_density_np = pred_density.cpu().numpy() if torch.is_tensor(pred_density) else pred_density
        target_density_np = target_density.cpu().numpy() if torch.is_tensor(target_density) else target_density
        returns_np = returns_grid.cpu().numpy() if torch.is_tensor(returns_grid) else returns_grid
        
        density_mae = np.abs(pred_density_np - target_density_np).mean()
        results['density_mae'] = density_mae
        
        pred_metrics = compute_batch_metrics(
            torch.tensor(pred_density_np), torch.tensor(returns_np)
        )
        
        for name in Metrics.METRIC_NAMES[:4]:
            pred_m = pred_metrics[name]
            target_m = target_metrics[name]
            results[f'MAE_{name}'] = np.abs(pred_m - target_m).mean()
        
        all_pred = np.stack([pred_metrics[n] for n in Metrics.METRIC_NAMES], axis=1)
        all_target = np.stack([target_metrics[n] for n in Metrics.METRIC_NAMES], axis=1)
        
        results['QS@0.05'] = Metrics.qs(all_pred, all_target, 0.05)
        results['QS@0.10'] = Metrics.qs(all_pred, all_target, 0.10)
        
        return results

def extract_news_text(news_list):
    texts = []
    for news_item in news_list:
        title = news_item.get('title', '').strip()
        body = news_item.get('body', '').strip()
        if title:
            texts.append(title)
        if body:
            texts.append(body)
    return ' '.join(texts) if texts else ''

def resample_density(returns, density, n_points=DENSITY_DIM):
    returns = np.asarray(returns)
    density = np.asarray(density)
    
    new_returns = np.linspace(returns.min(), returns.max(), n_points)
    
    try:
        f = interpolate.interp1d(returns, density, kind='linear', bounds_error=False, fill_value=0)
        new_density = f(new_returns)
    except:
        indices = np.linspace(0, len(returns) - 1, n_points, dtype=int)
        new_returns = returns[indices]
        new_density = density[indices]
    
    new_density = np.clip(new_density, 0, None)
    total = np.trapz(new_density, new_returns)
    if total > 1e-10:
        new_density = new_density / total
    
    return new_returns.astype(np.float32), new_density.astype(np.float32)

def load_raw_data(data_dir, density_dim=DENSITY_DIM):
    data_path = Path(data_dir)
    date_folders = sorted([d for d in data_path.iterdir() if d.is_dir()])
    
    samples = []
    
    for date_folder in tqdm(date_folders, desc="Loading raw data"):
        json_files = list(date_folder.glob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                before_data = data.get('before', {})
                after_data = data.get('after', {})
                
                if 'rnd' not in before_data or 'rnd' not in after_data:
                    continue
                
                news_list = before_data.get('news', [])
                news_text = extract_news_text(news_list)
                if not news_text:
                    continue
                
                before_returns = np.array(before_data['rnd']['returns'])
                before_density = np.array(before_data['rnd']['density'])
                after_returns = np.array(after_data['rnd']['returns'])
                after_density = np.array(after_data['rnd']['density'])
                
                before_returns_r, before_density_r = resample_density(before_returns, before_density, density_dim)
                after_returns_r, after_density_r = resample_density(after_returns, after_density, density_dim)
                
                after_metrics = compute_metrics_from_density(after_returns_r, after_density_r)
                if any(np.isnan(v) for v in after_metrics.values()):
                    continue
                
                samples.append({
                    'news_text': news_text,
                    'before_returns': before_returns_r,
                    'before_density': before_density_r,
                    'after_returns': after_returns_r,
                    'after_density': after_density_r,
                    'after_metrics': after_metrics,
                    'before_date': before_data.get('date', ''),
                    'ticker': data.get('ticker', ''),
                })
            except Exception as e:
                continue
    
    print(f"Loaded {len(samples)} samples")
    return samples

from torch.utils.data import Dataset

class DensityDataset(Dataset):
    def __init__(self, samples, tokenizer=None, max_length=512):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        result = {
            'before_returns': torch.tensor(s['before_returns']),
            'before_density': torch.tensor(s['before_density']),
            'after_returns': torch.tensor(s['after_returns']),
            'after_density': torch.tensor(s['after_density']),
            'after_metrics': torch.tensor([s['after_metrics'][k] for k in Metrics.METRIC_NAMES], dtype=torch.float32),
            'news_text': s['news_text'],
        }
        
        if self.tokenizer:
            enc = self.tokenizer(s['news_text'], max_length=self.max_length, 
                                 padding='max_length', truncation=True, return_tensors='pt')
            result['input_ids'] = enc['input_ids'].squeeze(0)
            result['attention_mask'] = enc['attention_mask'].squeeze(0)
        
        return result

def create_dataloaders(samples, tokenizer, batch_size=64, train_ratio=0.8, val_ratio=0.1):
    samples = sorted(samples, key=lambda x: x.get('before_date', ''))
    
    dataset = DensityDataset(samples, tokenizer)
    
    n = len(dataset)
    train_n = int(train_ratio * n)
    val_n = int(val_ratio * n)
    
    train_indices = list(range(0, train_n))
    val_indices = list(range(train_n, train_n + val_n))
    test_indices = list(range(train_n + val_n, n))
    
    from torch.utils.data import Subset
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    def collate(batch):
        result = {
            'before_returns': torch.stack([b['before_returns'] for b in batch]),
            'before_density': torch.stack([b['before_density'] for b in batch]),
            'after_returns': torch.stack([b['after_returns'] for b in batch]),
            'after_density': torch.stack([b['after_density'] for b in batch]),
            'after_metrics': torch.stack([b['after_metrics'] for b in batch]),
        }
        if 'input_ids' in batch[0]:
            result['input_ids'] = torch.stack([b['input_ids'] for b in batch])
            result['attention_mask'] = torch.stack([b['attention_mask'] for b in batch])
        return result
    
    train_loader = DataLoader(train_set, batch_size, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size, shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
    
    return train_loader, val_loader, test_loader

@torch.no_grad()
def extract_bert_features(loader, bert, device):
    bert.eval()
    all_feats = []
    all_before_returns, all_before_density = [], []
    all_after_returns, all_after_density = [], []
    all_metrics = []
    
    for batch in tqdm(loader, desc='Extracting BERT features'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        out = bert(input_ids=input_ids, attention_mask=attention_mask)
        all_feats.append(out.last_hidden_state[:, 0, :].cpu())
        all_before_returns.append(batch['before_returns'])
        all_before_density.append(batch['before_density'])
        all_after_returns.append(batch['after_returns'])
        all_after_density.append(batch['after_density'])
        all_metrics.append(batch['after_metrics'])
    
    return (torch.cat(all_feats), torch.cat(all_before_returns), torch.cat(all_before_density),
            torch.cat(all_after_returns), torch.cat(all_after_density), torch.cat(all_metrics))

def load_or_extract_features(loader, bert, device, cache_path, logger):
    if os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        data = torch.load(cache_path)
        return (data['feats'], data['before_returns'], data['before_density'],
                data['after_returns'], data['after_density'], data['metrics'])
    else:
        feats, br, bd, ar, ad, m = extract_bert_features(loader, bert, device)
        torch.save({'feats': feats, 'before_returns': br, 'before_density': bd,
                    'after_returns': ar, 'after_density': ad, 'metrics': m}, cache_path)
        logger.info(f"Saved features to {cache_path}")
        return feats, br, bd, ar, ad, m

def train_epoch(model, loader, optimizer, criterion, device, epoch, epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{epochs} [Train]')
    for text_feat, before_returns, before_density, after_returns, after_density, metrics in pbar:
        text_feat = text_feat.to(device)
        before_density = before_density.to(device)
        after_density = after_density.to(device)
        after_returns = after_returns.to(device)
        
        optimizer.zero_grad()
        pred = model(text_feat, before_density)
        loss = criterion(pred, after_density, after_returns)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    return total_loss / len(loader)

@torch.no_grad()
def evaluate(model, loader, criterion, device, desc='Eval'):
    model.eval()
    all_pred, all_after_density, all_after_returns, all_metrics = [], [], [], []
    total_loss = 0
    pbar = tqdm(loader, desc=desc)
    
    for text_feat, before_returns, before_density, after_returns, after_density, metrics in pbar:
        text_feat = text_feat.to(device)
        before_density = before_density.to(device)
        after_density = after_density.to(device)
        after_returns = after_returns.to(device)
        
        pred = model(text_feat, before_density)
        loss = criterion(pred, after_density, after_returns)
        total_loss += loss.item()
        
        all_pred.append(pred.cpu())
        all_after_density.append(after_density.cpu())
        all_after_returns.append(after_returns.cpu())
        all_metrics.append(metrics)
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    all_pred = torch.cat(all_pred, dim=0)
    all_after_density = torch.cat(all_after_density, dim=0)
    all_after_returns = torch.cat(all_after_returns, dim=0)
    all_metrics = torch.cat(all_metrics, dim=0).numpy()
    
    target_metrics_dict = {
        name: all_metrics[:, i] for i, name in enumerate(Metrics.METRIC_NAMES)
    }
    
    results = Metrics.compute_all(all_pred, all_after_density, all_after_returns, target_metrics_dict)
    results['loss'] = total_loss / len(loader)
    return results

def main():
    set_seed(42)
    logger = setup_logger()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    train_cache = os.path.join(CACHE_DIR, "train_density_features.pt")
    val_cache = os.path.join(CACHE_DIR, "val_density_features.pt")
    test_cache = os.path.join(CACHE_DIR, "test_density_features.pt")
    
    all_cached = all(os.path.exists(p) for p in [train_cache, val_cache, test_cache])
    
    if all_cached:
        logger.info("Loading all features from cache...")
        train_data = torch.load(train_cache)
        val_data = torch.load(val_cache)
        test_data = torch.load(test_cache)
        
        train_feats = train_data['feats']
        train_br, train_bd = train_data['before_returns'], train_data['before_density']
        train_ar, train_ad = train_data['after_returns'], train_data['after_density']
        train_metrics = train_data['metrics']
        
        val_feats = val_data['feats']
        val_br, val_bd = val_data['before_returns'], val_data['before_density']
        val_ar, val_ad = val_data['after_returns'], val_data['after_density']
        val_metrics = val_data['metrics']
        
        test_feats = test_data['feats']
        test_br, test_bd = test_data['before_returns'], test_data['before_density']
        test_ar, test_ad = test_data['after_returns'], test_data['after_density']
        test_metrics = test_data['metrics']
    else:
        samples = load_raw_data(RAW_DATA_DIR, density_dim=DENSITY_DIM)
        
        bert_path = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(bert_path)
        bert = AutoModel.from_pretrained(bert_path).to(device)
        bert.eval()
        
        train_loader, val_loader, test_loader = create_dataloaders(samples, tokenizer, batch_size=64)
        
        logger.info("Extracting BERT features...")
        train_feats, train_br, train_bd, train_ar, train_ad, train_metrics = \
            load_or_extract_features(train_loader, bert, device, train_cache, logger)
        val_feats, val_br, val_bd, val_ar, val_ad, val_metrics = \
            load_or_extract_features(val_loader, bert, device, val_cache, logger)
        test_feats, test_br, test_bd, test_ar, test_ad, test_metrics = \
            load_or_extract_features(test_loader, bert, device, test_cache, logger)
        
        del bert
        torch.cuda.empty_cache()
    
    logger.info(f"Train: {train_feats.shape[0]}, Val: {val_feats.shape[0]}, Test: {test_feats.shape[0]}")
    
    batch_size = 128
    train_loader = DataLoader(
        TensorDataset(train_feats, train_br, train_bd, train_ar, train_ad, train_metrics),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_feats, val_br, val_bd, val_ar, val_ad, val_metrics),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(test_feats, test_br, test_bd, test_ar, test_ad, test_metrics),
        batch_size=batch_size, shuffle=False
    )
    
    model = RNDDensityPredictor(bert_dim=768, returns_dim=DENSITY_DIM, hidden_dim=512, output_dim=DENSITY_DIM).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    
    criterion = DensityLoss(lambda_smooth=0.01)
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    epochs = 1
    
    logger.info("=" * 70)
    logger.info("Starting training (Density prediction)...")
    logger.info("=" * 70)
    
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)
        val_results = evaluate(model, val_loader, criterion, device, desc=f'Epoch {epoch}/{epochs} [Val]')
        
        log_msg = (f"Epoch {epoch:02d}/{epochs} | "
                   f"Train: {train_loss:.4f} | Val: {val_results['loss']:.4f} | "
                   f"D_MAE: {val_results['density_mae']:.6f} | "
                   f"MAE[μ={val_results['MAE_mean']:.4f}, σ²={val_results['MAE_variance']:.4f}, "
                   f"γ={val_results['MAE_skewness']:.4f}, κ={val_results['MAE_excess_kurtosis']:.4f}] | "
                   f"QS@5%={val_results['QS@0.05']:.4f}, @10%={val_results['QS@0.10']:.4f}")
        
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save(model.state_dict(), 'best_model_density.pt')
            log_msg += " *"
            no_improve = 0
        else:
            no_improve += 1
        
        logger.info(log_msg)
        scheduler.step()
        
        if no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    logger.info("=" * 70)
    logger.info("Test Evaluation")
    logger.info("=" * 70)
    
    model.load_state_dict(torch.load('best_model_density.pt'))
    test_results = evaluate(model, test_loader, criterion, device, desc='Test')
    
    logger.info(f"Test Loss: {test_results['loss']:.6f}")
    logger.info(f"Density MAE: {test_results['density_mae']:.6f}")
    logger.info(f"Reconstructed Metrics MAE - μ: {test_results['MAE_mean']:.6f}, var: {test_results['MAE_variance']:.6f}, "
                f"skew: {test_results['MAE_skewness']:.6f}, kurt: {test_results['MAE_excess_kurtosis']:.6f}")
    logger.info(f"QS@0.05: {test_results['QS@0.05']:.4f}, QS@0.10: {test_results['QS@0.10']:.4f}")

if __name__ == "__main__":
    main()

