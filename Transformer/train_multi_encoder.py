import os
import random
import logging
import argparse
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import json

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

HF_CACHE = "./cache"
CACHE_DIR = "./cache"
LOG_DIR = "./logs"
DATA_PATH = "./data/dataset.json"

ENCODER_CONFIGS = {
    'bert': {'path': 'bert-base-uncased', 'dim': 768},
    'roberta': {'path': 'roberta-base', 'dim': 768},
    'finbert': {'path': './cache', 'dim': 768},
}

def setup_logger(encoder_name):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"train_{encoder_name}_{timestamp}.log")
    
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
    def __init__(self, bert_dim=768, density_dim=1000, hidden_dim=512, output_dim=8, n_proto=6, n_heads=8, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_proto = n_proto
        
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.density_proj = nn.Sequential(
            nn.Linear(density_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.prototypes = nn.Parameter(torch.randn(n_proto, hidden_dim) * 0.02)
        self.proto_norm = nn.LayerNorm(hidden_dim)
        
        self.pos_embed = nn.Parameter(torch.randn(1, 3, hidden_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads, dim_feedforward=hidden_dim*4,
            dropout=0.1, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.prototypes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, text_feat, density):
        B = text_feat.size(0)
        text_emb = self.text_proj(text_feat)
        
        text_norm = nn.functional.normalize(text_emb, dim=-1)
        proto_norm = nn.functional.normalize(self.prototypes, dim=-1)
        sim = torch.matmul(text_norm, proto_norm.t())
        idx = sim.argmax(dim=-1)
        selected_proto = self.proto_norm(self.prototypes[idx])
        
        text_emb = text_emb.unsqueeze(1)
        density_emb = self.density_proj(density).unsqueeze(1)
        proto_emb = selected_proto.unsqueeze(1)
        
        tokens = torch.cat([text_emb, density_emb, proto_emb], dim=1)
        tokens = tokens + self.pos_embed
        
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        
        out = self.transformer(tokens)
        cls_out = out[:, 0]
        
        return self.output_head(cls_out)

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

class MetricsDataset(Dataset):
    def __init__(self, data_path, tokenizer=None, max_length=512, density_dim=1000):
        with open(data_path, 'r') as f:
            self.samples = json.load(f)
        
        self.samples = [s for s in self.samples 
                       if all(s.get(f'after_{name}') is not None for name in Metrics.TARGET_NAMES)]
        print(f"Loaded {len(self.samples)} samples with all metrics")
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.density_dim = density_dim
    
    def __len__(self):
        return len(self.samples)
    
    def _pad_density(self, density):
        density = np.array(density, dtype=np.float32)
        if len(density) >= self.density_dim:
            indices = np.linspace(0, len(density) - 1, self.density_dim, dtype=int)
            density = density[indices]
        else:
            padded = np.zeros(self.density_dim, dtype=np.float32)
            padded[:len(density)] = density
            density = padded
        return density
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        
        news_text = s.get('news_text', '')
        density = self._pad_density(s.get('before_density', []))
        targets = np.array([s[f'after_{name}'] for name in Metrics.TARGET_NAMES], dtype=np.float32)
        
        result = {
            'density': torch.tensor(density),
            'targets': torch.tensor(targets),
            'news_text': news_text,
        }
        
        if self.tokenizer:
            enc = self.tokenizer(news_text, max_length=self.max_length, 
                                 padding='max_length', truncation=True, return_tensors='pt')
            result['input_ids'] = enc['input_ids'].squeeze(0)
            result['attention_mask'] = enc['attention_mask'].squeeze(0)
        
        return result

def create_dataloaders(data_path, tokenizer, batch_size=64, train_ratio=0.8, val_ratio=0.1):
    dataset = MetricsDataset(data_path, tokenizer)
    
    dataset.samples = sorted(dataset.samples, key=lambda x: x.get('before_date', ''))
    
    n = len(dataset)
    train_n = int(train_ratio * n)
    val_n = int(val_ratio * n)
    
    train_indices = list(range(0, train_n))
    val_indices = list(range(train_n, train_n + val_n))
    test_indices = list(range(train_n + val_n, n))
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)
    
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    
    def collate(batch):
        result = {
            'density': torch.stack([b['density'] for b in batch]),
            'targets': torch.stack([b['targets'] for b in batch]),
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
def extract_features(loader, encoder, device):
    encoder.eval()
    all_feats, all_density, all_targets = [], [], []
    for batch in tqdm(loader, desc='Extracting features'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        out = encoder(input_ids=input_ids, attention_mask=attention_mask)
        all_feats.append(out.last_hidden_state[:, 0, :].cpu())
        all_density.append(batch['density'])
        all_targets.append(batch['targets'])
    return torch.cat(all_feats), torch.cat(all_density), torch.cat(all_targets)

def load_or_extract_features(loader, encoder, device, cache_path, logger):
    if os.path.exists(cache_path):
        logger.info(f"Loading cached features from {cache_path}")
        data = torch.load(cache_path)
        return data['feats'], data['density'], data['targets']
    else:
        feats, density, targets = extract_features(loader, encoder, device)
        torch.save({'feats': feats, 'density': density, 'targets': targets}, cache_path)
        logger.info(f"Saved features to {cache_path}")
        return feats, density, targets

def train_epoch(model, loader, optimizer, device, epoch, epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch}/{epochs} [Train]')
    for text_feat, density, targets in pbar:
        text_feat, density, targets = text_feat.to(device), density.to(device), targets.to(device)
        optimizer.zero_grad()
        pred = model(text_feat, density)
        loss = nn.L1Loss()(pred, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
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
    for text_feat, density, targets in pbar:
        text_feat, density, targets = text_feat.to(device), density.to(device), targets.to(device)
        pred = model(text_feat, density)
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

def run_experiment(encoder_name, seed, device, logger):
    config = ENCODER_CONFIGS[encoder_name]
    encoder_path = config['path']
    encoder_dim = config['dim']
    
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    train_cache = os.path.join(CACHE_DIR, f"train_{encoder_name}_features.pt")
    val_cache = os.path.join(CACHE_DIR, f"val_{encoder_name}_features.pt")
    test_cache = os.path.join(CACHE_DIR, f"test_{encoder_name}_features.pt")
    
    all_cached = all(os.path.exists(p) for p in [train_cache, val_cache, test_cache])
    
    if all_cached:
        logger.info("Loading all features from cache...")
        train_data = torch.load(train_cache)
        val_data = torch.load(val_cache)
        test_data = torch.load(test_cache)
        train_feats, train_density, train_targets = train_data['feats'], train_data['density'], train_data['targets']
        val_feats, val_density, val_targets = val_data['feats'], val_data['density'], val_data['targets']
        test_feats, test_density, test_targets = test_data['feats'], test_data['density'], test_data['targets']
    else:
        logger.info(f"Loading {encoder_name} from {encoder_path}...")
        tokenizer = AutoTokenizer.from_pretrained(encoder_path, cache_dir=HF_CACHE)
        encoder = AutoModel.from_pretrained(encoder_path, cache_dir=HF_CACHE).to(device)
        encoder.eval()
        
        train_loader, val_loader, test_loader = create_dataloaders(DATA_PATH, tokenizer, batch_size=64)
        
        logger.info(f"Extracting {encoder_name} features...")
        train_feats, train_density, train_targets = load_or_extract_features(train_loader, encoder, device, train_cache, logger)
        val_feats, val_density, val_targets = load_or_extract_features(val_loader, encoder, device, val_cache, logger)
        test_feats, test_density, test_targets = load_or_extract_features(test_loader, encoder, device, test_cache, logger)
        
        del encoder
        torch.cuda.empty_cache()
    
    logger.info(f"Train: {train_feats.shape[0]}, Val: {val_feats.shape[0]}, Test: {test_feats.shape[0]}")
    
    batch_size = 128
    train_loader = DataLoader(TensorDataset(train_feats, train_density, train_targets), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_density, val_targets), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_feats, test_density, test_targets), batch_size=batch_size, shuffle=False)
    
    model = RNDPredictor(bert_dim=encoder_dim, density_dim=1000, hidden_dim=512, output_dim=8).to(device)
    
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    
    best_val_loss = float('inf')
    patience = 20
    no_improve = 0
    epochs = 100
    
    model_save_path = f'best_model_{encoder_name}.pt'
    
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
            torch.save(model.state_dict(), model_save_path)
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
    
    model.load_state_dict(torch.load(model_save_path))
    test_metrics = evaluate(model, test_loader, device, desc='Test')
    
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"MAE - μ: {test_metrics['MAE_mean']:.4f}, var: {test_metrics['MAE_variance']:.4f}, "
                f"skew: {test_metrics['MAE_skewness']:.4f}, kurt: {test_metrics['MAE_excess_kurtosis']:.4f}")
    logger.info(f"QS@0.05: {test_metrics['QS@0.05']:.4f}, QS@0.10: {test_metrics['QS@0.10']:.4f}")
    
    return test_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--encoder', type=str, default='bert', choices=['bert', 'roberta', 'finbert', 'all'])
    args = parser.parse_args()
    
    seed = args.seed if args.seed is not None else 42
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.encoder == 'all':
        encoders = ['bert', 'roberta', 'finbert']
    else:
        encoders = [args.encoder]
    
    all_results = {}
    
    for encoder_name in encoders:
        set_seed(seed)
        logger = setup_logger(encoder_name)
        logger.info(f"Running experiment with {encoder_name.upper()}")
        logger.info(f"Random seed: {seed}")
        logger.info(f"Using device: {device}")
        
        test_metrics = run_experiment(encoder_name, seed, device, logger)
        all_results[encoder_name] = test_metrics
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS ({encoder_name.upper()}):")
        print(f"MAE_mean: {test_metrics['MAE_mean']:.4f}")
        print(f"MAE_variance: {test_metrics['MAE_variance']:.4f}")
        print(f"MAE_skewness: {test_metrics['MAE_skewness']:.4f}")
        print(f"MAE_excess_kurtosis: {test_metrics['MAE_excess_kurtosis']:.4f}")
        print(f"QS@0.05: {test_metrics['QS@0.05']:.4f}")
        print(f"QS@0.10: {test_metrics['QS@0.10']:.4f}")
        print("="*60)
    
    if len(all_results) > 1:
        results_path = os.path.join(CACHE_DIR, "all_encoder_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_path}")
        
        print("\n" + "="*60)
        print("LaTeX Table:")
        print("="*60)
        generate_latex_table(all_results)

def generate_latex_table(results):
    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparison of Different Text Encoders for RND Prediction}
\label{tab:encoder_comparison}
\begin{tabular}{l|cccc|cc}
\toprule
\textbf{Encoder} & \textbf{MAE}$_\mu$ & \textbf{MAE}$_{\sigma^2}$ & \textbf{MAE}$_\gamma$ & \textbf{MAE}$_\kappa$ & \textbf{QS@5\%} & \textbf{QS@10\%} \\
\midrule
\end{tabular}
\end{table}"""
    
    print(latex)
    
    latex_path = os.path.join(CACHE_DIR, "encoder_comparison.tex")
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"\nLaTeX table saved to {latex_path}")

if __name__ == "__main__":
    main()

