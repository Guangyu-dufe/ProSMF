
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

# Optional transformers import
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    AutoTokenizer = None
    HAS_TRANSFORMERS = False
    print("Warning: transformers not available. Tokenizer features disabled.")


class RNDPredictionDataset(Dataset):
    
    # Target column names
    TARGET_COLS = [
        'delta_mean',
        'delta_variance', 
        'delta_skewness',
        'delta_excess_kurtosis',
        'delta_var_5',
        'delta_var_10',
        'delta_es_5',
        'delta_es_10'
    ]
    
    def __init__(
        self,
        data_path: str,
        tokenizer: Optional[Any] = None,
        max_length: int = 512,
        density_dim: int = 1000,
        normalize_targets: bool = False,
        target_stats: Optional[Dict] = None
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.density_dim = density_dim
        self.normalize_targets = normalize_targets
        self.target_stats = target_stats
        
        # Load data
        self.samples = self._load_data()
        
        # Compute target statistics if needed
        if self.normalize_targets and self.target_stats is None:
            self.target_stats = self._compute_target_stats()
    
    def _load_data(self) -> List[Dict]:
        samples = []
        
        if self.data_path.is_file() and self.data_path.suffix == '.json':
            # Load from processed JSON file
            with open(self.data_path, 'r') as f:
                samples = json.load(f)
        elif self.data_path.is_dir():
            # Load from directory of raw JSON files
            samples = self._load_from_directory()
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")
        
        print(f"Loaded {len(samples)} samples")
        return samples
    
    def _load_from_directory(self) -> List[Dict]:
        from process_rnd_prediction_data import process_single_file
        
        samples = []
        date_folders = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        for date_folder in date_folders:
            json_files = list(date_folder.glob("*.json"))
            for json_file in json_files:
                result = process_single_file(str(json_file))
                if result:
                    samples.append(result)
        
        return samples
    
    def _compute_target_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for col in self.TARGET_COLS:
            values = [s[col] for s in self.samples if col in s]
            stats[col] = {
                'mean': np.mean(values),
                'std': np.std(values)
            }
        return stats
    
    def _normalize_target(self, value: float, col: str) -> float:
        if self.target_stats and col in self.target_stats:
            mean = self.target_stats[col]['mean']
            std = self.target_stats[col]['std']
            if std > 1e-10:
                return (value - mean) / std
        return value
    
    def _denormalize_target(self, value: float, col: str) -> float:
        if self.target_stats and col in self.target_stats:
            mean = self.target_stats[col]['mean']
            std = self.target_stats[col]['std']
            return value * std + mean
        return value
    
    def _pad_or_truncate_density(self, returns: List[float]) -> np.ndarray:
        returns = np.array(returns, dtype=np.float32)
        
        if len(returns) >= self.density_dim:
            # Truncate - sample evenly
            indices = np.linspace(0, len(returns) - 1, self.density_dim, dtype=int)
            returns = returns[indices]
        else:
            # Pad with zeros
            padded = np.zeros(self.density_dim, dtype=np.float32)
            padded[:len(returns)] = returns
            returns = padded
        
        return returns
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        # Get news text
        news_text = sample.get('news_text', '')
        
        # Get returns
        returns = sample.get('before_density', [])
        returns = self._pad_or_truncate_density(returns)
        
        # Get targets
        targets = []
        for col in self.TARGET_COLS:
            value = sample.get(col, 0.0)
            if self.normalize_targets:
                value = self._normalize_target(value, col)
            targets.append(value)
        targets = np.array(targets, dtype=np.float32)
        
        result = {
            'density': torch.tensor(returns, dtype=torch.float32),
            'targets': torch.tensor(targets, dtype=torch.float32),
            'news_text': news_text,
        }
        
        # Tokenize if tokenizer provided
        if self.tokenizer:
            encoding = self.tokenizer(
                news_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['input_ids'] = encoding['input_ids'].squeeze(0)
            result['attention_mask'] = encoding['attention_mask'].squeeze(0)
        
        # Add metadata
        result['ticker'] = sample.get('ticker', '')
        result['before_date'] = sample.get('before_date', '')
        result['after_date'] = sample.get('after_date', '')
        
        return result
    
    def get_target_stats(self) -> Dict:
        return self.target_stats


class RNDCollator:
    
    def __init__(self, tokenizer: Optional[Any] = None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Stack numerical tensors
        returns = torch.stack([item['returns'] for item in batch])
        targets = torch.stack([item['targets'] for item in batch])
        
        result = {
            'density': returns,
            'targets': targets,
        }
        
        # Handle tokenized inputs
        if 'input_ids' in batch[0]:
            result['input_ids'] = torch.stack([item['input_ids'] for item in batch])
            result['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])
        elif self.tokenizer:
            # Tokenize batch
            texts = [item['news_text'] for item in batch]
            encoding = self.tokenizer(
                texts,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['input_ids'] = encoding['input_ids']
            result['attention_mask'] = encoding['attention_mask']
        
        # Keep raw text for reference
        result['news_text'] = [item['news_text'] for item in batch]
        result['ticker'] = [item['ticker'] for item in batch]
        result['before_date'] = [item['before_date'] for item in batch]
        result['after_date'] = [item['after_date'] for item in batch]
        
        return result


def create_dataloaders(
    data_path: str,
    tokenizer: Optional[Any] = None,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_length: int = 512,
    density_dim: int = 1000,
    normalize_targets: bool = True,
    num_workers: int = 4,
    shuffle_train: bool = False,
    sort_by_date: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    # Load full dataset
    full_dataset = RNDPredictionDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        density_dim=density_dim,
        normalize_targets=normalize_targets
    )
    
    # Sort samples by date if needed
    if sort_by_date:
        full_dataset.samples = sorted(
            full_dataset.samples, 
            key=lambda x: x.get('before_date', '')
        )
        print("Data sorted by date (chronological order)")
    
    # Recompute target stats after sorting (should be the same)
    if normalize_targets:
        full_dataset.target_stats = full_dataset._compute_target_stats()
    
    # Get target stats
    target_stats = full_dataset.get_target_stats()
    
    # Split dataset chronologically (not randomly)
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Use Subset for chronological split
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)
    
    # Print split info with date ranges
    if len(full_dataset.samples) > 0:
        train_start = full_dataset.samples[train_indices[0]].get('before_date', 'N/A') if train_indices else 'N/A'
        train_end = full_dataset.samples[train_indices[-1]].get('before_date', 'N/A') if train_indices else 'N/A'
        val_start = full_dataset.samples[val_indices[0]].get('before_date', 'N/A') if val_indices else 'N/A'
        val_end = full_dataset.samples[val_indices[-1]].get('before_date', 'N/A') if val_indices else 'N/A'
        test_start = full_dataset.samples[test_indices[0]].get('before_date', 'N/A') if test_indices else 'N/A'
        test_end = full_dataset.samples[test_indices[-1]].get('before_date', 'N/A') if test_indices else 'N/A'
        
        print(f"Dataset split (chronological):")
        print(f"  Train: {train_size} samples ({train_start} ~ {train_end})")
        print(f"  Val:   {val_size} samples ({val_start} ~ {val_end})")
        print(f"  Test:  {test_size} samples ({test_start} ~ {test_end})")
    
    # Create collator
    collator = RNDCollator(tokenizer=tokenizer, max_length=max_length)
    
    # Create dataloaders (no shuffle for time series data by default)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,  # Default False for time series
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, target_stats


def create_dataloader_from_csv(
    csv_path: str,
    tokenizer: Optional[Any] = None,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    class CSVDataset(Dataset):
        def __init__(self, dataframe, tokenizer, max_length):
            self.df = dataframe
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.target_cols = [
                'delta_mean', 'delta_variance', 'delta_skewness',
                'delta_excess_kurtosis', 'delta_var_5', 'delta_var_10',
                'delta_es_5', 'delta_es_10'
            ]
        
        def __len__(self):
            return len(self.df)
        
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            
            news_text = str(row['news_text'])
            targets = torch.tensor(
                [row[col] for col in self.target_cols],
                dtype=torch.float32
            )
            
            result = {
                'news_text': news_text,
                'targets': targets,
                'ticker': row['ticker'],
                'before_date': row['before_date']
            }
            
            if self.tokenizer:
                encoding = self.tokenizer(
                    news_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                result['input_ids'] = encoding['input_ids'].squeeze(0)
                result['attention_mask'] = encoding['attention_mask'].squeeze(0)
            
            return result
    
    dataset = CSVDataset(df, tokenizer, max_length)
    
    def collate_fn(batch):
        result = {
            'targets': torch.stack([item['targets'] for item in batch]),
            'news_text': [item['news_text'] for item in batch],
            'ticker': [item['ticker'] for item in batch],
            'before_date': [item['before_date'] for item in batch]
        }
        if 'input_ids' in batch[0]:
            result['input_ids'] = torch.stack([item['input_ids'] for item in batch])
            result['attention_mask'] = torch.stack([item['attention_mask'] for item in batch])
        return result
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test the dataloader
    print("Testing RND DataLoader...")
    
    # Path to processed data
    data_path = "./data/rnd_prediction_dataset.json"
    
    # Test without tokenizer first
    print("\n1. Testing dataset without tokenizer...")
    dataset = RNDPredictionDataset(
        data_path=data_path,
        tokenizer=None,
        max_length=512,
        density_dim=1000,
        normalize_targets=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Returns shape: {sample['returns'].shape}")
    print(f"Targets shape: {sample['targets'].shape}")
    print(f"Targets: {sample['targets']}")
    print(f"News text (first 100 chars): {sample['news_text'][:100]}...")
    print(f"Ticker: {sample['ticker']}")
    
    # Test with dataloader
    print("\n2. Testing DataLoader...")
    collator = RNDCollator()
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator
    )
    
    batch = next(iter(loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Returns batch shape: {batch['returns'].shape}")
    print(f"Targets batch shape: {batch['targets'].shape}")
    
    # Test with tokenizer
    print("\n3. Testing with tokenizer...")
    if HAS_TRANSFORMERS:
        try:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            train_loader, val_loader, test_loader, target_stats = create_dataloaders(
                data_path=data_path,
                tokenizer=tokenizer,
                batch_size=8,
                train_ratio=0.8,
                val_ratio=0.1,
                test_ratio=0.1,
                normalize_targets=True,
                num_workers=0  # Use 0 for testing
            )
            
            batch = next(iter(train_loader))
            print(f"\nBatch with tokenizer:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  returns shape: {batch['returns'].shape}")
            print(f"  targets shape: {batch['targets'].shape}")
            
            print(f"\nTarget statistics (for denormalization):")
            for col, stats in target_stats.items():
                print(f"  {col}: mean={stats['mean']:.6f}, std={stats['std']:.6f}")
            
        except Exception as e:
            print(f"Tokenizer test skipped: {e}")
    else:
        print("Skipped (transformers not available)")
    
    print("\n✓ DataLoader test completed!")

