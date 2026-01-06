
QUANTILE_KEYS = ['0.1', '0.15', '0.2', '0.25', '0.3', '0.35', '0.4', '0.45', '0.5',
                 '0.55', '0.6', '0.65', '0.7', '0.75', '0.8', '0.85', '0.9']

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
from scipy import interpolate


def compute_moments_from_density(returns: np.ndarray, density: np.ndarray) -> Dict[str, float]:
    returns = np.asarray(returns)
    density = np.asarray(density)
    
    # Handle negative density values by clipping
    density = np.clip(density, 0, None)
    
    # Normalize density
    total = np.trapz(density, returns)
    if total > 1e-10:
        density = density / total
    else:
        return {'mean': np.nan, 'variance': np.nan, 'std': np.nan, 
                'skewness': np.nan, 'kurtosis': np.nan, 'excess_kurtosis': np.nan}
    
    # Mean (first moment)
    mean = np.trapz(returns * density, returns)
    
    # Variance (second central moment)
    variance = np.trapz((returns - mean)**2 * density, returns)
    std = np.sqrt(max(variance, 0))
    
    # Skewness (third standardized moment)
    if std > 1e-10:
        skewness = np.trapz(((returns - mean) / std)**3 * density, returns)
    else:
        skewness = 0.0
    
    # Kurtosis (fourth standardized moment)
    if std > 1e-10:
        kurtosis = np.trapz(((returns - mean) / std)**4 * density, returns)
        excess_kurtosis = kurtosis - 3.0
    else:
        kurtosis = 0.0
        excess_kurtosis = 0.0
    
    return {
        'mean': mean,
        'variance': variance,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'excess_kurtosis': excess_kurtosis
    }


def compute_var_from_density(returns: np.ndarray, density: np.ndarray, alpha: float = 0.05) -> float:
    returns = np.asarray(returns)
    density = np.asarray(density)
    
    # Handle negative density
    density = np.clip(density, 0, None)
    
    # Compute CDF by numerical integration
    dx = np.diff(returns)
    cdf = np.zeros_like(returns)
    cdf[1:] = np.cumsum(0.5 * (density[:-1] + density[1:]) * dx)
    
    # Normalize CDF
    if cdf[-1] > 1e-10:
        cdf = cdf / cdf[-1]
    else:
        return np.nan
    
    # Find the return value where CDF = alpha using interpolation
    try:
        f_inv = interpolate.interp1d(cdf, returns, kind='linear', 
                                      bounds_error=False, fill_value='extrapolate')
        var = float(f_inv(alpha))
    except:
        idx = np.searchsorted(cdf, alpha)
        var = returns[idx] if idx < len(returns) else returns[-1]
    
    return var


def compute_es_from_density(returns: np.ndarray, density: np.ndarray, alpha: float = 0.05) -> float:
    returns = np.asarray(returns)
    density = np.asarray(density)
    
    # Handle negative density
    density = np.clip(density, 0, None)
    
    # First compute VaR
    var = compute_var_from_density(returns, density, alpha)
    
    if np.isnan(var):
        return np.nan
    
    # Find indices below VaR
    mask = returns <= var
    
    if not np.any(mask):
        return returns[0]
    
    tail_returns = returns[mask]
    tail_density = density[mask]
    
    # Compute conditional expectation
    tail_prob = np.trapz(tail_density, tail_returns)
    
    if tail_prob > 1e-10:
        es = np.trapz(tail_returns * tail_density, tail_returns) / tail_prob
    else:
        es = var
    
    return es


def reconstruct_density_from_quantiles(quantile_probs: np.ndarray, quantile_values: np.ndarray, 
                                       n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    q_probs = np.asarray(quantile_probs)
    q_vals = np.asarray(quantile_values)
    
    sorted_idx = np.argsort(q_probs)
    q_probs = q_probs[sorted_idx]
    q_vals = q_vals[sorted_idx]
    
    eps = 0.02
    if q_probs[0] > eps:
        q_probs = np.concatenate([[eps], q_probs])
        q_vals = np.concatenate([[q_vals[0] - (q_vals[1] - q_vals[0])], q_vals])
    if q_probs[-1] < 1 - eps:
        q_probs = np.concatenate([q_probs, [1 - eps]])
        q_vals = np.concatenate([q_vals, [q_vals[-1] + (q_vals[-1] - q_vals[-2])]])
    
    try:
        f_quantile = interpolate.PchipInterpolator(q_probs, q_vals, extrapolate=True)
    except:
        f_quantile = interpolate.interp1d(q_probs, q_vals, kind='linear', 
                                          fill_value='extrapolate')
    
    cdf_probs = np.linspace(0.01, 0.99, n_points)
    returns = f_quantile(cdf_probs)
    
    dr = np.diff(returns)
    dr = np.where(np.abs(dr) < 1e-10, 1e-10, dr)
    dp = np.diff(cdf_probs)
    
    density_mid = dp / dr
    density_mid = np.clip(density_mid, 0, None)
    
    density = np.zeros(n_points)
    density[:-1] = density_mid
    density[-1] = density_mid[-1]
    
    total = np.trapz(density, returns)
    if total > 1e-10:
        density = density / total
    
    return returns, density


def extract_quantiles(quantiles_dict: Dict) -> Tuple[np.ndarray, np.ndarray]:
    probs = []
    vals = []
    for key in QUANTILE_KEYS:
        if key in quantiles_dict:
            probs.append(float(key))
            vals.append(quantiles_dict[key])
    return np.array(probs), np.array(vals)


def compute_all_metrics(returns: np.ndarray, density: np.ndarray) -> Dict[str, float]:
    # Compute moments
    moments = compute_moments_from_density(returns, density)
    
    # Compute risk metrics
    var_5 = compute_var_from_density(returns, density, alpha=0.05)
    var_10 = compute_var_from_density(returns, density, alpha=0.10)
    es_5 = compute_es_from_density(returns, density, alpha=0.05)
    es_10 = compute_es_from_density(returns, density, alpha=0.10)
    
    return {
        'mean': moments['mean'],
        'variance': moments['variance'],
        'skewness': moments['skewness'],
        'excess_kurtosis': moments['excess_kurtosis'],
        'var_5': var_5,
        'var_10': var_10,
        'es_5': es_5,
        'es_10': es_10
    }


def compute_delta_metrics(before_metrics: Dict[str, float], 
                          after_metrics: Dict[str, float]) -> Dict[str, float]:
    delta = {}
    for key in ['mean', 'variance', 'skewness', 'excess_kurtosis', 'var_5', 'var_10', 'es_5', 'es_10']:
        delta[f'delta_{key}'] = after_metrics[key] - before_metrics[key]
    return delta


def extract_news_text(news_list: List[Dict]) -> str:
    texts = []
    for news_item in news_list:
        title = news_item.get('title', '').strip()
        body = news_item.get('body', '').strip()
        if title:
            texts.append(title)
        if body:
            texts.append(body)
    return ' '.join(texts) if texts else ''


def process_single_file(file_path: str, include_quantiles: bool = True) -> Optional[Dict]:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        ticker = data.get('ticker', '')
        expire_date = data.get('expire_date', '')
        
        before_data = data.get('before', {})
        after_data = data.get('after', {})
        
        # Check if we have required data
        if 'rnd' not in before_data or 'rnd' not in after_data:
            return None
        
        # Extract before RND data
        before_returns = np.array(before_data['rnd']['returns'])
        before_density = np.array(before_data['rnd']['density'])
        
        # Extract after RND data
        after_returns = np.array(after_data['rnd']['returns'])
        after_density = np.array(after_data['rnd']['density'])
        
        # Extract news
        news_list = before_data.get('news', [])
        news_text = extract_news_text(news_list)
        news_count = before_data.get('news_count', len(news_list))
        
        # Skip if no news
        if not news_text or news_count == 0:
            return None
        
        # Compute metrics
        before_metrics = compute_all_metrics(before_returns, before_density)
        after_metrics = compute_all_metrics(after_returns, after_density)
        
        # Check for NaN values
        if any(np.isnan(v) for v in before_metrics.values()) or \
           any(np.isnan(v) for v in after_metrics.values()):
            return None
        
        # Compute deltas
        delta_metrics = compute_delta_metrics(before_metrics, after_metrics)
        
        # Build result
        result = {
            'ticker': ticker,
            'expire_date': expire_date,
            'before_date': before_data.get('date', ''),
            'after_date': after_data.get('date', ''),
            'before_tau': before_data.get('tau', 0),
            'after_tau': after_data.get('tau', 0),
            'news_count': news_count,
            'news_text': news_text,
            # Before metrics (for reference)
            'before_mean': before_metrics['mean'],
            'before_variance': before_metrics['variance'],
            'before_skewness': before_metrics['skewness'],
            'before_excess_kurtosis': before_metrics['excess_kurtosis'],
            'before_var_5': before_metrics['var_5'],
            'before_var_10': before_metrics['var_10'],
            'before_es_5': before_metrics['es_5'],
            'before_es_10': before_metrics['es_10'],
            # After metrics (for reference)
            'after_mean': after_metrics['mean'],
            'after_variance': after_metrics['variance'],
            'after_skewness': after_metrics['skewness'],
            'after_excess_kurtosis': after_metrics['excess_kurtosis'],
            'after_var_5': after_metrics['var_5'],
            'after_var_10': after_metrics['var_10'],
            'after_es_5': after_metrics['es_5'],
            'after_es_10': after_metrics['es_10'],
            # Delta metrics (targets for prediction)
            **delta_metrics,
            # Store density for use as features
            'before_density': before_density.tolist(),
        }
        
        # Extract quantiles if available
        if include_quantiles:
            before_quantiles = before_data.get('quantiles', {})
            after_quantiles = after_data.get('quantiles', {})
            
            if before_quantiles and after_quantiles:
                # Store before quantiles
                for key in QUANTILE_KEYS:
                    if key in before_quantiles:
                        result[f'before_q{key}'] = before_quantiles[key]
                    if key in after_quantiles:
                        result[f'after_q{key}'] = after_quantiles[key]
                
                # Store as arrays for easier processing
                _, before_q_vals = extract_quantiles(before_quantiles)
                _, after_q_vals = extract_quantiles(after_quantiles)
                result['before_quantiles'] = before_q_vals.tolist()
                result['after_quantiles'] = after_q_vals.tolist()
        
        return result
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def process_all_data(data_dir: str, output_dir: str = None):
    data_path = Path(data_dir)
    
    if output_dir is None:
        output_dir = './data'
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all date folders
    date_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    all_records = []
    processed_count = 0
    skipped_count = 0
    
    print(f"Processing {len(date_folders)} date folders...")
    
    for date_folder in tqdm(date_folders, desc="Processing folders"):
        json_files = list(date_folder.glob("*.json"))
        
        for json_file in json_files:
            result = process_single_file(str(json_file))
            if result:
                result['source_file'] = str(json_file.relative_to(data_path))
                all_records.append(result)
                processed_count += 1
            else:
                skipped_count += 1
    
    print(f"\nProcessed: {processed_count}, Skipped: {skipped_count}")
    
    if not all_records:
        print("No records processed!")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(all_records)
    
    # Sort by date
    df = df.sort_values(['before_date', 'ticker']).reset_index(drop=True)
    
    # Save full dataset
    csv_path = output_path / 'rnd_prediction_dataset.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV to: {csv_path}")
    
    # Save a version without the full density array (smaller file)
    df_compact = df.drop(columns=['before_density'])
    compact_csv_path = output_path / 'rnd_prediction_dataset_compact.csv'
    df_compact.to_csv(compact_csv_path, index=False)
    print(f"Saved compact CSV to: {compact_csv_path}")
    
    # Save as JSON for models that need the density array
    json_path = output_path / 'rnd_prediction_dataset.json'
    df.to_json(json_path, orient='records', indent=2)
    print(f"Saved JSON to: {json_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Total samples: {len(df)}")
    print(f"Unique tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['before_date'].min()} to {df['before_date'].max()}")
    
    print("\nTarget variable statistics (Delta values):")
    target_cols = ['delta_mean', 'delta_variance', 'delta_skewness', 
                   'delta_excess_kurtosis', 'delta_var_5', 'delta_var_10', 
                   'delta_es_5', 'delta_es_10']
    for col in target_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.6f}")
        print(f"  Std:  {df[col].std():.6f}")
        print(f"  Min:  {df[col].min():.6f}")
        print(f"  Max:  {df[col].max():.6f}")
    
    # Save a summary statistics file
    summary_stats = df[target_cols].describe()
    summary_path = output_path / 'target_statistics.csv'
    summary_stats.to_csv(summary_path)
    print(f"\nSaved target statistics to: {summary_path}")
    
    return df


def create_train_format(df: pd.DataFrame, output_path: Path):
    train_data = []
    
    instruction = """Based on the news content and the current return distribution, predict the changes in the following Risk-Neutral Density (RND) statistics:
1. Δmean: Change in mean return
2. Δvariance: Change in variance
3. Δskewness: Change in skewness
4. Δexcess_kurtosis: Change in excess kurtosis
5. Δvar_5: Change in 5% Value at Risk
6. Δvar_10: Change in 10% Value at Risk
7. Δes_5: Change in 5% Expected Shortfall
8. Δes_10: Change in 10% Expected Shortfall"""

    for _, row in df.iterrows():
        # Create input with news and return statistics
        returns_summary = (
            f"Current RND: mean={row['before_mean']:.6f}, "
            f"variance={row['before_variance']:.6f}, "
            f"skewness={row['before_skewness']:.4f}, "
            f"excess_kurtosis={row['before_excess_kurtosis']:.4f}, "
            f"VaR_5={row['before_var_5']:.6f}, "
            f"VaR_10={row['before_var_10']:.6f}, "
            f"ES_5={row['before_es_5']:.6f}, "
            f"ES_10={row['before_es_10']:.6f}"
        )
        
        input_text = f"Ticker: {row['ticker']}\nNews: {row['news_text']}\n{returns_summary}"
        
        # Create output
        output_text = (
            f"delta_mean: {row['delta_mean']:.8f}, "
            f"delta_variance: {row['delta_variance']:.8f}, "
            f"delta_skewness: {row['delta_skewness']:.6f}, "
            f"delta_excess_kurtosis: {row['delta_excess_kurtosis']:.6f}, "
            f"delta_var_5: {row['delta_var_5']:.8f}, "
            f"delta_var_10: {row['delta_var_10']:.8f}, "
            f"delta_es_5: {row['delta_es_5']:.8f}, "
            f"delta_es_10: {row['delta_es_10']:.8f}"
        )
        
        train_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text,
            "ticker": row['ticker'],
            "before_date": row['before_date'],
            "after_date": row['after_date']
        })
    
    # Save training data
    train_path = output_path / 'rnd_prediction_train.json'
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Saved training format data to: {train_path}")
    
    return train_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process RND data for prediction')
    parser.add_argument('--data_dir', type=str, 
                        default='./raw_data',
                        help='Path to Final_data_v2 directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: processed_prediction_data)')
    parser.add_argument('--create_train', action='store_true',
                        help='Create training format data for LLM fine-tuning')
    
    args = parser.parse_args()
    
    # Process all data
    df = process_all_data(args.data_dir, args.output_dir)
    
    if df is not None and args.create_train:
        output_path = Path(args.output_dir) if args.output_dir else \
                      Path('./data')
        create_train_format(df, output_path)

