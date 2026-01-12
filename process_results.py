import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from baseline_hypers import hyperparameters as baseline_hyperparameters


def check_reinmax_cv_status(results_dir='./results'):
    """Check status of reinmax_cv files - missing, incomplete, or completed"""
    results_path = Path(results_dir)
    if not results_path.exists():
        return
    
    categorical_dim_options = [8, 4, 8, 16, 64, 10]
    latent_dim_options = [4, 24, 16, 12, 8, 30]
    seeds = range(10)
    tau2_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    eta_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    missing = []
    incomplete = []
    completed = []
    
    for cat_dim, lat_dim in zip(categorical_dim_options, latent_dim_options):
        # Get baseline reinmax hyperparameters
        key = ('reinmax', cat_dim, lat_dim)
        if key not in baseline_hyperparameters:
            continue
        
        baseline_lr, baseline_temp, baseline_optimizer = baseline_hyperparameters[key]
        
        for seed in seeds:
            configs_for_seed = []
            for tau2 in tau2_options:
                for eta in eta_options:
                    fname = f"results_seed{seed}_reinmax_cv_cat{cat_dim}_lat{lat_dim}_opt{baseline_optimizer}_lr{baseline_lr}_temp{baseline_temp}_eta{eta}_tau2{tau2}.txt"
                    fpath = results_path / fname
                    
                    if not fpath.exists():
                        configs_for_seed.append((eta, tau2, 'MISSING'))
                    else:
                        try:
                            data = np.loadtxt(fpath, delimiter=',')
                            if len(data) < 160:
                                configs_for_seed.append((eta, tau2, f'INCOMPLETE({len(data)}/160)'))
                            else:
                                configs_for_seed.append((eta, tau2, 'OK'))
                        except:
                            configs_for_seed.append((eta, tau2, 'ERROR'))
            
            # Summarize for this seed
            statuses = [s[2] for s in configs_for_seed]
            ok_count = statuses.count('OK')
            missing_count = sum(1 for s in statuses if 'MISSING' in s)
            incomplete_count = sum(1 for s in statuses if 'INCOMPLETE' in s)
            error_count = statuses.count('ERROR')
            
            if ok_count == len(configs_for_seed):
                completed.append((cat_dim, lat_dim, seed))
            elif missing_count > 0 or incomplete_count > 0:
                incomplete.append(f"Cat{cat_dim}_Lat{lat_dim}_Seed{seed}: {ok_count}/{len(configs_for_seed)} OK, {missing_count} MISSING, {incomplete_count} INCOMPLETE, {error_count} ERROR")
    
    print("\n" + "="*120)
    print("REINMAX_CV FILE STATUS")
    print("="*120)
    print(f"\nCompleted configurations (all 10 seeds with all eta/tau2 combos): {len(completed)}")
    
    if incomplete:
        print(f"\nIncomplete/Missing ({len(incomplete)} seed-configs):")
        for item in incomplete[:30]:  # Show first 30
            print(f"  {item}")
        if len(incomplete) > 30:
            print(f"  ... and {len(incomplete) - 30} more")
    else:
        print("\nAll reinmax_cv files completed!")


def load_results(results_dir='./results'):
    """Load all result files organized by configuration"""
    results = defaultdict(lambda: defaultdict(dict))
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} not found")
        return results
    
    # Define parameter options
    optimizer_options = ["Adam", "RAdam"]
    learning_rate_options = [0.0007, 0.0005, 0.0003]
    temperature_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    categorical_dim_options = [8, 4, 8, 16, 64, 10]
    latent_dim_options = [4, 24, 16, 12, 8, 30]
    # methods = ["reinmax_v2", "reinmax_v3"]
    methods = ["reinmax_v3"]
    seeds = range(10)
    
    # Define tau2 and eta options for reinmax_cv
    tau2_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    eta_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    # Loop through all parameter combinations for reinmax_v3
    for cat_dim, lat_dim in zip(categorical_dim_options, latent_dim_options):
        for method in methods:
            for optimizer in optimizer_options:
                for lr in learning_rate_options:
                    for temp in temperature_options:
                        for seed in seeds:
                            # Construct filename
                            filename = (
                                results_path / 
                                f"results_seed{seed}_{method}_cat{cat_dim}_lat{lat_dim}_"
                                f"opt{optimizer}_lr{lr}_temp{temp}.txt"
                            )
                            
                            if filename.exists():
                                try:
                                    # Load metrics from file
                                    metrics = np.loadtxt(filename, delimiter=',')
                                    
                                    # Skip files that don't have all 160 epochs
                                    if len(metrics) < 160:
                                        print(f"Skipping {filename}: only {len(metrics)} epochs (expected 160)")
                                        continue
                                    
                                    # Get last epoch: [train_loss, train_bce, train_kld, test_loss, test_bce, test_kld, ..., sample_std, ...]
                                    last_epoch = metrics[-1]
                                    train_loss = last_epoch[0]
                                    test_loss = last_epoch[3]
                                    sample_std = last_epoch[-2]  # Second last column
                                    
                                    # Create config key
                                    config_key = (cat_dim, lat_dim, method, optimizer, lr, temp)
                                    
                                    # Store results by seed
                                    results[config_key][seed] = {
                                        'train_loss': train_loss,
                                        'test_loss': test_loss,
                                        'sample_std': sample_std,
                                    }
                                    
                                except Exception as e:
                                    print(f"Error loading {filename}: {e}")
                                    continue
                            else:
                                print(f"File not found: {filename}")
    
    # Handle reinmax_cv separately - find best tau2/eta for each config using baseline reinmax hyperparameters
    # Average across seeds first, then pick the best (eta, tau2) combination
    reinmax_cv_best_hypers = {}  # Track best hyperparameters for each config
    
    for cat_dim, lat_dim in zip(categorical_dim_options, latent_dim_options):
        # Get baseline reinmax hyperparameters for this configuration
        key = ('reinmax', cat_dim, lat_dim)
        if key not in baseline_hyperparameters:
            continue
            
        baseline_lr, baseline_temp, baseline_optimizer = baseline_hyperparameters[key]
        
        # Collect results for all (eta, tau2) combinations across all seeds
        combo_results = {}  # (eta, tau2) -> {seed: {train_loss, test_loss, sample_std}}
        
        for tau2 in tau2_options:
            for eta in eta_options:
                combo_results[(eta, tau2)] = {}
                for seed in seeds:
                    fname = f"results_seed{seed}_reinmax_cv_cat{cat_dim}_lat{lat_dim}_opt{baseline_optimizer}_lr{baseline_lr}_temp{baseline_temp}_eta{eta}_tau2{tau2}.txt"
                    fpath = results_path / fname
                    
                    if fpath.exists():
                        try:
                            metrics = np.loadtxt(fpath, delimiter=',')
                            
                            # Skip files that don't have all 160 epochs
                            if len(metrics) < 160:
                                continue
                            
                            # Get last epoch
                            last_epoch = metrics[-1]
                            combo_results[(eta, tau2)][seed] = {
                                'train_loss': last_epoch[0],
                                'test_loss': last_epoch[3],
                                'sample_std': last_epoch[-2],
                            }
                        except Exception as e:
                            pass
                    else:
                        print(f"File not found: {fpath}")
        
        # Find the best (eta, tau2) based on average train loss across seeds
        best_avg_loss = float('inf')
        best_combo = None
        
        for (eta, tau2), seed_results in combo_results.items():
            # if len(seed_results) >= 5:  # Require at least 5 seeds for a valid average
            avg_train_loss = np.mean([r['train_loss'] for r in seed_results.values()])
            if avg_train_loss < best_avg_loss:
                best_avg_loss = avg_train_loss
                best_combo = (eta, tau2)
        
        # Store results using the best (eta, tau2) for all seeds
        if best_combo is not None:
            config_key = (cat_dim, lat_dim, 'reinmax_cv', baseline_optimizer, baseline_lr, baseline_temp)
            for seed, result in combo_results[best_combo].items():
                results[config_key][seed] = result
            
            reinmax_cv_best_hypers[(cat_dim, lat_dim)] = {
                'best_eta': best_combo[0],
                'best_tau2': best_combo[1],
                'avg_train_loss': best_avg_loss,
                'n_seeds': len(combo_results[best_combo]),
            }
    
    # Print summary of best hyperparameters
    if reinmax_cv_best_hypers:
        print_reinmax_cv_hyperparameters(reinmax_cv_best_hypers)
        
    return results

def print_reinmax_cv_hyperparameters(reinmax_cv_best_hypers):
    """Print summary of best eta and tau2 for reinmax_cv"""
    print("\n" + "="*80)
    print("REINMAX-CV BEST HYPERPARAMETERS (selected by avg train loss across seeds)")
    print("="*80)
    print(f"{'Config':<12} {'Best η':<10} {'Best τ²':<10} {'Avg Train Loss':<18} {'N Seeds'}")
    print("-"*80)
    
    configs = [(8, 4), (4, 24), (8, 16), (16, 12), (64, 8), (10, 30)]
    
    for cat_dim, lat_dim in configs:
        if (cat_dim, lat_dim) not in reinmax_cv_best_hypers:
            print(f"{cat_dim}×{lat_dim:<9} {'N/A':<10} {'N/A':<10} {'N/A':<18} N/A")
            continue
        
        info = reinmax_cv_best_hypers[(cat_dim, lat_dim)]
        config_label = f"{cat_dim}×{lat_dim}"
        
        print(f"{config_label:<12} {info['best_eta']:<10} {info['best_tau2']:<10} {info['avg_train_loss']:<18.4f} {info['n_seeds']}")
    
    print("="*80 + "\n")


def compute_statistics(results):
    """Compute mean and std error for each configuration"""
    stats = {}
    
    for config_key, seed_results in results.items():
        seeds = sorted(seed_results.keys())
        
        if len(seeds) < 10:
            print(f"Warning: Configuration {config_key} has only {len(seeds)} seeds")
        
        train_losses = [seed_results[s]['train_loss'] for s in seeds]
        test_losses = [seed_results[s]['test_loss'] for s in seeds]
        sample_stds = [seed_results[s].get('sample_std', np.nan) for s in seeds]
        
        train_loss_mean = np.mean(train_losses)
        train_loss_std = np.std(train_losses) / np.sqrt(len(seeds))
        
        test_loss_mean = np.mean(test_losses)
        test_loss_std = np.std(test_losses) / np.sqrt(len(seeds))
        
        # Filter out NaN values for sample_std
        valid_sample_stds = [s for s in sample_stds if not np.isnan(s)]
        if len(valid_sample_stds) > 0:
            sample_std_mean = np.mean(valid_sample_stds)
            sample_std_std = np.std(valid_sample_stds) / np.sqrt(len(valid_sample_stds))
        else:
            sample_std_mean = np.nan
            sample_std_std = np.nan
        
        stats[config_key] = {
            'train_loss_mean': train_loss_mean,
            'train_loss_std': train_loss_std,
            'test_loss_mean': test_loss_mean,
            'test_loss_std': test_loss_std,
            'sample_std_mean': sample_std_mean,
            'sample_std_std': sample_std_std,
            'n_seeds': len(seeds),
        }
    
    return stats


def load_baseline_results(results_dir='./results'):
    """Load baseline results using tuned hyperparameters from baseline_hypers.py"""
    results = defaultdict(lambda: defaultdict(dict))
    
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory {results_dir} not found")
        return results
    
    seeds = range(10)
    
    # Load baseline results using tuned hyperparameters
    for (method, cat_dim, lat_dim), (lr, temp, optimizer) in baseline_hyperparameters.items():
        for seed in seeds:
            # Construct filename
            filename = (
                results_path / 
                f"results_seed{seed}_{method}_cat{cat_dim}_lat{lat_dim}_"
                f"opt{optimizer}_lr{lr}_temp{temp}.txt"
            )
            
            if filename.exists():
                try:
                    # Load metrics from file
                    metrics = np.loadtxt(filename, delimiter=',')
                    
                    # Skip files that don't have all 160 epochs
                    if len(metrics) < 160:
                        print(f"Skipping {filename}: only {len(metrics)} epochs (expected 160)")
                        continue
                    
                    # Get last epoch
                    last_epoch = metrics[-1]
                    train_loss = last_epoch[0]
                    test_loss = last_epoch[3]
                    sample_std = last_epoch[-2]  # Second last column
                    
                    # Create config key
                    config_key = (cat_dim, lat_dim, method, optimizer, lr, temp)
                    
                    # Store results by seed
                    results[config_key][seed] = {
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'sample_std': sample_std,
                    }
                    
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    continue
            else:
                print(f"File not found: {filename}")
    
    return results


def print_best_results(stats):
    """Print best results grouped by configuration and method"""
    
    # Group by (cat_dim, lat_dim, method)
    grouped = defaultdict(list)
    
    for config_key, stat in stats.items():
        cat_dim, lat_dim, method, optimizer, lr, temp = config_key
        group_key = (cat_dim, lat_dim, method)
        grouped[group_key].append((config_key, stat))
    
    # Print results grouped by cat_dim, lat_dim
    configs = sorted(set((k[0], k[1]) for k in grouped.keys()))
    
    for cat_dim, lat_dim in configs:
        print(f"\n{'='*100}")
        print(f"Categorical Dim: {cat_dim}, Latent Dim: {lat_dim}")
        print(f"{'='*100}")
        
        # for method in ['reinmax_v2', 'reinmax_v3']:
        for method in ['reinmax_v3']:
            group_key = (cat_dim, lat_dim, method)
            
            if group_key not in grouped:
                print(f"\n{method}: No results found")
                continue
            
            # Find best result (lowest test loss)
            # best_config, best_stat = min(
            #     grouped[group_key],
            #     key=lambda x: x[1]['test_loss_mean']
            # )
            best_config, best_stat = min(
                grouped[group_key],
                key=lambda x: x[1]['train_loss_mean']
            )
            
            cat_dim_best, lat_dim_best, method_best, optimizer_best, lr_best, temp_best = best_config
            
            print(f"\n{method}:")
            print(f"  Best hyperparameters:")
            print(f"    Optimizer: {optimizer_best}")
            print(f"    Learning Rate: {lr_best}")
            print(f"    Temperature: {temp_best}")
            print(f"  Results (n={best_stat['n_seeds']} seeds):")
            print(f"    Train Loss: {best_stat['train_loss_mean']:.6f} ± {best_stat['train_loss_std']:.6f}")
            print(f"    Test Loss:  {best_stat['test_loss_mean']:.6f} ± {best_stat['test_loss_std']:.6f}")


def print_sample_std_comparison(stats, baseline_stats):
    """Print comparison of sample standard deviation between methods"""
    
    # Get all unique (cat_dim, lat_dim) configurations
    configs = set()
    for config_key in stats.keys():
        cat_dim, lat_dim = config_key[0], config_key[1]
        configs.add((cat_dim, lat_dim))
    for config_key in baseline_stats.keys():
        cat_dim, lat_dim = config_key[0], config_key[1]
        configs.add((cat_dim, lat_dim))
    
    configs = sorted(configs)
    
    baseline_methods = ['gumbel', 'rao_gumbel', 'gst-1.0', 'st', 'reinmax']
    # new_methods = ['reinmax_v2', 'reinmax_v3']
    new_methods = ['reinmax_v3', 'reinmax_cv']
    
    for cat_dim, lat_dim in configs:
        print(f"\n{'='*100}")
        print(f"Sample Standard Deviation - Categorical Dim: {cat_dim}, Latent Dim: {lat_dim}")
        print(f"{'='*100}")
        print(f"{'Method':<15} {'Optimizer':<10} {'LR':<10} {'Temp':<8} {'Sample Std':<30} {'N':<5}")
        print(f"{'-'*100}")
        
        # Print baseline results
        for method in baseline_methods:
            found = False
            for config_key, stat in baseline_stats.items():
                c_dim, l_dim, m, opt, lr, temp = config_key
                if c_dim == cat_dim and l_dim == lat_dim and m == method:
                    if not np.isnan(stat.get('sample_std_mean', np.nan)):
                        std_str = f"{stat['sample_std_mean']:.6f} ± {stat['sample_std_std']:.6f}"
                    else:
                        std_str = "N/A"
                    print(f"{method:<15} {opt:<10} {lr:<10} {temp:<8} {std_str:<30} {stat['n_seeds']:<5}")
                    found = True
                    break
            if not found:
                print(f"{method:<15} {'--':<10} {'--':<10} {'--':<8} {'N/A':<30} {'--':<5}")
        
        print(f"{'-'*100}")
        
        # Print new method results (best hyperparameters based on train loss)
        for method in new_methods:
            best_stat = None
            best_config = None
            for config_key, stat in stats.items():
                c_dim, l_dim, m, opt, lr, temp = config_key
                if c_dim == cat_dim and l_dim == lat_dim and m == method:
                    if best_stat is None or stat['train_loss_mean'] < best_stat['train_loss_mean']:
                        best_stat = stat
                        best_config = config_key
            
            if best_stat is not None:
                _, _, _, opt, lr, temp = best_config
                if not np.isnan(best_stat.get('sample_std_mean', np.nan)):
                    std_str = f"{best_stat['sample_std_mean']:.6f} ± {best_stat['sample_std_std']:.6f}"
                else:
                    std_str = "N/A"
                print(f"{method:<15} {opt:<10} {lr:<10} {temp:<8} {std_str:<30} {best_stat['n_seeds']:<5}")
            else:
                print(f"{method:<15} {'--':<10} {'--':<10} {'--':<8} {'N/A':<30} {'--':<5}")


def print_comparison_results(stats, baseline_stats):
    """Print comparison between reinmax variants and baselines"""
    
    # Get all unique (cat_dim, lat_dim) configurations
    configs = set()
    for config_key in stats.keys():
        cat_dim, lat_dim = config_key[0], config_key[1]
        configs.add((cat_dim, lat_dim))
    for config_key in baseline_stats.keys():
        cat_dim, lat_dim = config_key[0], config_key[1]
        configs.add((cat_dim, lat_dim))
    
    configs = sorted(configs)
    
    baseline_methods = ['gumbel', 'rao_gumbel', 'gst-1.0', 'st', 'reinmax']
    # new_methods = ['reinmax_v2', 'reinmax_v3']
    new_methods = ['reinmax_v3', 'reinmax_cv']
    
    for cat_dim, lat_dim in configs:
        print(f"\n{'='*120}")
        print(f"Categorical Dim: {cat_dim}, Latent Dim: {lat_dim}")
        print(f"{'='*120}")
        print(f"{'Method':<15} {'Optimizer':<10} {'LR':<10} {'Temp':<8} {'Train Loss':<25} {'Test Loss':<25} {'N':<5}")
        print(f"{'-'*120}")
        
        # Print baseline results
        for method in baseline_methods:
            found = False
            for config_key, stat in baseline_stats.items():
                c_dim, l_dim, m, opt, lr, temp = config_key
                if c_dim == cat_dim and l_dim == lat_dim and m == method:
                    train_str = f"{stat['train_loss_mean']:.4f} ± {stat['train_loss_std']:.4f}"
                    test_str = f"{stat['test_loss_mean']:.4f} ± {stat['test_loss_std']:.4f}"
                    print(f"{method:<15} {opt:<10} {lr:<10} {temp:<8} {train_str:<25} {test_str:<25} {stat['n_seeds']:<5}")
                    found = True
                    break
            if not found:
                print(f"{method:<15} {'--':<10} {'--':<10} {'--':<8} {'N/A':<25} {'N/A':<25} {'--':<5}")
        
        print(f"{'-'*120}")
        
        # Print new method results (best hyperparameters based on train loss)
        for method in new_methods:
            # Find best result for this method and config
            best_stat = None
            best_config = None
            for config_key, stat in stats.items():
                c_dim, l_dim, m, opt, lr, temp = config_key
                if c_dim == cat_dim and l_dim == lat_dim and m == method:
                    if best_stat is None or stat['train_loss_mean'] < best_stat['train_loss_mean']:
                        best_stat = stat
                        best_config = config_key
            
            if best_stat is not None:
                _, _, _, opt, lr, temp = best_config
                train_str = f"{best_stat['train_loss_mean']:.4f} ± {best_stat['train_loss_std']:.4f}"
                test_str = f"{best_stat['test_loss_mean']:.4f} ± {best_stat['test_loss_std']:.4f}"
                print(f"{method:<15} {opt:<10} {lr:<10} {temp:<8} {train_str:<25} {test_str:<25} {best_stat['n_seeds']:<5}")
            else:
                print(f"{method:<15} {'--':<10} {'--':<10} {'--':<8} {'N/A':<25} {'N/A':<25} {'--':<5}")


def print_elbo_tables(stats, baseline_stats, results=None, baseline_results=None):
    """
    Print train and test ELBO tables in plain text and LaTeX format.
    Rows: methods (Gumbel, Gumbel-Rao, ST, GST-1.0, Reinmax, Reinmax-V2, Reinmax-V3)
    Columns: configurations (8x4, 4x24, 8x16, 16x12, 64x8, 10x30)
    Highlights best (bold) and second best (underline).
    
    If results and baseline_results are provided, computes average rank based on 
    individual seed rankings (not aggregated stats).
    """
    # Define methods and their display names
    methods_info = [
        ('gumbel', 'Gumbel'),
        ('rao_gumbel', 'Gumbel-Rao'),
        ('st', 'ST'),
        ('gst-1.0', 'GST-1.0'),
        ('reinmax', 'ReinMax'),
        # ('reinmax_v2', 'ReinMax-V2'),
        ('reinmax_v3', 'ReinMax-Rao'),
        ('reinmax_cv', 'ReinMax-CV'),
    ]
    
    # Define configurations in order
    configs = [(8, 4), (4, 24), (8, 16), (16, 12), (64, 8), (10, 30)]
    config_labels = ['8×4', '4×24', '8×16', '16×12', '64×8', '10×30']
    
    # Helper function to get best config_key for a method (selected by train loss mean)
    def get_best_config_key(method, cat_dim, lat_dim, metric='train_loss_mean'):
        best_stat = None
        best_config_key = None
        
        # Check baseline stats first
        for config_key, stat in baseline_stats.items():
            c_dim, l_dim, m, opt, lr, temp = config_key
            if c_dim == cat_dim and l_dim == lat_dim and m == method:
                if best_stat is None or stat[metric] < best_stat[metric]:
                    best_stat = stat
                    best_config_key = config_key
        
        # Check new method stats
        for config_key, stat in stats.items():
            c_dim, l_dim, m, opt, lr, temp = config_key
            if c_dim == cat_dim and l_dim == lat_dim and m == method:
                if best_stat is None or stat[metric] < best_stat[metric]:
                    best_stat = stat
                    best_config_key = config_key
        
        return best_config_key, best_stat
    
    # Helper function to get best result for a method and config (selected by train loss)
    def get_best_result(method, cat_dim, lat_dim, metric='train_loss_mean'):
        _, best_stat = get_best_config_key(method, cat_dim, lat_dim, metric)
        return best_stat
    
    # Build data for tables: (mean, std_error)
    train_data = {}
    test_data = {}
    
    for method_key, method_name in methods_info:
        train_data[method_name] = []
        test_data[method_name] = []
        
        for cat_dim, lat_dim in configs:
            stat = get_best_result(method_key, cat_dim, lat_dim)
            if stat is not None:
                train_data[method_name].append((stat['train_loss_mean'], stat['train_loss_std']))
                test_data[method_name].append((stat['test_loss_mean'], stat['test_loss_std']))
            else:
                train_data[method_name].append((None, None))
                test_data[method_name].append((None, None))
    
    def find_top_three_per_column(data):
        """
        Find best, second best, and third best methods for each column.
        Returns: (best_methods, second_best_methods, third_best_methods)
        """
        best_methods = []
        second_best_methods = []
        third_best_methods = []
        
        for col_idx in range(len(configs)):
            # Get all valid (method, mean) tuples for this column
            valid_entries = []
            for method_name in [m[1] for m in methods_info]:
                val, std = data[method_name][col_idx]
                if val is not None:
                    valid_entries.append((method_name, val))
            
            # Sort by mean (ascending - lower is better)
            valid_entries.sort(key=lambda x: x[1])
            
            if len(valid_entries) >= 1:
                best_methods.append(valid_entries[0][0])
            else:
                best_methods.append(None)
            
            if len(valid_entries) >= 2:
                second_best_methods.append(valid_entries[1][0])
            else:
                second_best_methods.append(None)
            
            if len(valid_entries) >= 3:
                third_best_methods.append(valid_entries[2][0])
            else:
                third_best_methods.append(None)
        
        return best_methods, second_best_methods, third_best_methods
    
    train_best, train_second, train_third = find_top_three_per_column(train_data)
    test_best, test_second, test_third = find_top_three_per_column(test_data)
    
    def compute_average_ranks_per_seed(metric='train_loss', selection_metric='train_loss_mean'):
        """
        Compute average rank for each method across all configurations and seeds.
        Ranks are computed for each individual seed, then averaged.
        
        Args:
            metric: 'train_loss' or 'test_loss' - the metric to rank on
            selection_metric: metric used to select best hyperparameters
        
        Returns: dict of method_name -> (average_rank, std_error)
        """
        if results is None or baseline_results is None:
            # Fall back to stats-based ranking if raw results not provided
            return compute_average_ranks_from_stats(train_data if metric == 'train_loss' else test_data)
        
        method_names = [m[1] for m in methods_info]
        method_keys = [m[0] for m in methods_info]
        ranks_per_method = {m: [] for m in method_names}
        
        # For each configuration
        for col_idx, (cat_dim, lat_dim) in enumerate(configs):
            # Get the best config_key for each method (based on selection_metric)
            method_config_keys = {}
            for method_key, method_name in methods_info:
                config_key, _ = get_best_config_key(method_key, cat_dim, lat_dim, selection_metric)
                if config_key is not None:
                    method_config_keys[method_name] = config_key
            
            # Get raw results for each method's best config
            method_raw_results = {}
            for method_name, config_key in method_config_keys.items():
                # Check both results and baseline_results
                if config_key in results:
                    method_raw_results[method_name] = results[config_key]
                elif config_key in baseline_results:
                    method_raw_results[method_name] = baseline_results[config_key]
            
            # Find all seeds that are common across all methods with data
            if not method_raw_results:
                continue
            
            # Get all seeds available
            all_seeds = set()
            for method_name, seed_data in method_raw_results.items():
                all_seeds.update(seed_data.keys())
            
            # For each seed, rank the methods
            for seed in all_seeds:
                # Collect (method_name, loss_value) for methods that have this seed
                seed_entries = []
                for method_name, seed_data in method_raw_results.items():
                    if seed in seed_data:
                        loss_val = seed_data[seed][metric]
                        seed_entries.append((method_name, loss_val))
                
                # Sort by loss (ascending - lower is better)
                seed_entries.sort(key=lambda x: x[1])
                
                # Assign ranks (1-based)
                for rank, (method_name, _) in enumerate(seed_entries, 1):
                    ranks_per_method[method_name].append(rank)
        
        # Compute average rank and standard error for each method
        avg_ranks = {}
        for method_name in method_names:
            if ranks_per_method[method_name]:
                ranks = ranks_per_method[method_name]
                mean_rank = np.mean(ranks)
                std_error = np.std(ranks) / np.sqrt(len(ranks))
                print(np.std(ranks), len(ranks))
                avg_ranks[method_name] = (mean_rank, std_error)
            else:
                avg_ranks[method_name] = None
        
        return avg_ranks
    
    def compute_average_ranks_from_stats(data):
        """
        Fallback: Compute average rank for each method using aggregated stats.
        Returns: dict of method_name -> (average_rank, std_error)
        """
        method_names = [m[1] for m in methods_info]
        ranks_per_method = {m: [] for m in method_names}
        
        for col_idx in range(len(configs)):
            # Get all valid (method, mean) tuples for this column
            valid_entries = []
            for method_name in method_names:
                val, std = data[method_name][col_idx]
                if val is not None:
                    valid_entries.append((method_name, val))
            
            # Sort by mean (ascending - lower is better)
            valid_entries.sort(key=lambda x: x[1])
            
            # Assign ranks (1-based)
            for rank, (method_name, _) in enumerate(valid_entries, 1):
                ranks_per_method[method_name].append(rank)
        
        # Compute average rank and standard error for each method
        avg_ranks = {}
        for method_name in method_names:
            if ranks_per_method[method_name]:
                ranks = ranks_per_method[method_name]
                mean_rank = np.mean(ranks)
                std_error = np.std(ranks) / np.sqrt(len(ranks))
                avg_ranks[method_name] = (mean_rank, std_error)
            else:
                avg_ranks[method_name] = None
        
        return avg_ranks
    
    # Compute average ranks using per-seed ranking (for tables selected by train loss)
    train_avg_ranks = compute_average_ranks_per_seed(metric='train_loss', selection_metric='train_loss_mean')
    test_avg_ranks = compute_average_ranks_per_seed(metric='test_loss', selection_metric='train_loss_mean')
    
    # Define ReinMax variants for rank computation
    reinmax_variants = ['ReinMax', 'ReinMax-Rao', 'ReinMax-CV']
    
    def compute_reinmax_only_ranks(metric='train_loss', selection_metric='train_loss_mean'):
        """
        Compute average rank for ReinMax variants only across all configurations and seeds.
        Returns: dict of method_name -> (average_rank, std_error) for ReinMax variants, None for others
        """
        if results is None or baseline_results is None:
            return {m: None for m in [mi[1] for mi in methods_info]}
        
        reinmax_methods_info = [(k, n) for k, n in methods_info if n in reinmax_variants]
        method_names = [m[1] for m in reinmax_methods_info]
        ranks_per_method = {m: [] for m in method_names}
        
        # For each configuration
        for col_idx, (cat_dim, lat_dim) in enumerate(configs):
            # Get the best config_key for each ReinMax method (based on selection_metric)
            method_config_keys = {}
            for method_key, method_name in reinmax_methods_info:
                config_key, _ = get_best_config_key(method_key, cat_dim, lat_dim, selection_metric)
                if config_key is not None:
                    method_config_keys[method_name] = config_key
            
            # Get raw results for each method's best config
            method_raw_results = {}
            for method_name, config_key in method_config_keys.items():
                if config_key in results:
                    method_raw_results[method_name] = results[config_key]
                elif config_key in baseline_results:
                    method_raw_results[method_name] = baseline_results[config_key]
            
            if not method_raw_results:
                continue
            
            # Get all seeds available
            all_seeds = set()
            for method_name, seed_data in method_raw_results.items():
                all_seeds.update(seed_data.keys())
            
            # For each seed, rank the ReinMax methods only
            for seed in all_seeds:
                seed_entries = []
                for method_name, seed_data in method_raw_results.items():
                    if seed in seed_data:
                        loss_val = seed_data[seed][metric]
                        seed_entries.append((method_name, loss_val))
                
                # Sort by loss (ascending - lower is better)
                seed_entries.sort(key=lambda x: x[1])
                
                # Assign ranks (1-based)
                for rank, (method_name, _) in enumerate(seed_entries, 1):
                    ranks_per_method[method_name].append(rank)
        
        # Compute average rank and standard error for ReinMax methods
        avg_ranks = {}
        for method_key, method_name in methods_info:
            if method_name in reinmax_variants and ranks_per_method.get(method_name):
                ranks = ranks_per_method[method_name]
                mean_rank = np.mean(ranks)
                std_error = np.std(ranks) / np.sqrt(len(ranks))
                avg_ranks[method_name] = (mean_rank, std_error)
            else:
                avg_ranks[method_name] = None
        
        return avg_ranks
    
    # Compute ReinMax-only ranks
    train_reinmax_ranks = compute_reinmax_only_ranks(metric='train_loss', selection_metric='train_loss_mean')
    test_reinmax_ranks = compute_reinmax_only_ranks(metric='test_loss', selection_metric='train_loss_mean')
    
    # ==================== PLAIN TEXT TABLES ====================
    print("\n" + "="*165)
    print("TRAIN ELBO TABLE (** = best, * = second best, ~ = third best)")
    print("="*165)
    
    # Header
    col_width = 20
    header = f"{'Method':<15}"
    for label in config_labels:
        header += f"{label:^{col_width}}"
    header += f"{'Avg Rank':^{col_width}}"
    print(header)
    print("-" * (15 + col_width * (len(configs) + 1)))
    
    # Rows
    for method_key, method_name in methods_info:
        row = f"{method_name:<15}"
        for col_idx, (val, std) in enumerate(train_data[method_name]):
            if val is not None:
                cell = f"{val:.2f}±{std:.2f}"
                if train_best[col_idx] == method_name:
                    cell = f"**{cell}"
                elif train_second[col_idx] == method_name:
                    cell = f"*{cell}"
                elif train_third[col_idx] == method_name:
                    cell = f"~{cell}"
                else:
                    cell = f"({cell})"
            else:
                cell = "N/A"
            row += f"{cell:^{col_width}}"
        # Add rank column (only for ReinMax variants)
        rank_data = train_reinmax_ranks.get(method_name)
        if rank_data is not None:
            rank_cell = f"{rank_data[0]:.2f}±{rank_data[1]:.2f}"
        else:
            rank_cell = "-"
        row += f"{rank_cell:^{col_width}}"
        print(row)
    
    print("\n" + "="*165)
    print("TEST ELBO TABLE (** = best, * = second best, ~ = third best)")
    print("="*165)
    
    # Header
    print(header)
    print("-" * (15 + col_width * (len(configs) + 1)))
    
    # Rows
    for method_key, method_name in methods_info:
        row = f"{method_name:<15}"
        for col_idx, (val, std) in enumerate(test_data[method_name]):
            if val is not None:
                cell = f"{val:.2f}±{std:.2f}"
                if test_best[col_idx] == method_name:
                    cell = f"**{cell}"
                elif test_second[col_idx] == method_name:
                    cell = f"*{cell}"
                elif test_third[col_idx] == method_name:
                    cell = f"~{cell}"
                else:
                    cell = f"({cell})"
            else:
                cell = "N/A"
            row += f"{cell:^{col_width}}"
        # Add rank column (only for ReinMax variants)
        rank_data = test_reinmax_ranks.get(method_name)
        if rank_data is not None:
            rank_cell = f"{rank_data[0]:.2f}±{rank_data[1]:.2f}"
        else:
            rank_cell = "-"
        row += f"{rank_cell:^{col_width}}"
        print(row)
    
    # ==================== LATEX TABLES ====================
    print("\n" + "="*120)
    print("LATEX TABLES")
    print("="*120)
    
    # Train ELBO LaTeX table
    print("\n% Train ELBO Table")
    print("% Best in bold, second best underlined, third best italic, rest grey")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Train ELBO ($\\downarrow$) across different configurations. Best results are in \\textbf{bold}, second best are \\underline{underlined}, third best are \\textit{italic}.}")
    print("\\label{tab:train_elbo}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{l" + "c" * len(configs) + "c}")
    print("\\toprule")
    print("Method & " + " & ".join(config_labels) + " & Avg Rank \\\\")
    print("\\midrule")
    
    for method_key, method_name in methods_info:
        row_cells = [method_name]
        for col_idx, (val, std) in enumerate(train_data[method_name]):
            if val is not None:
                cell = f"{val:.2f}$\\pm${std:.2f}"
                if train_best[col_idx] == method_name:
                    cell = f"\\textbf{{{cell}}}"
                elif train_second[col_idx] == method_name:
                    cell = f"\\underline{{{cell}}}"
                elif train_third[col_idx] == method_name:
                    cell = f"\\textit{{{cell}}}"
                else:
                    cell = f"\\textcolor{{gray}}{{{cell}}}"
            else:
                cell = "N/A"
            row_cells.append(cell)
        # Add rank column (only for ReinMax variants)
        rank_data = train_reinmax_ranks.get(method_name)
        if rank_data is not None:
            row_cells.append(f"{rank_data[0]:.2f}$\\pm${rank_data[1]:.2f}")
        else:
            row_cells.append("-")
        print(" & ".join(row_cells) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table}")
    
    # Test ELBO LaTeX table
    print("\n% Test ELBO Table")
    print("% Best in bold, second best underlined, third best italic, rest grey")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Test ELBO ($\\downarrow$) across different configurations. Best results are in \\textbf{bold}, second best are \\underline{underlined}, third best are \\textit{italic}.}")
    print("\\label{tab:test_elbo}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{l" + "c" * len(configs) + "c}")
    print("\\toprule")
    print("Method & " + " & ".join(config_labels) + " & Avg Rank \\\\")
    print("\\midrule")
    
    for method_key, method_name in methods_info:
        row_cells = [method_name]
        for col_idx, (val, std) in enumerate(test_data[method_name]):
            if val is not None:
                cell = f"{val:.2f}$\\pm${std:.2f}"
                if test_best[col_idx] == method_name:
                    cell = f"\\textbf{{{cell}}}"
                elif test_second[col_idx] == method_name:
                    cell = f"\\underline{{{cell}}}"
                elif test_third[col_idx] == method_name:
                    cell = f"\\textit{{{cell}}}"
                else:
                    cell = f"\\textcolor{{gray}}{{{cell}}}"
            else:
                cell = "N/A"
            row_cells.append(cell)
        # Add rank column (only for ReinMax variants)
        rank_data = test_reinmax_ranks.get(method_name)
        if rank_data is not None:
            row_cells.append(f"{rank_data[0]:.2f}$\\pm${rank_data[1]:.2f}")
        else:
            row_cells.append("-")
        print(" & ".join(row_cells) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table}")
    
    # ==================== TABLES SELECTED BY TEST LOSS ====================
    # Build data for tables selected by test loss: (mean, std_error)
    train_data_by_test = {}
    test_data_by_test = {}
    
    for method_key, method_name in methods_info:
        train_data_by_test[method_name] = []
        test_data_by_test[method_name] = []
        
        for cat_dim, lat_dim in configs:
            stat = get_best_result(method_key, cat_dim, lat_dim, metric='test_loss_mean')
            if stat is not None:
                train_data_by_test[method_name].append((stat['train_loss_mean'], stat['train_loss_std']))
                test_data_by_test[method_name].append((stat['test_loss_mean'], stat['test_loss_std']))
            else:
                train_data_by_test[method_name].append((None, None))
                test_data_by_test[method_name].append((None, None))
    
    train_best_by_test, train_second_by_test, train_third_by_test = find_top_three_per_column(train_data_by_test)
    test_best_by_test, test_second_by_test, test_third_by_test = find_top_three_per_column(test_data_by_test)
    
    # Compute average ranks for tables selected by test loss (using per-seed ranking)
    train_avg_ranks_by_test = compute_average_ranks_per_seed(metric='train_loss', selection_metric='test_loss_mean')
    test_avg_ranks_by_test = compute_average_ranks_per_seed(metric='test_loss', selection_metric='test_loss_mean')
    
    # Compute ReinMax-only ranks for tables selected by test loss
    train_reinmax_ranks_by_test = compute_reinmax_only_ranks(metric='train_loss', selection_metric='test_loss_mean')
    test_reinmax_ranks_by_test = compute_reinmax_only_ranks(metric='test_loss', selection_metric='test_loss_mean')
    
    # ==================== PLAIN TEXT TABLES (SELECTED BY TEST) ====================
    print("\n" + "="*165)
    print("TRAIN ELBO TABLE - HYPERPARAMETERS SELECTED BY TEST LOSS (** = best, * = second best, ~ = third best)")
    print("="*165)
    
    print(header)
    print("-" * (15 + col_width * (len(configs) + 1)))
    
    for method_key, method_name in methods_info:
        row = f"{method_name:<15}"
        for col_idx, (val, std) in enumerate(train_data_by_test[method_name]):
            if val is not None:
                cell = f"{val:.2f}±{std:.2f}"
                if train_best_by_test[col_idx] == method_name:
                    cell = f"**{cell}"
                elif train_second_by_test[col_idx] == method_name:
                    cell = f"*{cell}"
                elif train_third_by_test[col_idx] == method_name:
                    cell = f"~{cell}"
                else:
                    cell = f"({cell})"
            else:
                cell = "N/A"
            row += f"{cell:^{col_width}}"
        # Add rank column (only for ReinMax variants)
        rank_data = train_reinmax_ranks_by_test.get(method_name)
        if rank_data is not None:
            rank_cell = f"{rank_data[0]:.2f}±{rank_data[1]:.2f}"
        else:
            rank_cell = "-"
        row += f"{rank_cell:^{col_width}}"
        print(row)
    
    print("\n" + "="*165)
    print("TEST ELBO TABLE - HYPERPARAMETERS SELECTED BY TEST LOSS (** = best, * = second best, ~ = third best)")
    print("="*165)
    
    print(header)
    print("-" * (15 + col_width * (len(configs) + 1)))
    
    for method_key, method_name in methods_info:
        row = f"{method_name:<15}"
        for col_idx, (val, std) in enumerate(test_data_by_test[method_name]):
            if val is not None:
                cell = f"{val:.2f}±{std:.2f}"
                if test_best_by_test[col_idx] == method_name:
                    cell = f"**{cell}"
                elif test_second_by_test[col_idx] == method_name:
                    cell = f"*{cell}"
                elif test_third_by_test[col_idx] == method_name:
                    cell = f"~{cell}"
                else:
                    cell = f"({cell})"
            else:
                cell = "N/A"
            row += f"{cell:^{col_width}}"
        # Add rank column (only for ReinMax variants)
        rank_data = test_reinmax_ranks_by_test.get(method_name)
        if rank_data is not None:
            rank_cell = f"{rank_data[0]:.2f}±{rank_data[1]:.2f}"
        else:
            rank_cell = "-"
        row += f"{rank_cell:^{col_width}}"
        print(row)
    
    # ==================== LATEX TABLES (SELECTED BY TEST) ====================
    print("\n" + "="*120)
    print("LATEX TABLES (HYPERPARAMETERS SELECTED BY TEST LOSS)")
    print("="*120)
    
    # Train ELBO LaTeX table (selected by test)
    print("\n% Train ELBO Table (hyperparameters selected by test loss)")
    print("% Best in bold, second best underlined, third best italic, rest grey")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Train ELBO ($\\downarrow$) across different configurations (hyperparameters selected by test loss). Best results are in \\textbf{bold}, second best are \\underline{underlined}, third best are \\textit{italic}.}")
    print("\\label{tab:train_elbo_by_test}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{l" + "c" * len(configs) + "c}")
    print("\\toprule")
    print("Method & " + " & ".join(config_labels) + " & Avg Rank \\\\")
    print("\\midrule")
    
    for method_key, method_name in methods_info:
        row_cells = [method_name]
        for col_idx, (val, std) in enumerate(train_data_by_test[method_name]):
            if val is not None:
                cell = f"{val:.2f}$\\pm${std:.2f}"
                if train_best_by_test[col_idx] == method_name:
                    cell = f"\\textbf{{{cell}}}"
                elif train_second_by_test[col_idx] == method_name:
                    cell = f"\\underline{{{cell}}}"
                elif train_third_by_test[col_idx] == method_name:
                    cell = f"\\textit{{{cell}}}"
                else:
                    cell = f"\\textcolor{{gray}}{{{cell}}}"
            else:
                cell = "N/A"
            row_cells.append(cell)
        # Add rank column (only for ReinMax variants)
        rank_data = train_reinmax_ranks_by_test.get(method_name)
        if rank_data is not None:
            row_cells.append(f"{rank_data[0]:.2f}$\\pm${rank_data[1]:.2f}")
        else:
            row_cells.append("-")
        print(" & ".join(row_cells) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table}")
    
    # Test ELBO LaTeX table (selected by test)
    print("\n% Test ELBO Table (hyperparameters selected by test loss)")
    print("% Best in bold, second best underlined, third best italic, rest grey")
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Test ELBO ($\\downarrow$) across different configurations (hyperparameters selected by test loss). Best results are in \\textbf{bold}, second best are \\underline{underlined}, third best are \\textit{italic}.}")
    print("\\label{tab:test_elbo_by_test}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{l" + "c" * len(configs) + "c}")
    print("\\toprule")
    print("Method & " + " & ".join(config_labels) + " & Avg Rank \\\\")
    print("\\midrule")
    
    for method_key, method_name in methods_info:
        row_cells = [method_name]
        for col_idx, (val, std) in enumerate(test_data_by_test[method_name]):
            if val is not None:
                cell = f"{val:.2f}$\\pm${std:.2f}"
                if test_best_by_test[col_idx] == method_name:
                    cell = f"\\textbf{{{cell}}}"
                elif test_second_by_test[col_idx] == method_name:
                    cell = f"\\underline{{{cell}}}"
                elif test_third_by_test[col_idx] == method_name:
                    cell = f"\\textit{{{cell}}}"
                else:
                    cell = f"\\textcolor{{gray}}{{{cell}}}"
            else:
                cell = "N/A"
            row_cells.append(cell)
        # Add rank column (only for ReinMax variants)
        rank_data = test_reinmax_ranks_by_test.get(method_name)
        if rank_data is not None:
            row_cells.append(f"{rank_data[0]:.2f}$\\pm${rank_data[1]:.2f}")
        else:
            row_cells.append("-")
        print(" & ".join(row_cells) + " \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\end{table}")

def plot_gradient_analysis(trajectories, cat_dim, lat_dim, output_dir='./figures'):
    """
    Plot gradient norm and variance analysis for a specific configuration.
    Creates multiple visualizations:
    1. Gradient norm vs epoch
    2. Gradient variance vs epoch  
    3. Signal-to-noise ratio (norm / std) vs epoch
    4. Scatter plot of norm vs variance at final epoch
    """
    setup_publication_style()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    baseline_methods = ['gumbel', 'rao_gumbel', 'gst-1.0', 'st', 'reinmax']
    # new_methods = ['reinmax_v2', 'reinmax_v3']
    new_methods = ['reinmax_v3', 'reinmax_cv']
    all_methods = baseline_methods + new_methods
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # For scatter plot
    final_norms = {}
    final_vars = {}
    
    for method in all_methods:
        # Find the best config for this method (selected by train loss)
        best_config = None
        best_train_loss = float('inf')
        
        for config_key, seed_data in trajectories.items():
            c_dim, l_dim, m, opt, lr, temp = config_key
            if c_dim == cat_dim and l_dim == lat_dim and m == method:
                if len(seed_data) > 0:
                    final_losses = [seed_data[s][-1, 0] for s in seed_data.keys()]
                    mean_final = np.mean(final_losses)
                    if mean_final < best_train_loss:
                        best_train_loss = mean_final
                        best_config = config_key
        
        if best_config is None:
            continue
        
        seed_data = trajectories[best_config]
        if len(seed_data) == 0:
            continue
        
        first_seed = list(seed_data.keys())[0]
        n_epochs = len(seed_data[first_seed])
        epochs = np.arange(1, n_epochs + 1)
        
        # Collect gradient stats across seeds
        grad_var_all = []
        grad_norm_all = []
        
        for seed, metrics in seed_data.items():
            grad_std = metrics[:, -2]  # Second last column: gradient std
            grad_norm = metrics[:, -1]  # Last column: gradient norm
            grad_var_all.append(grad_std ** 2)
            grad_norm_all.append(grad_norm)
        
        grad_var_all = np.array(grad_var_all)
        grad_norm_all = np.array(grad_norm_all)
        
        # Compute mean and std across seeds
        var_mean = np.mean(grad_var_all, axis=0)
        var_std = np.std(grad_var_all, axis=0)
        norm_mean = np.mean(grad_norm_all, axis=0)
        norm_std = np.std(grad_norm_all, axis=0)
        
        # Signal-to-noise ratio: norm / std
        snr_all = grad_norm_all / (np.sqrt(grad_var_all) + 1e-10)
        snr_mean = np.mean(snr_all, axis=0)
        snr_std = np.std(snr_all, axis=0)
        
        style = get_method_style(method)
        
        # Plot 1: Gradient Norm vs Epoch
        axes[0, 0].plot(epochs, norm_mean, color=style['color'], linestyle=style['linestyle'],
                        label=style['label'], marker=style['marker'], markevery=max(1, n_epochs//10))
        axes[0, 0].fill_between(epochs, norm_mean - norm_std, norm_mean + norm_std,
                                color=style['color'], alpha=0.15)
        
        # Plot 2: Gradient Variance vs Epoch
        axes[0, 1].plot(epochs, var_mean, color=style['color'], linestyle=style['linestyle'],
                        label=style['label'], marker=style['marker'], markevery=max(1, n_epochs//10))
        axes[0, 1].fill_between(epochs, var_mean - var_std, var_mean + var_std,
                                color=style['color'], alpha=0.15)
        
        # Plot 3: Signal-to-Noise Ratio vs Epoch
        axes[1, 0].plot(epochs, snr_mean, color=style['color'], linestyle=style['linestyle'],
                        label=style['label'], marker=style['marker'], markevery=max(1, n_epochs//10))
        axes[1, 0].fill_between(epochs, snr_mean - snr_std, snr_mean + snr_std,
                                color=style['color'], alpha=0.15)
        
        # Store final values for scatter plot
        final_norms[method] = (np.mean(grad_norm_all[:, -1]), np.std(grad_norm_all[:, -1]))
        final_vars[method] = (np.mean(grad_var_all[:, -1]), np.std(grad_var_all[:, -1]))
    
    # Plot 4: Scatter plot of final norm vs variance
    for method in all_methods:
        if method in final_norms and method in final_vars:
            style = get_method_style(method)
            norm_mean, norm_std = final_norms[method]
            var_mean, var_std = final_vars[method]
            
            axes[1, 1].errorbar(var_mean, norm_mean, xerr=var_std, yerr=norm_std,
                               color=style['color'], marker=style['marker'], 
                               markersize=8, capsize=3, label=style['label'])
    
    # Formatting
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Gradient Norm')
    axes[0, 0].set_title('Gradient Norm vs Epoch')
    axes[0, 0].legend(loc='best', framealpha=0.9, fontsize=8)
    axes[0, 0].set_yscale('log')
    
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Gradient Variance')
    axes[0, 1].set_title('Gradient Variance vs Epoch')
    axes[0, 1].legend(loc='best', framealpha=0.9, fontsize=8)
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SNR (Norm / Std)')
    axes[1, 0].set_title('Signal-to-Noise Ratio vs Epoch')
    axes[1, 0].legend(loc='best', framealpha=0.9, fontsize=8)
    axes[1, 0].set_yscale('log')
    
    axes[1, 1].set_xlabel('Gradient Variance (final epoch)')
    axes[1, 1].set_ylabel('Gradient Norm (final epoch)')
    axes[1, 1].set_title('Norm vs Variance Trade-off')
    axes[1, 1].legend(loc='best', framealpha=0.9, fontsize=8)
    axes[1, 1].set_xscale('log')
    axes[1, 1].set_yscale('log')
    
    plt.suptitle(f'Gradient Analysis (Cat={cat_dim}, Lat={lat_dim})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = output_path / f'gradient_analysis_cat{cat_dim}_lat{lat_dim}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


def plot_efficiency_frontier(trajectories, cat_dim, lat_dim, output_dir='./figures'):
    """
    Plot loss vs gradient variance trade-off (efficiency frontier).
    Shows which methods achieve low loss with low variance.
    """
    setup_publication_style()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    baseline_methods = ['gumbel', 'rao_gumbel', 'gst-1.0', 'st', 'reinmax']
    # new_methods = ['reinmax_v2', 'reinmax_v3']
    new_methods = ['reinmax_v3', 'reinmax_cv']
    all_methods = baseline_methods + new_methods
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for method in all_methods:
        # Find the best config for this method
        best_config = None
        best_train_loss = float('inf')
        
        for config_key, seed_data in trajectories.items():
            c_dim, l_dim, m, opt, lr, temp = config_key
            if c_dim == cat_dim and l_dim == lat_dim and m == method:
                if len(seed_data) > 0:
                    final_losses = [seed_data[s][-1, 0] for s in seed_data.keys()]
                    mean_final = np.mean(final_losses)
                    if mean_final < best_train_loss:
                        best_train_loss = mean_final
                        best_config = config_key
        
        if best_config is None:
            continue
        
        seed_data = trajectories[best_config]
        if len(seed_data) == 0:
            continue
        
        # Collect final epoch stats
        train_losses = []
        test_losses = []
        grad_vars = []
        
        for seed, metrics in seed_data.items():
            train_losses.append(metrics[-1, 0])
            test_losses.append(metrics[-1, 3])
            grad_vars.append(metrics[-1, -2] ** 2)  # variance = std^2
        
        train_mean, train_std = np.mean(train_losses), np.std(train_losses)
        test_mean, test_std = np.mean(test_losses), np.std(test_losses)
        var_mean, var_std = np.mean(grad_vars), np.std(grad_vars)
        
        style = get_method_style(method)
        
        # Train loss vs variance
        axes[0].errorbar(var_mean, train_mean, xerr=var_std, yerr=train_std,
                        color=style['color'], marker=style['marker'],
                        markersize=10, capsize=4, label=style['label'],
                        linewidth=2)
        
        # Test loss vs variance
        axes[1].errorbar(var_mean, test_mean, xerr=var_std, yerr=test_std,
                        color=style['color'], marker=style['marker'],
                        markersize=10, capsize=4, label=style['label'],
                        linewidth=2)
    
    # Formatting
    axes[0].set_xlabel('Gradient Variance')
    axes[0].set_ylabel('Train Loss (ELBO)')
    axes[0].set_title('Train Loss vs Gradient Variance')
    axes[0].legend(loc='best', framealpha=0.9)
    axes[0].set_xscale('log')
    
    axes[1].set_xlabel('Gradient Variance')
    axes[1].set_ylabel('Test Loss (ELBO)')
    axes[1].set_title('Test Loss vs Gradient Variance')
    axes[1].legend(loc='best', framealpha=0.9)
    axes[1].set_xscale('log')
    
    # Add annotation for ideal region (bottom-left)
    for ax in axes:
        ax.annotate('Better', xy=(0.05, 0.05), xycoords='axes fraction',
                   fontsize=12, color='green', fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    plt.suptitle(f'Efficiency Frontier (Cat={cat_dim}, Lat={lat_dim})', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = output_path / f'efficiency_frontier_cat{cat_dim}_lat{lat_dim}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


def print_gradient_stats_table(stats, baseline_stats):
    """
    Print gradient norm and variance statistics in table format.
    """
    methods_info = [
        ('gumbel', 'Gumbel'),
        ('rao_gumbel', 'Gumbel-Rao'),
        ('st', 'ST'),
        ('gst-1.0', 'GST-1.0'),
        ('reinmax', 'ReinMax'),
        # ('reinmax_v2', 'ReinMax-V2'),
        ('reinmax_v3', 'ReinMax-V3'),
        ('reinmax_cv', 'ReinMax-CV'),
    ]
    
    configs = [(8, 4), (4, 24), (8, 16), (16, 12), (64, 8), (10, 30)]
    config_labels = ['8×4', '4×24', '8×16', '16×12', '64×8', '10×30']
    
    print("\n" + "="*100)
    print("GRADIENT VARIANCE TABLE (lower is better)")
    print("="*100)
    
    col_width = 16
    header = f"{'Method':<15}"
    for label in config_labels:
        header += f"{label:^{col_width}}"
    print(header)
    print("-" * (15 + col_width * len(configs)))
    
    for method_key, method_name in methods_info:
        row = f"{method_name:<15}"
        for cat_dim, lat_dim in configs:
            # Find best config for this method
            best_stat = None
            for config_key, stat in baseline_stats.items():
                c_dim, l_dim, m, opt, lr, temp = config_key
                if c_dim == cat_dim and l_dim == lat_dim and m == method_key:
                    if best_stat is None or stat['train_loss_mean'] < best_stat['train_loss_mean']:
                        best_stat = stat
            
            for config_key, stat in stats.items():
                c_dim, l_dim, m, opt, lr, temp = config_key
                if c_dim == cat_dim and l_dim == lat_dim and m == method_key:
                    if best_stat is None or stat['train_loss_mean'] < best_stat['train_loss_mean']:
                        best_stat = stat
            
            if best_stat is not None and not np.isnan(best_stat.get('sample_std_mean', np.nan)):
                # Display variance (std^2)
                variance = best_stat['sample_std_mean'] ** 2
                cell = f"{variance:.2e}"
            else:
                cell = "N/A"
            row += f"{cell:^{col_width}}"
        print(row)


def plot_all_configurations(results_dir='./results', output_dir='./figures'):
    """Generate plots for all configurations"""
    print("Loading full trajectories...")
    trajectories = load_full_trajectories(results_dir)
    
    # Get all unique configurations
    configs = set()
    for config_key in trajectories.keys():
        cat_dim, lat_dim = config_key[0], config_key[1]
        configs.add((cat_dim, lat_dim))
    
    configs = sorted(configs)
    print(f"Found {len(configs)} configurations")
    
    for cat_dim, lat_dim in configs:
        print(f"\nPlotting Cat={cat_dim}, Lat={lat_dim}...")
        plot_losses_vs_epoch(trajectories, cat_dim, lat_dim, output_dir)
        plot_sample_variance_vs_epoch(trajectories, cat_dim, lat_dim, output_dir)
        plot_gradient_analysis(trajectories, cat_dim, lat_dim, output_dir)
        plot_efficiency_frontier(trajectories, cat_dim, lat_dim, output_dir)
    
    print(f"\nAll figures saved to {output_dir}/")



def main():
    print("Loading results...")
    results = load_results()
    
    print(f"Found {len(results)} configurations")
    
    # Check reinmax_cv file status
    check_reinmax_cv_status()
    
    print("Computing statistics...")
    stats = compute_statistics(results)
    
    print("\nLoading baseline results...")
    baseline_results = load_baseline_results()
    print(f"Found {len(baseline_results)} baseline configurations")
    
    print("Computing baseline statistics...")
    baseline_stats = compute_statistics(baseline_results)
    
    print("\n" + "="*120)
    print("COMPARISON: Baselines vs New Methods")
    print("="*120)
    print_comparison_results(stats, baseline_stats)
    
    # print("\n\n" + "="*120)
    # print("SAMPLE STANDARD DEVIATION COMPARISON")
    # print("="*120)
    # print_sample_std_comparison(stats, baseline_stats)
    
    print("\n\n")
    print_elbo_tables(stats, baseline_stats, results, baseline_results)
    
    print("\n\nBest Results by Configuration and Method (New Methods Only)")
    print_best_results(stats)


def load_full_trajectories(results_dir='./results'):
    """Load full training trajectories for all methods"""
    trajectories = defaultdict(lambda: defaultdict(list))
    
    results_path = Path(results_dir)
    if not results_path.exists():
        return trajectories
    
    # Load new methods
    optimizer_options = ["Adam", "RAdam"]
    learning_rate_options = [0.0007, 0.0005, 0.0003]
    temperature_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    categorical_dim_options = [8, 4, 8, 16, 64, 10]
    latent_dim_options = [4, 24, 16, 12, 8, 30]
    # methods = ["reinmax_v2", "reinmax_v3"]
    methods = ["reinmax_v3"]
    seeds = range(10)
    
    # Define tau2 and eta options for reinmax_cv
    tau2_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    eta_options = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    
    for cat_dim, lat_dim in zip(categorical_dim_options, latent_dim_options):
        for method in methods:
            for optimizer in optimizer_options:
                for lr in learning_rate_options:
                    for temp in temperature_options:
                        for seed in seeds:
                            filename = (
                                results_path / 
                                f"results_seed{seed}_{method}_cat{cat_dim}_lat{lat_dim}_"
                                f"opt{optimizer}_lr{lr}_temp{temp}.txt"
                            )
                            
                            if filename.exists():
                                try:
                                    metrics = np.loadtxt(filename, delimiter=',')
                                    # Skip files that don't have all 160 epochs
                                    if len(metrics) < 160:
                                        continue
                                    config_key = (cat_dim, lat_dim, method, optimizer, lr, temp)
                                    trajectories[config_key][seed] = metrics
                                except:
                                    continue
    
    # Load baseline methods
    for (method, cat_dim, lat_dim), (lr, temp, optimizer) in baseline_hyperparameters.items():
        for seed in seeds:
            filename = (
                results_path / 
                f"results_seed{seed}_{method}_cat{cat_dim}_lat{lat_dim}_"
                f"opt{optimizer}_lr{lr}_temp{temp}.txt"
            )
            
            if filename.exists():
                try:
                    metrics = np.loadtxt(filename, delimiter=',')
                    # Skip files that don't have all 160 epochs
                    if len(metrics) < 160:
                        continue
                    config_key = (cat_dim, lat_dim, method, optimizer, lr, temp)
                    trajectories[config_key][seed] = metrics
                except:
                    continue
    
    # Handle reinmax_cv separately - find best tau2/eta by averaging across seeds first
    print("\nFinding best eta and tau2 for reinmax_cv (averaging across seeds)...")
    
    for cat_dim, lat_dim in zip(categorical_dim_options, latent_dim_options):
        # Get baseline reinmax hyperparameters for this configuration
        key = ('reinmax', cat_dim, lat_dim)
        if key not in baseline_hyperparameters:
            continue
            
        baseline_lr, baseline_temp, baseline_optimizer = baseline_hyperparameters[key]
        
        # Collect full trajectories for all (eta, tau2) combinations across all seeds
        combo_trajectories = {}  # (eta, tau2) -> {seed: metrics_array}
        
        for tau2 in tau2_options:
            for eta in eta_options:
                combo_trajectories[(eta, tau2)] = {}
                for seed in seeds:
                    fname = f"results_seed{seed}_reinmax_cv_cat{cat_dim}_lat{lat_dim}_opt{baseline_optimizer}_lr{baseline_lr}_temp{baseline_temp}_eta{eta}_tau2{tau2}.txt"
                    fpath = results_path / fname
                    
                    if fpath.exists():
                        try:
                            data = np.loadtxt(fpath, delimiter=',')
                            if len(data) >= 160:
                                combo_trajectories[(eta, tau2)][seed] = data
                        except Exception as e:
                            pass
        
        # Find the best (eta, tau2) based on average final train loss across seeds
        best_avg_loss = float('inf')
        best_combo = None
        
        for (eta, tau2), seed_data in combo_trajectories.items():
            if len(seed_data) >= 5:  # Require at least 5 seeds
                avg_train_loss = np.mean([seed_data[s][-1, 0] for s in seed_data.keys()])
                if avg_train_loss < best_avg_loss:
                    best_avg_loss = avg_train_loss
                    best_combo = (eta, tau2)
        
        # Store trajectories using the best (eta, tau2) for all seeds
        if best_combo is not None:
            config_key = (cat_dim, lat_dim, 'reinmax_cv', baseline_optimizer, baseline_lr, baseline_temp)
            for seed, data in combo_trajectories[best_combo].items():
                trajectories[config_key][seed] = data
            
            print(f"  {cat_dim}×{lat_dim}: best η={best_combo[0]}, τ²={best_combo[1]} (avg loss={best_avg_loss:.4f}, {len(combo_trajectories[best_combo])} seeds)")
    
    return trajectories


def setup_publication_style():
    """Set up matplotlib for publication-quality figures"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman'],
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'lines.markersize': 4,
        'figure.figsize': (7, 5),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
    })


def get_method_style(method):
    """Return consistent color and style for each method"""
    styles = {
        'gumbel': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'label': 'Gumbel-Softmax'},
        'rao_gumbel': {'color': '#ff7f0e', 'linestyle': '-', 'marker': 's', 'label': 'Rao-Blackwell Gumbel'},
        'gst-1.0': {'color': '#2ca02c', 'linestyle': '-', 'marker': '^', 'label': 'GST'},
        'st': {'color': '#d62728', 'linestyle': '-', 'marker': 'v', 'label': 'Straight-Through'},
        'reinmax': {'color': '#9467bd', 'linestyle': '-', 'marker': 'D', 'label': 'ReinMax'},
        # 'reinmax_v2': {'color': '#8c564b', 'linestyle': '--', 'marker': 'p', 'label': 'ReinMax-v2'},
        'reinmax_v3': {'color': '#e377c2', 'linestyle': '--', 'marker': 'h', 'label': 'ReinMax-Rao'},
        'reinmax_cv': {'color': '#8c564b', 'linestyle': '--', 'marker': 'p', 'label': 'ReinMax-CV'},
    }
    return styles.get(method, {'color': 'gray', 'linestyle': '-', 'marker': 'x', 'label': method})


def plot_losses_vs_epoch(trajectories, cat_dim, lat_dim, output_dir='./figures'):
    """
    Plot train and test losses vs epoch for a specific configuration.
    Creates publication-quality figures and saves to PDF.
    """
    setup_publication_style()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    baseline_methods = ['gumbel', 'rao_gumbel', 'gst-1.0', 'st', 'reinmax']
    # new_methods = ['reinmax_v2', 'reinmax_v3']
    new_methods = ['reinmax_v3', 'reinmax_cv']
    all_methods = baseline_methods + new_methods
    
    # Create figure with two subplots (train and test)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    for method in all_methods:
        # Find the best config for this method (selected by train loss)
        best_config = None
        best_train_loss = float('inf')
        
        for config_key, seed_data in trajectories.items():
            c_dim, l_dim, m, opt, lr, temp = config_key
            if c_dim == cat_dim and l_dim == lat_dim and m == method:
                if len(seed_data) > 0:
                    # Compute mean train loss at final epoch
                    final_losses = [seed_data[s][-1, 0] for s in seed_data.keys()]
                    mean_final = np.mean(final_losses)
                    if mean_final < best_train_loss:
                        best_train_loss = mean_final
                        best_config = config_key
        
        if best_config is None:
            continue
        
        seed_data = trajectories[best_config]
        if len(seed_data) == 0:
            continue
        
        # Get number of epochs from first seed
        first_seed = list(seed_data.keys())[0]
        n_epochs = len(seed_data[first_seed])
        epochs = np.arange(1, n_epochs + 1)
        
        # Collect train and test losses across seeds
        train_losses_all = []
        test_losses_all = []
        
        for seed, metrics in seed_data.items():
            train_losses_all.append(metrics[:, 0])  # Column 0: train loss
            test_losses_all.append(metrics[:, 3])   # Column 3: test loss
        
        train_losses_all = np.array(train_losses_all)
        test_losses_all = np.array(test_losses_all)
        
        # Compute mean and std
        train_mean = np.mean(train_losses_all, axis=0)
        train_std = np.std(train_losses_all, axis=0)
        test_mean = np.mean(test_losses_all, axis=0)
        test_std = np.std(test_losses_all, axis=0)
        
        style = get_method_style(method)
        
        # Plot train loss
        axes[0].plot(epochs, train_mean, color=style['color'], linestyle=style['linestyle'],
                     label=style['label'], marker=style['marker'], markevery=max(1, n_epochs//10))
        axes[0].fill_between(epochs, train_mean - train_std, train_mean + train_std,
                             color=style['color'], alpha=0.15)
        
        # Plot test loss
        axes[1].plot(epochs, test_mean, color=style['color'], linestyle=style['linestyle'],
                     label=style['label'], marker=style['marker'], markevery=max(1, n_epochs//10))
        axes[1].fill_between(epochs, test_mean - test_std, test_mean + test_std,
                             color=style['color'], alpha=0.15)
    
    # Formatting
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Train Loss (ELBO)')
    axes[0].set_title(f'Training Loss (Cat={cat_dim}, Lat={lat_dim})')
    axes[0].legend(loc='upper right', framealpha=0.9)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Test Loss (ELBO)')
    axes[1].set_title(f'Test Loss (Cat={cat_dim}, Lat={lat_dim})')
    axes[1].legend(loc='upper right', framealpha=0.9)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    filename = output_path / f'losses_cat{cat_dim}_lat{lat_dim}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


def plot_sample_variance_vs_epoch(trajectories, cat_dim, lat_dim, output_dir='./figures'):
    """
    Plot sample variance (std) vs epoch for a specific configuration.
    Creates publication-quality figures and saves to PDF.
    """
    setup_publication_style()
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    baseline_methods = ['gumbel', 'rao_gumbel', 'gst-1.0', 'st', 'reinmax']
    # new_methods = ['reinmax_v2', 'reinmax_v3']
    new_methods = ['reinmax_v3', 'reinmax_cv']
    all_methods = baseline_methods + new_methods
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    for method in all_methods:
        # Find the best config for this method (selected by train loss)
        best_config = None
        best_train_loss = float('inf')
        
        for config_key, seed_data in trajectories.items():
            c_dim, l_dim, m, opt, lr, temp = config_key
            if c_dim == cat_dim and l_dim == lat_dim and m == method:
                if len(seed_data) > 0:
                    # Compute mean train loss at final epoch
                    final_losses = [seed_data[s][-1, 0] for s in seed_data.keys()]
                    mean_final = np.mean(final_losses)
                    if mean_final < best_train_loss:
                        best_train_loss = mean_final
                        best_config = config_key
        
        if best_config is None:
            continue
        
        seed_data = trajectories[best_config]
        if len(seed_data) == 0:
            continue
        
        # Get number of epochs from first seed
        first_seed = list(seed_data.keys())[0]
        n_epochs = len(seed_data[first_seed])
        epochs = np.arange(1, n_epochs + 1)
        
        # Collect sample variance across seeds (second last column)
        sample_var_all = []
        
        for seed, metrics in seed_data.items():
            # Sample std is in second last column, square it to get variance
            sample_std = metrics[:, -2]
            sample_var = sample_std ** 2
            sample_var_all.append(sample_var)
        
        sample_var_all = np.array(sample_var_all)
        
        # Compute mean and std
        var_mean = np.mean(sample_var_all, axis=0)
        var_std = np.std(sample_var_all, axis=0)
        
        style = get_method_style(method)
        
        ax.plot(epochs, var_mean, color=style['color'], linestyle=style['linestyle'],
                label=style['label'], marker=style['marker'], markevery=max(1, n_epochs//10))
        ax.fill_between(epochs, var_mean - var_std, var_mean + var_std,
                        color=style['color'], alpha=0.15)
    
    # Formatting
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Sample Variance')
    ax.set_title(f'Gradient Estimator Sample Variance (Cat={cat_dim}, Lat={lat_dim})')
    ax.legend(loc='upper right', framealpha=0.9, ncol=2)
    ax.set_yscale('log')  # Often variance spans orders of magnitude
    
    plt.tight_layout()
    
    # Save figure
    filename = output_path / f'sample_variance_cat{cat_dim}_lat{lat_dim}.pdf'
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process and visualize results')
    parser.add_argument('--plot', action='store_true', help='Generate publication-quality plots')
    parser.add_argument('--output-dir', type=str, default='./figures', help='Output directory for figures')
    args = parser.parse_args()
    
    main()
    
    if args.plot:
        plot_all_configurations(output_dir=args.output_dir)