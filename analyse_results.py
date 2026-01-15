import json
from pathlib import Path
import numpy as np

json_dir = Path("results/results_reinmax_cv_full")
sizes = ["8x4", "4x24", "8x16", "16x12", "64x8", "10x30"]  # Add all your sizes here

all_configs = {}

for json_path in json_dir.glob("*.json"):
    with open(json_path) as f:
        all_configs[json_path.stem] = json.load(f)

print(len(all_configs.keys()))

# Process each size
for size in sizes:
    print(f"\n--- Processing size: {size} ---")
    
    # average the 10 seeds
    avg_results = {}
    for key, value in all_configs.items():
        # filter
        if size in key:
            train_losses = np.zeros(len(value))
            test_losses = np.zeros(len(value))
            index = 0
            for k, v in value.items():
                train_losses[index] = v[0]
                test_losses[index] = v[1]
                index += 1
            avg_results[key] = [train_losses, test_losses]
    
    print(f"Found {len(avg_results)} configs for size {size}")
    
    # find min
    min_loss = 3000
    for key, value in avg_results.items():
        train_losses = value[0]
        test_losses = value[1]
        if train_losses.mean() < min_loss:
            min_loss = train_losses.mean()
            train_ste = train_losses.std() / np.sqrt(10)
            test_ste = test_losses.std() / np.sqrt(10)
            min_train_loss = train_losses.mean()
            min_test_loss = test_losses.mean()
            min_key = key
    
    print(min_key)
    print(f"Train: {min_train_loss:.4f} ± {train_ste:.4f}")
    print(f"Test:  {min_test_loss:.4f} ± {test_ste:.4f}")

    # Save results to file
    with open(f"results_{size}.txt", "w") as f:
        f.write(f"{min_key}\n")
        f.write(f"{min_train_loss:.4f}, {train_ste:.4f}\n")
        f.write(f"{min_test_loss:.4f}, {test_ste:.4f}\n")