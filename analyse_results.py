import json
from pathlib import Path
import numpy as np
json_dir = Path("results/results_reinmax_cv_full")
size = "8x4"
#json_dir = Path("results_8x4")
all_configs = {}

for json_path in json_dir.glob("*.json"):
    with open(json_path) as f:
        all_configs[json_path.stem] = json.load(f)
print(len(all_configs.keys()))
# average the 10 seeds
avg_results = {}
for key, value in all_configs.items():
    # filter
    if size in key:
        train_losses = np.zeros(len(value))
        test_losses = np.zeros(len(value))
        index = 0
        for k, v in value.items():
        #print(v)
            train_losses[index] = v[0]
            test_losses[index] = v[1]
            index += 1
        avg_results[key] = [train_losses, test_losses]
print(avg_results, len(avg_results))

# find min
min_loss = 3000
for key, value in avg_results.items():
    train_losses = value[0]
    test_losses = value[1]
    if train_losses.mean() < min_loss:
        min_loss = train_losses.mean()

        train_std = train_losses.std()
        test_std = test_losses.std()
        min_train_loss = train_losses.mean()
        min_test_loss = test_losses.mean()
        min_key = key

print(min_key)
print(min_train_loss, train_std)
print(min_test_loss, test_std)

#reinmax_cv-160-RAdam-10x30-0.7-0.0005-1.0
#reinmax_cv-160-RAdam-64x8-0.7-0.0005-1.5
#reinmax_cv-160-RAdam-16x12-0.7-0.0005-1.5
#reinmax_cv-160-RAdam-8x16-1.1-0.0005-1.5
#reinmax_cv-160-Adam-4x24-1.3-0.0005-1.5
#reinmax_cv-160-Adam-8x4-1.0-0.0005-1.5