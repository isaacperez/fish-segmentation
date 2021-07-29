import matplotlib.pyplot as plt
import json
import numpy as np
import glob
import os 

import CONST

# Get the files paths
LOG_FILE_PATH = "./log.log"
JSON_FILES_PATHS = glob.glob("./results" + os.sep + "*.json")

# Sort the JSON files by its epoch
JSON_FILES_EPOCH_IDX = [int(path.split("/")[-1][:-5]) for path in JSON_FILES_PATHS]
JSON_FILES_PATHS = [path for _, path in sorted(zip(JSON_FILES_EPOCH_IDX, JSON_FILES_PATHS))]

# Read the data
mean_metrics = {}
metrics = {}

for idx, path in enumerate(JSON_FILES_PATHS):
    
    with open(path) as json_file:
        data = json.load(json_file)
        
        for key in data.keys():
            if "LOSS" in key:
                continue
            if "MEAN" in key:
                if key in mean_metrics:
                    mean_metrics[key].append(float(data[key]))
                else:
                    mean_metrics[key] = [float(data[key])]
        
# Show the results
x = np.arange(1, len(JSON_FILES_PATHS) + 1, 1)

# Metrics
fig_metrics = plt.figure(num='Mean metrics results')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Value')

for idx, key in enumerate(mean_metrics.keys()):
    plt.plot(x, mean_metrics[key], 'C' + str(idx))

    best_epoch = np.argmax(mean_metrics[key])
    best_value = np.max(mean_metrics[key])
    print(key, "- Epoch", best_epoch + 1, "-", best_value)

plt.legend(mean_metrics.keys(), loc='best')
plt.ylim(0, 1 + 0.1)


plt.show()


