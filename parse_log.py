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
mean_losses_val = []
mean_losses_train = []

for idx, path in enumerate(JSON_FILES_PATHS):
    
    with open(path) as json_file:
        data = json.load(json_file)
        
        mean_losses_val.append(float(data[CONST.MEAN_LOSS_VAL]))
        mean_losses_train.append(float(data[CONST.MEAN_LOSS_TRAIN]) if idx != 0 else float(data[CONST.MEAN_LOSS_VAL]))
        
# Show the results
x = np.arange(1, len(JSON_FILES_PATHS) + 1, 1)

# Losses
fig_losses = plt.figure(num='Training losses results')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')

for i in range(len(mean_losses_val)):
    if mean_losses_val[i] > 10 or mean_losses_val[i] != mean_losses_val[i]:
        mean_losses_val[i] = 5.0

for i in range(len(mean_losses_train)):
    if mean_losses_train[i] > 10 or mean_losses_train[i] != mean_losses_train[i]:
        mean_losses_train[i] = 5.0

plt.ylim(0, max([max(mean_losses_val), max(mean_losses_train)]) + 0.1)

plt.plot(x, mean_losses_train, 'c-')
plt.plot(x, mean_losses_val, 'b-')

best_epoch = np.argmin(mean_losses_train)
best_value = np.min(mean_losses_train)
print("Mean loss train", "- Epoch", best_epoch + 1, "-", best_value)


best_epoch = np.argmin(mean_losses_val)
best_value = np.min(mean_losses_val)
print("Mean loss val", "- Epoch", best_epoch + 1, "-", best_value)

plt.legend(['Training', 'Validation'], loc='best')
plt.show()
