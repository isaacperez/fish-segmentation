import logging
import os
from time import gmtime, strftime
import glob 
import shutil
import json

import numpy as np 

import torch
import torch.optim as optim
from torch.utils.data import Dataset
from pytorch_model_summary import summary

import dataset
import loss
import metrics
import model
import CONST

if __name__ == "__main__":
    # Construct the model
    network = model.Unet()
    print(summary(network, torch.zeros((1, 3, CONST.HEIGHT_TRAIN, CONST.WIDTH_TRAIN)), show_input=True))


# -----------------------------------------------------------------------------
# Hiperparams
# -----------------------------------------------------------------------------
LEARNING_RATE = 0.001
BATCH_SIZE = int(8)
NUM_EPOCH = int(1000)
GPU = int(0)
MAX_NORM = 0.1

PATH_PREVIOUS_CHECKPOINT = None

if __name__ == "__main__":
    #optimizer = optim.SGD(network.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.00001)
    optimizer = optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=0.00001)

if not torch.cuda.is_available():
    print("Error no cuda device avaiable.")
    exit()

device = torch.device("cuda:" + str(GPU))

if PATH_PREVIOUS_CHECKPOINT:
    network = torch.load(PATH_PREVIOUS_CHECKPOINT)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Where the training results folder are going to be build
BASE_PATH = "../models/"

# Directory name where all the results are going to be saved in the output directory
RESULTS_DIR_NAME = "results"

# Name for the log file
LOG_FILE_NAME = "log.log"

# Make the logger
logger = logging.getLogger()
logger.setLevel('INFO')

# create console handler and set level to debug
ch = logging.StreamHandler()

ch.setLevel('INFO')

# create formatter
formatter = logging.Formatter('[%(asctime)s][%(levelname)s] - %(message)s',
                              '%d-%m-%y %H:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
if (logger.hasHandlers()):
    logger.handlers.clear()
logger.addHandler(ch)


# -----------------------------------------------------------------------------
# Module functions: UTILS
# -----------------------------------------------------------------------------
def createOutputDirAndFillWithInitialCode():
    """Make a directory with the current time as name and return the path"""
    ouput_directory_path = strftime("%Y%m%d_%H%M", gmtime())

    if not os.path.exists(BASE_PATH + ouput_directory_path):
        
        # Create the directories
        os.makedirs(BASE_PATH + ouput_directory_path)
        os.makedirs(BASE_PATH + ouput_directory_path + os.sep + RESULTS_DIR_NAME)
        output_directory_path = BASE_PATH + ouput_directory_path + os.sep + RESULTS_DIR_NAME + os.sep

        # Create the log file
        fh = logging.FileHandler(BASE_PATH + ouput_directory_path
                                 + os.sep + LOG_FILE_NAME)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Copy all python files
        python_files_to_copy = \
        glob.glob(__file__[:-len(os.path.basename(__file__))] + "*.py")

        for file in python_files_to_copy:
            shutil.copy(file, BASE_PATH + ouput_directory_path + os.sep)

        return output_directory_path

    else:
        logger.error("Output directory: " + BASE_PATH
                     + ouput_directory_path + " already exists")

        exit()


def updateBestEvaluationResultsAndSaveTheModel(evaluation_results, best_evaluation_results, model, output_directory_path, current_epoch):
    """Check whether the current evaluation results improves the best evaluation results"""

    for key in best_evaluation_results.keys():
        
        if "LOSS" in key and evaluation_results[key] <= best_evaluation_results[key]:
            logger.info((key + " has been improved: ").ljust(38) + str(evaluation_results[key]).ljust(6) + " < " + str(best_evaluation_results[key]).ljust(6))

            best_evaluation_results[key] = evaluation_results[key]
            
            # Save the model
            torch.save(model, output_directory_path + key + ".pt")

        elif "LOSS" not in key and evaluation_results[key] >= best_evaluation_results[key]:
            logger.info((key + " has been improved: ").ljust(38) + str(evaluation_results[key]).ljust(6) + " > " + str(best_evaluation_results[key]).ljust(6))
            best_evaluation_results[key] = evaluation_results[key]
 
            # Save the model
            torch.save(model, output_directory_path + key + ".pt")

    current_epoch_results = {}
    for key in evaluation_results.keys():
        current_epoch_results[key] = str(evaluation_results[key])

    # Save the results for the current epoch
    with open(output_directory_path + str(current_epoch) + '.json', 'w') as f:
        json.dump(current_epoch_results, f)
        
    return best_evaluation_results


def evaluateModel(dataset, model):
    """Calculate the loss and all the metrics for the model over the dataset"""
    
    logger.info("Begining evaluation...")

    # Prepare the model
    model.eval()

    # Prepare the loss class
    loss_class = loss.Loss()
    loss_class.to(device)

    with torch.no_grad():

        # Loop over the dataset
        metrics_handler = metrics.Metrics()
        idx_batch = 0
        for img, expected_output in dataset:
            
            # Move to GPU
            img, expected_output = img.to(device), expected_output.to(device)

            # Get results
            model_output = model(img)

            # Get the loss
            loss_value = loss_class(model_output, expected_output)

            # Update results
            metrics_handler.updateStatus(torch.argmax(model_output, 1)[0, :, :], expected_output[0, :, :], loss_value.item())

            logger.info(str(int(idx_batch + 1)) + " batches processed of " + str(len(dataset)))
            idx_batch += 1

    # Calculate the final evaluation values
    evaluation_results = metrics_handler.calculateMetrics()

    # Reset model status
    model.train()

    return evaluation_results


# -----------------------------------------------------------------------------
# Module functions: TRAIN
# -----------------------------------------------------------------------------
def train():
    """Launch the training with the hiperparams defined"""

    # Move the model to the GPU
    network.to(device)
    network.train()

    # Prepare the output dir
    output_directory_path = createOutputDirAndFillWithInitialCode()

    # Prepare the loss class
    loss_class = loss.Loss()
    loss_class.to(device)

    # Build the datasets
    current_train_set = dataset.CustomImageDataset(CONST.TRAIN_FOLDER_NAME, do_data_augmentation=True)
    train_dataset = torch.utils.data.DataLoader(current_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=6)
    batches_in_the_train_dataset = len(train_dataset)

    current_val_set = dataset.CustomImageDataset(CONST.VAL_FOLDER_NAME, do_data_augmentation=False)
    val_dataset = torch.utils.data.DataLoader(current_val_set, batch_size=1, shuffle=False, num_workers=6)

    # First evaluation
    best_evaluation_results = metrics.DEFAULT_EVALUATION_RESULTS

    # Begining training
    logger.info("Begining training...")

    # Training loop
    for epoch in range(NUM_EPOCH):

        it_train = 0
        accumulated_loss = 0.0

        for img, expected_output in train_dataset:
            
            # Move to GPU
            img, expected_output = img.to(device), expected_output.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(img)
            loss_value = loss_class(outputs, expected_output)
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=MAX_NORM, norm_type=2.0)
            optimizer.step()

            loss_value = loss_value.item()
            
            # Show the results of the current iteration
            logger.info("[EPOCH " + str(epoch).ljust(6) + " / "
                        + str(NUM_EPOCH) + "][TRAIN It: "
                        + str(it_train).ljust(6) + " / " + str(batches_in_the_train_dataset)  +"]: "
                        + str(np.round(loss_value, 4)).ljust(6))

            accumulated_loss += loss_value

            it_train += 1

        # Do the validation after the training dataset
        logger.info("[EPOCH " + str(epoch).ljust(6) + " of "
                    + str(NUM_EPOCH) + "][VALIDATION]")

        # Validate the model
        evaluation_results = evaluateModel(val_dataset, network)

        # Save the training loss
        evaluation_results[CONST.MEAN_LOSS_TRAIN] = np.round(accumulated_loss / float(it_train), 4)

        # Shot the results for the current epoch
        for key in best_evaluation_results.keys():
            logger.info("[EPOCH " + str(epoch).ljust(6) + " of "
                    + str(NUM_EPOCH) + "] " \
                    + (key + ": ").ljust(18) + str(evaluation_results[key]).ljust(6))

        # Look for improvements
        best_evaluation_results  = updateBestEvaluationResultsAndSaveTheModel(
            evaluation_results=evaluation_results, 
            best_evaluation_results=best_evaluation_results, 
            model=network, 
            output_directory_path=output_directory_path, 
            current_epoch=epoch+1)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Begin the training
    train()