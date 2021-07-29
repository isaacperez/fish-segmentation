import time
import glob 
import os 

import numpy as np 
import torch 
from torchvision.io import read_image
from torchvision.transforms import Pad
import matplotlib.pyplot as plt 
from torchvision.transforms import Resize

import dataset
import model
import loss
import CONST
import metrics


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
GPU_TO_USE = int(1)
device = torch.device("cuda:" + str(GPU_TO_USE))

MODEL_PATH = "/home/isaac/workspace/2021_Estancia/models/20210720_1521/results/MEAN_IOU.pt"

DO_INFERENCE_OVER_A_DATASET = False
DATASET_NAME = CONST.VAL_FOLDER_NAME

INFERENCE_OVER_A_FOLDER = True
FOLDER_PATH = "/home/isaac/workspace/2021_Estancia/datasets/CCMAR/test_images_CCMAR/rgb/"

INFERENCE_OVER_AN_IMAGE = False
IMAGE_PATH = "/home/isaac/workspace/2021_Estancia/datasets/DEEPFISH_AND_LUDERICK_DATASET/TEST/images/04C2_Luderick_6_000000.jpg"

OUTPUT_FOLDER = "./"

MODEL_FINAL_SCALE_REDUCTION = 5

# -----------------------------------------------------------------------------
# Module functions: INFERENCE
# -----------------------------------------------------------------------------
def pad(image):
    """Calculate the padding need for the image to be evenly divisible by 2 
       MODEL_FINAL_SCALE_REDUCTION times"""
    height = image.shape[1]
    width = image.shape[2]

    final_height = int(np.ceil(height / 2 ** MODEL_FINAL_SCALE_REDUCTION) * 2 ** MODEL_FINAL_SCALE_REDUCTION)
    final_width = int(np.ceil(width / 2 ** MODEL_FINAL_SCALE_REDUCTION) * 2 ** MODEL_FINAL_SCALE_REDUCTION)
    
    h_padding = (final_width - width) / 2
    v_padding = (final_height - height) / 2

    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5

    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))

    padding_op = Pad(padding)
  
    image = padding_op(image)

    return image, padding


def inference_over_image(model_loaded, img_path):
    """Do the inference over the image with the model given.

    Args:
      - model_loaded: Tensorflow model or None.
      - image_path: a str path or None.
    """
    # Read the image
    image = read_image(img_path)

    # Remove the alpha channel from the image if it has one
    if image.shape[0] == 4:
        image = image[:3, :, :]

    # Check if the image needs some padding
    image, padding_used = pad(image)

    # Prepare the model
    model_loaded.eval()

    # Prepare the softmax
    softmax = torch.nn.Softmax(dim=1)

    # Do the inference over the dataset
    with torch.no_grad():
            
        # Move to GPU
        image = image.to(device)
        
        # Cast to float, normalize and add the batch dimension
        image = image.float() / 255.0
        image = torch.unsqueeze(image, 0)

        # Process the image with the model
        model_output = model_loaded(image)

        # Get the final and probabilities output
        probabilities_output = softmax(model_output)[:, 1, :, :]
        model_output = torch.argmax(model_output, dim=1)
        
    # Reset model status
    model_loaded.train()

    # Remove the padding
    (l_pad, t_pad, r_pad, b_pad) = padding_used
    
    height = model_output.shape[1]
    width = model_output.shape[2]

    model_output = model_output[:, t_pad:height-b_pad, l_pad:width-r_pad]
    probabilities_output = probabilities_output[:, t_pad:height-b_pad, l_pad:width-r_pad]
    
    return model_output, probabilities_output


def inference_over_folder(model_loaded, folder_path, output_folder):
    """Do the inference over all the images in the folder with the model given.

    Args:
      - model_loaded: Tensorflow model or None.
      - folder_path: folder with images.
    """
    processing_time = []
    
    if folder_path[-1] != os.sep:
        folder_path += os.sep

    if output_folder[-1] != os.sep:
        output_folder += os.sep

    path_list = []
    for ext in CONST.IMG_EXTENSION:
        path_list += glob.glob(folder_path + "*" + ext) 

    for idx, img_path in enumerate(path_list):

        print("Processing image", idx + 1, "of", len(path_list), end="\r")

        # Get its results
        start = time.process_time()
        model_output, probabilities_output = \
            inference_over_image(model_loaded, img_path)
        end = time.process_time()

        # Save the results
        processing_time.append(end - start)

        # Move to the CPU and to NumPy array
        model_output = model_output.cpu().numpy()
        probabilities_output = probabilities_output.cpu().numpy()

        # Remove batch dimension
        model_output = model_output[0, :, :]
        probabilities_output = probabilities_output[0, :, :]

        #plt.imsave(output_folder + img_path.split(os.sep)[-1], model_output.astype(np.uint8) * 255, cmap="gray")
        plt.imsave("/home/isaac/workspace/2021_Estancia/datasets/CCMAR/test_images_CCMAR/segmentation (threshold 50)/" + img_path.split(os.sep)[-1], model_output.astype(np.uint8) * 255, cmap="gray")
        plt.imsave("/home/isaac/workspace/2021_Estancia/datasets/CCMAR/test_images_CCMAR/probabilities/" + img_path.split(os.sep)[-1], (probabilities_output * 255).astype(np.uint8), cmap="gray")
        
    print("")

    # Remove the first value
    if len(processing_time) > 1:
        processing_time = processing_time[1:]

    # Calculate the average time
    average_processing_time = sum(processing_time) / len(processing_time)
    
    print("Average processing time by image (sec.):", average_processing_time)

    return average_processing_time


def inference_over_dataset(name_of_dataset, model_loaded):
    """Get the evaluation results for the model in the dataset"""
    # Load the dataset
    current_val_set = dataset.CustomImageDataset(name_of_dataset, do_data_augmentation=False)
    val_dataset = torch.utils.data.DataLoader(current_val_set, batch_size=1, shuffle=False, num_workers=6)

    # Prepare the loss class
    loss_class = loss.Loss()
    loss_class.to(device)

    # Prepare the model
    model_loaded.eval()

    # Do the inference over the dataset
    with torch.no_grad():

        # Loop over the dataset
        metrics_handler = metrics.Metrics()
        idx_batch = 0
        for img, expected_output in val_dataset:
            
            # Move to GPU
            img, expected_output = img.to(device), expected_output.to(device)

            # Get results
            model_output = model_loaded(img)
            
            # Get the loss
            loss_value = loss_class(model_output, expected_output)

            # Update results
            metrics_handler.updateStatus(torch.argmax(model_output, 1)[0, :, :], 
                                        expected_output[0, :, :], loss_value.item())

            #### DEBUG
            # Get the final and probabilities output
            softmax = torch.nn.Softmax(dim=1)
            prob_output = softmax(model_output)[:, 1, :, :]
            model_output = torch.argmax(model_output, dim=1)
            
            image = img

            # Move to the CPU and to NumPy array
            model_output = model_output.cpu().numpy()
            prob_output = prob_output.cpu().numpy()
            expected_output = expected_output.cpu().numpy()

            # Remove batch dimension
            model_output = model_output[0, :, :]
            prob_output = prob_output[0, :, :]
            expected_output = expected_output[0, :, :]
            image = image[0, :, :].cpu()

            

            # In Pytorch, the first dimension is used for channels
            image = image.permute(1, 2, 0)

            # Convert to NumPy array
            image = image.numpy()
            
            # Show the results
            plt.figure(1)
            plt.title("Original image")
            plt.imshow(image)
    
            plt.figure(2)
            plt.title("Model output")
            plt.imshow(model_output)

            plt.figure(3)
            plt.title("Probability output")
            plt.imshow(prob_output)

            plt.figure(4)
            plt.title("Expected output")
            plt.imshow(expected_output)

            # Calculate the final evaluation values
            print(metrics_handler.calculateMetrics())

            plt.show()
            exit()
            #### DEBUG

            print(str(int(idx_batch + 1)) + " batches processed of " + str(len(val_dataset)))
            idx_batch += 1

    # Calculate the final evaluation values
    evaluation_results = metrics_handler.calculateMetrics()

    # Reset model status
    model_loaded.train()

    return evaluation_results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Load the model
    model_loaded = torch.load(MODEL_PATH, map_location=torch.device(device))
    
    # Inference over a dataset
    if DO_INFERENCE_OVER_A_DATASET:
        print(inference_over_dataset(DATASET_NAME, model_loaded))    

    # Inference over a folder
    if INFERENCE_OVER_A_FOLDER:
        print(inference_over_folder(model_loaded, FOLDER_PATH, OUTPUT_FOLDER))

    # Inference over a image
    if INFERENCE_OVER_AN_IMAGE:
        model_output, prob_output = inference_over_image(model_loaded, IMAGE_PATH)

        # Move to the CPU and to NumPy array
        model_output = model_output.cpu().numpy()
        prob_output = prob_output.cpu().numpy()

        # Remove batch dimension
        model_output = model_output[0, :, :]
        prob_output = prob_output[0, :, :]

        # Read the original image
        image = read_image(IMAGE_PATH)

        # Remove the alpha channel from the image if it has one
        if image.shape[0] == 4:
            image = image[:3, :, :]

        # In Pytorch, the first dimension is used for channels
        image = image.permute(1, 2, 0)

        # Convert to NumPy array
        image = image.numpy()
        
        # Blend both images
        model_output_3_channels = np.expand_dims(model_output, axis=-1)
        model_output_3_channels = np.concatenate([model_output_3_channels, model_output_3_channels, model_output_3_channels], axis=-1)
        
        blend_factor = 0.5
        blended_image = blend_factor * image.astype(np.float32) / 255.0 + (1 - blend_factor) * model_output_3_channels
        blended_image = (blended_image * 255).astype(np.uint8)

        # Show the results
        plt.figure(1)
        plt.title("Original image")
        plt.imshow(image)
  
        plt.figure(2)
        plt.title("Model output")
        plt.imshow(model_output)

        plt.figure(3)
        plt.title("Probability output")
        plt.imshow(prob_output)

        plt.figure(4)
        plt.title("Blended image")
        plt.imshow(blended_image)
        
        plt.show()

        # Save the output if a folder has been given
        if OUTPUT_FOLDER:
            plt.imsave(OUTPUT_FOLDER + IMAGE_PATH.split(os.sep)[-1], model_output.astype(np.uint8) * 255, cmap="gray")
