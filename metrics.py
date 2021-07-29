import torch
import numpy as np

import CONST


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DEFAULT_EVALUATION_RESULTS = {
    CONST.MEAN_LOSS_TRAIN: float("inf"),
    CONST.MEAN_LOSS_VAL: float("inf"), 
    CONST.MEAN_ACCURACY: 0.0,
    CONST.MEAN_RECALL: 0.0,
    CONST.MEAN_F1_SCORE: 0.0,
    CONST.MEAN_PRECISION: 0.0,
    CONST.MEAN_IOU: 0.0}


# -----------------------------------------------------------------------------
# Module functions: METRICS
# -----------------------------------------------------------------------------
class Metrics():

    def __init__(self):

        self.accumulated_loss = torch.tensor(0.0)
        self.num_examples = 0
        self.TP = [torch.tensor(0.0) for _ in CONST.CLASSES]
        self.TN = [torch.tensor(0.0) for _ in CONST.CLASSES]
        self.FP = [torch.tensor(0.0) for _ in CONST.CLASSES]
        self.FN = [torch.tensor(0.0) for _ in CONST.CLASSES]


    def updateStatus(self, ordinal_model_output, ordinal_expected_output, loss_value):

        assert ordinal_model_output.shape == ordinal_expected_output.shape

        self.accumulated_loss += loss_value
        self.num_examples += 1

        # Calculate the primary metrics
        for idx, _ in enumerate(CONST.CLASSES):

            current_class_expected_idx = torch.eq(ordinal_expected_output, idx)

            self.TP[idx] += torch.count_nonzero((torch.logical_and(
                torch.eq(ordinal_model_output, ordinal_expected_output),
                current_class_expected_idx).type(torch.FloatTensor)))
       
            self.FP[idx] += torch.count_nonzero((torch.logical_and(
                torch.eq(ordinal_model_output, idx),
                torch.logical_not(current_class_expected_idx))).type(torch.FloatTensor))
    
            self.FN[idx] += torch.count_nonzero((torch.logical_and(
                torch.ne(ordinal_model_output, idx),
                current_class_expected_idx)).type(torch.FloatTensor))

            self.TN[idx] += torch.count_nonzero((torch.logical_and(
                torch.ne(ordinal_model_output, idx),
                torch.logical_not(current_class_expected_idx)).type(torch.FloatTensor)))

    def calculateMetrics(self):
        
        # Calculate each metric
        mean_loss = self.accumulated_loss / self.num_examples
        
        accuracy = [torch.tensor(0.0) for _ in CONST.CLASSES]
        recall = [torch.tensor(0.0) for _ in CONST.CLASSES]
        precision = [torch.tensor(0.0) for _ in CONST.CLASSES]
        f1_score = [torch.tensor(0.0) for _ in CONST.CLASSES]
        IOU = [torch.tensor(0.0) for _ in CONST.CLASSES]

        for idx, _ in enumerate(CONST.CLASSES):
            
            accuracy[idx] = (self.TP[idx] + self.TN[idx]) / \
                (self.TP[idx] + self.TN[idx] + self.FP[idx]  + self.FN[idx])

            if (self.TP[idx] + self.FN[idx]) > 0.0:
                recall[idx] =  self.TP[idx] / (self.TP[idx] + self.FN[idx])

            if (self.TP[idx] + self.FP[idx]) > 0.0:
                precision[idx] = self.TP[idx] / (self.TP[idx] + self.FP[idx]) 

            if not(recall[idx] == 0.0  and precision[idx] == 0.0):
                f1_score[idx] = (2 * precision[idx] * recall[idx]) / (precision[idx] + recall[idx])

            if (self.TP[idx] + self.FP[idx] + self.FN[idx]) > 0.0:
                IOU[idx] = self.TP[idx] / (self.TP[idx] + self.FP[idx] + self.FN[idx])


        mean_accuracy = sum(accuracy) / float(len(CONST.CLASSES))
        mean_recall = sum(recall) / float(len(CONST.CLASSES))
        mean_precision = sum(precision) / float(len(CONST.CLASSES))
        mean_f1_score = sum(f1_score) / float(len(CONST.CLASSES))
        mean_IoU_score = sum(IOU) / float(len(CONST.CLASSES))
        
        # Build the final dict with the results
        evaluation_results = {CONST.MEAN_LOSS_TRAIN: float("inf"),
                CONST.MEAN_LOSS_VAL: np.round(mean_loss.item(), 4), 
                CONST.MEAN_ACCURACY: np.round(mean_accuracy.item(), 4),
                CONST.MEAN_RECALL: np.round(mean_recall.item(), 4),
                CONST.MEAN_F1_SCORE: np.round(mean_f1_score.item(), 4),
                CONST.MEAN_PRECISION: np.round(mean_precision.item(), 4),
                CONST.MEAN_IOU: np.round(mean_IoU_score.item(), 4)}

        for idx, class_name in enumerate(CONST.CLASSES):
            evaluation_results[CONST.ACCURACY + "_" + class_name] = np.round(accuracy[idx].item(), 4)
            evaluation_results[CONST.F1_SCORE + "_" + class_name] = np.round(f1_score[idx].item(), 4)
            evaluation_results[CONST.RECALL + "_" + class_name] = np.round(recall[idx].item(), 4)
            evaluation_results[CONST.PRECISION + "_" + class_name] = np.round(precision[idx].item(), 4)
            evaluation_results[CONST.IOU + "_" + class_name] = np.round(IOU[idx].item(), 4)
        
        print(self.TP, self.FP, self.TN, self.FN)

        return evaluation_results


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_Metrics():
    """Test for Metrics Class"""

    metrics = Metrics()

    ordinal_model_output = torch.ones((2, 3, CONST.HEIGHT, CONST.WIDTH))
    ordinal_expected_output = torch.ones((2, 3, CONST.HEIGHT, CONST.WIDTH))
    loss_value = torch.tensor(0.71)

    metrics.updateStatus(ordinal_model_output, ordinal_expected_output, loss_value)
    metrics.updateStatus(ordinal_model_output, ordinal_expected_output, loss_value)

    evaluation_results = metrics.calculateMetrics()
    for key in evaluation_results.keys():
        print(key, ":", evaluation_results[key])

    print("TP:", metrics.TP)
    print("TN:", metrics.TN)
    print("FP:", metrics.FP)
    print("FN:", metrics.FN)
    print("IOU:", metrics.IOU)



def do_tests():
    """Launch all test avaiable in this module"""
    test_Metrics()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    # Only launch all tests
    do_tests()
