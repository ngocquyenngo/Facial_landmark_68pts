import torch
from pca import PCAUtility
import numpy as np


class ASMLoss:

    def calculate_landmark_ASM_assisted_loss(self, landmark_pr, landmark_gt, current_epoch, num_epochs):
        """
        :param landmark_pr:
        :param landmark_gt:
        :param current_epoch:
        :return:
        """
        # calculating ASMLoss weight:
        alpha = 0
        if current_epoch < num_epochs/5: alpha = 2
        elif num_epochs/5 <= current_epoch < 2*num_epochs/5: alpha = 1
        elif 2*num_epochs/5 <= current_epoch < 3*num_epochs/5: alpha = 0.5

        # creating the ASM-ground truth
        landmark_gt_asm = self._calculate_asm(input_tensor=landmark_gt)

        # calculating ASMLoss
        asm_loss = torch.mean(torch.square((torch.from_numpy(landmark_gt_asm).cuda()) - landmark_pr))

        # calculating MSELoss
        mse_loss = torch.mean(torch.square(landmark_gt - landmark_pr))

        # calculating total loss
        return mse_loss + alpha * asm_loss

    def _calculate_asm(self, input_tensor):
        pca_utility = PCAUtility()
 
        eigenvalues, eigenvectors, meanvector = pca_utility.load_pca( pca_percentages=90)
        input_tensor = input_tensor.cpu()

        input_tensor = input_tensor.detach().numpy()
        input_vector = np.array(input_tensor)
        out_asm_vector = []
        batch_size = input_vector.shape[0]
        for i in range(batch_size):
            b_vector_p = self._calculate_b_vector(input_vector[i], eigenvalues, eigenvectors, meanvector)
            out_asm_vector.append(meanvector + np.dot(eigenvectors, b_vector_p))

        out_asm_vector = np.array(out_asm_vector)
        return out_asm_vector

    def _calculate_b_vector(self, predicted_vector, eigenvalues, eigenvectors, meanvector):
        b_vector = np.dot(eigenvectors.T, predicted_vector - meanvector)
        i = 0
        for b_item in b_vector:
            lambda_i_sqr = 3 * np.sqrt(eigenvalues[i])
            if b_item > 0:
                b_item = min(b_item, lambda_i_sqr)
            else:
                b_item = max(b_item, -1 * lambda_i_sqr)
            b_vector[i] = b_item
            i += 1

        return b_vector