from sklearn.decomposition import PCA
import numpy as np
from numpy import save

class PCAUtility:
    eigenvalues_prefix = "_eigenvalues_"
    eigenvectors_prefix = "_eigenvectors_"
    meanvector_prefix = "_meanvector_"

    def calc_pca(self,landmark_arr, pca_percentages):
        """
        generate and save eigenvalues, eigenvectors, meanvector
        :param pca_percentages: % of eigenvalues that will be used
        :return: generate
        """
        landmark_arr = np.reshape(landmark_arr,(10648, 136))
        norm = np.linalg.norm(landmark_arr)
        lbl_arr = landmark_arr/norm
        reduced_lbl_arr, eigenvalues, eigenvectors = self._func_PCA(lbl_arr, pca_percentages)
        meanvector = np.mean(lbl_arr, axis=0)
        eigenvectors = eigenvectors.T
        # return eigenvalues, eigenvectors, meanvector
        save('/content/drive/Shareddrives/FacialLandmark/input/' + self.eigenvalues_prefix + str(pca_percentages), eigenvalues)
        save('/content/drive/Shareddrives/FacialLandmark/input/' + self.eigenvectors_prefix + str(pca_percentages), eigenvectors)
        save('/content/drive/Shareddrives/FacialLandmark/input/' + self.meanvector_prefix + str(pca_percentages), meanvector)
        
    
    def load_pca(self, pca_percentages):
        eigenvalues = np.load('/content/drive/Shareddrives/FacialLandmark/input/' + self.eigenvalues_prefix + str(pca_percentages)+'.npy')
        eigenvectors = np.load('/content/drive/Shareddrives/FacialLandmark/input/' + self.eigenvectors_prefix + str(pca_percentages)+'.npy')
        meanvector = np.load('/content/drive/Shareddrives/FacialLandmark/input/' + self.meanvector_prefix + str(pca_percentages)+'.npy')
        return eigenvalues, eigenvectors, meanvector

    def _func_PCA(self, input_data, pca_postfix):
        # inputdata = input_data.cpu()
        # inputdata = np.array(inputdata)
        pca = PCA(n_components=pca_postfix /100)
        pca.fit(input_data)
        pca_inputdata = pca.transform(input_data)
        eigenvalues = pca.explained_variance_
        eigenvectors = pca.components_
        return pca_inputdata, eigenvalues, eigenvectors