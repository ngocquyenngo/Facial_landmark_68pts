from pca  import PCAUtility
from dataset import landmark_arr

pca_utility = PCAUtility()
pca_utility.calc_pca(landmark_arr, pca_percentages=90)