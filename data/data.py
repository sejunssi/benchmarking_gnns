"""
    File to load dataset based on user control from main file
"""
#from data.superpixels import SuperPixDataset
#from data.molecules import MoleculeDataset
#from data.TUs import TUsDataset
from data.SBMs import SBMsDataset, SmoothedSBMsDataset
#from data.TSP import TSPDataset
#from data.CitationGraphs import CitationGraphsDataset
import re

def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME == 'MNIST' or DATASET_NAME == 'CIFAR10':
        return SuperPixDataset(DATASET_NAME)
    
    # handling for (ZINC) molecule dataset
    if DATASET_NAME == 'ZINC':
        return MoleculeDataset(DATASET_NAME)

    # handling for the TU Datasets
    TU_DATASETS = ['COLLAB', 'ENZYMES', 'DD', 'PROTEINS_full']
    if DATASET_NAME in TU_DATASETS: 
        return TUsDataset(DATASET_NAME)

    # handling for SBM datasets
    SBM_DATASETS = ['SBM_CLUSTER', 'SBM_PATTERN']
    if DATASET_NAME in SBM_DATASETS: 
        return SBMsDataset(DATASET_NAME)

    # a = [1,2,3,4,8]
    # SMOOTHED_SBM_CLUSTER_DATASETS = [f'SBM_CLUSTER_a{x}' for x in a]
    # SMOOTHED_SBM_PATTERN_DATASETS = [f'SBM_PATTERN_a{x}' for x in a]
    if DATASET_NAME in re.match('(SBM_CLUSTER_a\d)', DATASET_NAME) or DATASET_NAME in re.match('(SBM_PATTERN_a\d)', DATASET_NAME):
        return SmoothedSBMsDataset(DATASET_NAME)
    if DATASET_NAME in ['SBM_CLUSTER_smoothing', 'SBM_PATTERN_smoothing']:
        return SmoothedSBMsDataset(DATASET_NAME)

    # handling for TSP dataset
    if DATASET_NAME == 'TSP':
        return TSPDataset(DATASET_NAME)

    # handling for the CITATIONGRAPHS Datasets
    CITATIONGRAPHS_DATASETS = ['CORA', 'CITESEER', 'PUBMED']
    if DATASET_NAME in CITATIONGRAPHS_DATASETS: 
        return CitationGraphsDataset(DATASET_NAME)
