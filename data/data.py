"""
    File to load dataset based on user control from main file
"""
#from data.superpixels import SuperPixDataset
#from data.molecules import MoleculeDataset
#from data.TUs import TUsDataset
from data.SBMs import SBMsDataset, SmoothedSBMsDataset, SBMsSampleDataset
from data.wiki import WikiDataset
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

    SAMPLE_SBMDATASET = ['SBM_CLUSTER_SAMPLE', 'SBM_CLUSTER_SAMPLE']
    if DATASET_NAME in SAMPLE_SBMDATASET:
        return SBMsSampleDataset(DATASET_NAME)

    SAMPLE_SBMDATASET = ['SBM_CLUSTER_SAMPLE_SMOOTH', 'SBM_CLUSTER_SAMPLE_SMOOTH']
    if DATASET_NAME in SAMPLE_SBMDATASET:
        return SmoothedSBMsDataset(DATASET_NAME)

    if re.match('(SBM_CLUSTER_a\d)', DATASET_NAME) or re.match('(SBM_PATTERN_a\d)', DATASET_NAME):
        return SmoothedSBMsDataset(DATASET_NAME)

    if re.match('(SBM_CLUSTER_w\d)', DATASET_NAME) or re.match('(SBM_PATTERN_w\d)', DATASET_NAME):
        return SmoothedSBMsDataset(DATASET_NAME)

    # handling for TSP dataset
    if DATASET_NAME == 'TSP':
        return TSPDataset(DATASET_NAME)

    # handling for the CITATIONGRAPHS Datasets
    CITATIONGRAPHS_DATASETS = ['CORA', 'CITESEER', 'PUBMED']
    if DATASET_NAME in CITATIONGRAPHS_DATASETS: 
        return CitationGraphsDataset(DATASET_NAME)

    if DATASET_NAME in ['chameleon', 'crocodile', 'squirrel']:
        return WikiDataset(DATASET_NAME)
