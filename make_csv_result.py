import csv
import glob
import re
import os

seeds = [12, 41, 95]
a_list = [8, 7, 9]
w_list = [0.5, 1, 1.5]

paper_acc = {
    'False_SBM_PATTERN_MLP': '50.13+-0.00', # MLP and MLP GATED is same
    'True_SBM_PATTERN_GCN': '74.36+-1.59',
    'False_SBM_PATTERN_GCN': '55.22+-0.17',
    'True_SBM_PATTERN_GraphSage': '78.20+-3.06',
    'False_SBM_PATTERN_GraphSage': '81.25+-3.84',
    'True_SBM_PATTERN_GIN': '96.98+-2.18',
    'False_SBM_PATTERN_GIN': '98.25+-0.38',
    'True_SBM_PATTERN_GAT': '90.72+-2.04',
    'False_SBM_PATTERN_GAT': '88.91+-4.48',
    'True_SBM_PATTERN_Monet': '95.52+-3.74',
    'False_SBM_PATTERN_Monet': '97.89+-0.89',
    'True_SBM_PATTERN_GatedGCN': '95.05+-2.80',
    'False_SBM_PATTERN_GatedGCN': '97.24+-1.19',


    'False_SBM_CLUSTER_MLP': '20.97+-0.01',
    'True_SBM_CLUSTER_GCN': '47.82+-4.91',
    'False_SBM_CLUSTER_GCN': '34.85+-0.65',
    'True_SBM_CLUSTER_GraphSage': '44.89+-3.70',
    'False_SBM_CLUSTER_GraphSage': '53.90+-4.12',
    'True_SBM_CLUSTER_GIN': '49.64+-2.09',
    'False_SBM_CLUSTER_GIN': '52.54+-1.09',
    'True_SBM_CLUSTER_GAT': '49.08+-6.47',
    'False_SBM_CLUSTER_GAT': '54.12+-1.21',
    'True_SBM_CLUSTER_MoNet': '45.95+-3.39',
    'False_SBM_CLUSTER_Monet': '39.48+-2.21',
    'True_SBM_CLUSTER_GatedGCN': '54.20+-3.58',
    'False_SBM_CLUSTER_GatedGCN': '50.18+-3.03',

}

d = '.'

cur_dir = os.getcwd()

file_name = []

with open("SBM_test_result.csv", 'w', newline='') as f2:
    header = ["dataset", "model name", "seed", "residual",  "paper accuracy", "test_accuracy", 'smooth']
    csvwriter = csv.writer(f2, delimiter=',')
    csvwriter.writerow(header)

for name in glob.glob('*.csv'):
    if re.match('(\w+)_test_result.csv', name):
        file_name.append(name)
        if re.match('(\d+)_(True|False)_SBM_(CLUSTER|PATTERN)_(a\d+|w\d+)_*', name):
            name_list = name.split("_")
            seed = name_list[0]
            residual = name_list[1]
            dataset = name_list[3]
            smoothing_name = name_list[4]
            model_name = name_list[5]

            paper_dict_name = residual+'_SBM_'+dataset+"_"+model_name

            with open(name) as f:
                csvreader = csv.reader(f,  delimiter=',')
                next(csvreader)
                for x in csvreader:
                    test_acc = float(x[0])
                    paper_acc[''.join(name_list[:6])] = test_acc
                with open("SBM_test_result.csv", 'a', newline='') as f2:
                    csvwriter = csv.writer(f2, delimiter=',')
                    csvwriter.writerow([dataset, model_name, seed, residual, paper_acc[paper_dict_name], test_acc, smoothing_name])



def read_csv(cur_dir, fileName):
    return

