import csv
import glob
import re
import os
from datetime import datetime
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")

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

data_dir = './result/'
file_name = []

with open(f"{data_dir}/SBM_test_result_{timestampStr}.csv", 'w', newline='') as f2:
    header = ["dataset", "model name", "seed", "residual",  "paper accuracy", "test_accuracy", 'how',
              'lb_delta', 'ub_delta', 'middle_dim', 'bottleneck']
    csvwriter = csv.writer(f2, delimiter=',')
    csvwriter.writerow(header)

for name in glob.glob(f'{data_dir}/*.csv'):
    if name is None or name == '':
        continue
    #name = str(name.split("\\")[1]).replace("./",'')
    name = str(name.split("/")[2]).replace("./",'')
    print(name, 0)
    if re.match('[a-zA-Z0-9_-]+_test_result.csv', name):
        file_name.append(name)
        print(name, 1)
        if re.match('(\d+)_(True|False)_SBM_(CLUSTER|PATTERN)_(a\d+|w\d+)_*', name):
            print(name, 2)
            name_list = name.split("_")
            seed = name_list[0]
            residual = name_list[1]
            dataset = name_list[3]
            smoothing_name = name_list[4]
            model_name = name_list[5]

            paper_dict_name = residual+'_SBM_'+dataset+"_"+model_name

            with open(data_dir+"/"+name, 'r') as f:
                csvreader = csv.reader(f,  delimiter=',')
                next(csvreader)
                for x in csvreader:
                    test_acc = float(x[0])
                    paper_acc[''.join(name_list[:6])] = test_acc
                with open(f"{data_dir}/SBM_test_result_{timestampStr}.csv", 'a', newline='') as f2:
                    csvwriter = csv.writer(f2, delimiter=',')
                    csvwriter.writerow([dataset, model_name, seed, residual, paper_acc[paper_dict_name], test_acc, smoothing_name])
        elif re.match('(\d+)_(True|False)_SBM_(CLUSTER|PATTERN)_(SMOOTH)_(\w+)_(\w+)_(\d+)*', name):
                print(name, 3)
                name_list = name.split("_")
                seed = name_list[0]
                residual = name_list[1]
                dataset = name_list[3]

                model_name = name_list[5]
                how_residual = name_list[6]
                lb_delta = name_list[8]
                ub_delta = name_list[9]
                lb_delta = "0."+lb_delta.replace('lb','')
                ub_delta = "0."+ ub_delta.replace('ub','')
                middle_dim = name_list[10].replace('md', '')
                bottleneck = name_list[11]

                paper_dict_name = residual + '_SBM_' + dataset + "_" + model_name

                with open(data_dir+"/"+name, 'r') as f:
                    csvreader = csv.reader(f, delimiter=',')
                    next(csvreader)
                    for x in csvreader:
                        test_acc = float(x[0])
                        paper_acc[''.join(name_list[:6])] = test_acc
                    with open(f"{data_dir}/SBM_test_result_{timestampStr}.csv", 'a', newline='') as f2:
                        csvwriter = csv.writer(f2, delimiter=',')
                        csvwriter.writerow(
                            [dataset, model_name, seed, residual, paper_acc[paper_dict_name], test_acc, how_residual, lb_delta, ub_delta, middle_dim, bottleneck])

# {params["seed"]}_{str(net_params["residual"])}_{DATASET_NAME}_{MODEL_NAME}_{net_params["how_residual"]}_rk{net_params["rki"]}_lb{str(lb_delta).split(".")[1]}_ub{str(ub_delta).split(".")[1]}_md{net_params["middle_dim"]}_{net_params["bottleneck"]}_{timestampStr}_test_result.csv'
# ex) 41_True_SBM_CLUSTER_SMOOTH_GIN_baseline_rk1_lb0_ub2_md32_True
#     0  1    2    3       4      5    6       7   8   9  10    11
