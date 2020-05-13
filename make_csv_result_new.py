import csv
import glob
import re
import os
from datetime import datetime
from collections import OrderedDict
dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%d-%b-%Y-%H-%M")

wo_acc = {
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

cur_dir = os.getcwd()


# with open(f"{data_dir}/SBM_test_result_{timestampStr}.csv", 'w', newline='') as f2:
#     header = ["dataset", "model name", "seed", "residual",  "bengio accuracy", "test_accuracy", 'how',
#               'lb_delta', 'ub_delta', 'middle_dim', 'bottleneck']
#     csvwriter = csv.writer(f2, delimiter=',')
#     csvwriter.writerow(header)

def read_data(data_dir):
    data_dict_list = []
    for name in glob.glob(f'{data_dir}/*.csv'):
        data_dict = {}
        if name is None or name == '':
            continue
        name = str(name.split("/")[1]).replace("./",'')
        print(name, 0)
        if re.match('[a-zA-Z0-9_-]+_test_result.csv', name):
            print(name, 1)
            if re.match('(\d+)_(True|False)_SBM_(CLUSTER|PATTERN)_(a\d+|w\d+)_*', name):
                print(name, 2)
                name_list = name.split("_")
                seed = name_list[0]
                residual = name_list[1]
                dataset = name_list[3]
                smoothing_name = name_list[4]
                model_name = name_list[5]

                bengio_dict_name = residual+'_SBM_'+dataset+"_"+model_name

                data_dict['dataset'] = dataset
                data_dict['smoothing_name'] = smoothing_name
                data_dict['wo_acc'] = wo_acc
                data_dict['seed'] = seed
                data_dict['model_name'] = model_name

                with open(data_dir+"/"+name, 'r') as f:
                    csvreader = csv.reader(f,  delimiter=',')
                    next(csvreader)
                    for x in csvreader:
                        test_acc = float(x[0])
                        data_dict['wo_acc'] = bengio_dict_name[bengio_dict_name]
                        data_dict['test_acc'] = test_acc
                data_dict_list.append(data_dict)

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

                    bengio_dict_name = residual + '_SBM_' + dataset + "_" + model_name

                    data_dict['dataset'] = model_name
                    data_dict['model'] = model_name
                    data_dict['seed'] = seed
                    data_dict['residual'] = residual
                    data_dict['how_residual'] = how_residual
                    data_dict['lb'] = lb_delta
                    data_dict['ub'] = ub_delta
                    data_dict['middle_dim'] = middle_dim

                    with open(data_dir+"/"+name, 'r') as f:
                        csvreader = csv.reader(f, delimiter=',')
                        next(csvreader)
                        for x in csvreader:
                            test_acc = float(x[0])
                            wo_acc[''.join(name_list[:6])] = test_acc
                            data_dict['wo_acc'] = bengio_dict_name[bengio_dict_name]
                            data_dict['test_acc'] = test_acc
                    data_dict_list.append(data_dict)
        return data_dict_list


new_data_dict_list = read_data('result')
classic_data_dict_list = read_data('result_w')

def get_table_dict(new_data_dict_list, classic_data_dict_list, residual='True'):


    total_data_dict = OrderedDict()
    lb_ub_list = [('0.', '0.2'), ('0.05', '0.15'), ('0.08', '0.12')]
    w_list = ['w05', 'w1', 'w15']
    dynamic_smoothing_model = ['baseline', 'rk1', 'rk2']
    seed_list = ['41', '95', '12', '35']
    model_list = ['GIN', 'GraphSage', 'GAT']
    dataset_name_list = ['CLUSTER', 'PATTERN']

    residual = 'True'

    # generate data empty dict list
    for dataset_name in dataset_name_list:
        for model in model_list:
            for seed in seed_list:
                bengio_dict_name = residual + '_SBM_' + dataset_name + "_" + model
                total_data_dict[bengio_dict_name + "_" + seed] = OrderedDict()
                total_data_dict[bengio_dict_name + "_" + seed]['wo_acc'] = wo_acc(bengio_dict_name)

    for dataset_name in dataset_name_list:
        for model in model_list:
            for seed in seed_list:
                bengio_dict_name = residual + '_SBM_' + dataset_name + "_" + model
                for w in w_list:
                    total_data_dict[bengio_dict_name + "_" + seed][w] = ''
                    for lb_ub in lb_ub_list:
                        for dnm in dynamic_smoothing_model:
                            total_data_dict[bengio_dict_name + "_" + seed][lb_ub[0]+"_"+lb_ub[1]+"_"+dnm] = ''

    for data_w in classic_data_dict_list:
        dataset_name = data_w['dataset']
        model_name = data_w['model']
        seed = data_w['seed']
        bengio_dict_name = residual + '_SBM_' + dataset_name + "_" + model_name
        total_data_dict[bengio_dict_name+"_"+seed][bengio_dict_name+"_"+seed][data_w['smoothing_name']] = data_w['test_acc']


    for data_n in new_data_dict_list:
        dataset_name = data_n['dataset']
        model_name = data_n['model']
        seed = data_n['seed']
        bengio_dict_name = residual + '_SBM_' + dataset_name + "_" + model_name
        total_data_dict[bengio_dict_name + "_" + seed][bengio_dict_name + "_" + seed][data_n['lb']+"_"+data_n['ub']+"_"+data_n['how_residual']] = data_w['test_acc']
    return total_data_dict


residual_total_data_dict = get_table_dict(new_data_dict_list, classic_data_dict_list, residual='True')
nonresidual_total_data_dict = get_table_dict(new_data_dict_list, classic_data_dict_list, residual='False')


with open(f'output_{timestampStr}.csv', 'w') as f:
    for k1, v1 in residual_total_data_dict.items():
        for k2, v2 in nonresidual_total_data_dict.items():
            k2_split_data = k2.split.data("_")
            if k1 == k2:
                k_split_data = k1.split("_")
                f.write(k_split_data[0])
                f.write(",")
                f.write(k_split_data[1])
                f.write(",")
                f.write(k_split_data[2])
                f.write(",")
                for data in v1.values():
                    f.write(data)
                    f.write(",")

                for data in k_split_data:
                    f.write(data)
                    f.write(",")

                for data in k2.values():
                    f.write(data)
                    f.write(",")
                f.write("\n")
            else:
                k_split_data = k1.split("_")
                f.write(k_split_data[0])
                f.write(",")
                f.write(k_split_data[1])
                f.write(",")
                f.write(k_split_data[2])
                f.write(",")
                for data in v1.values():
                    f.write(data)
                    f.write(",")

                for data in k_split_data:
                    f.write(data)
                    f.write(",")

                for data in v2.values():
                    f.write(data)
                    f.write(",")
                f.write("\n")





# {params["seed"]}_{str(net_params["residual"])}_{DATASET_NAME}_{MODEL_NAME}_{net_params["how_residual"]}_rk{net_params["rki"]}_lb{str(lb_delta).split(".")[1]}_ub{str(ub_delta).split(".")[1]}_md{net_params["middle_dim"]}_{net_params["bottleneck"]}_{timestampStr}_test_result.csv'
# ex) 41_True_SBM_CLUSTER_SMOOTH_GIN_baseline_rk1_lb0_ub2_md32_True
#     0  1    2    3       4      5    6       7   8   9  10    11
