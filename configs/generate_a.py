import json
Model_Name = ['GCN', 'GAT', 'GraphSage', 'GIN', 'Monet', 'MLP', 'MLP_GATED', 'GatedGCN']
CONFIG_NAME = []
for model_name in Model_Name:
    CONFIG_NAME.append((f'SBMs_node_clustering_{model_name}_CLUSTER', 'SBM_CLUSTER'))
    CONFIG_NAME.append((f'SBMs_node_clustering_{model_name}_PATTERN', 'SBM_PATTERN'))

a_list = [6, 7, 10, 12, 13, 9]
for name in CONFIG_NAME:
    filename, data_name = name
    for a in a_list:
        print(f"writing [a]: {a}", filename)
        smoothing_fname = f'{filename}_a{a}'
        smoothing_data_name = f'{data_name}_a{a}'
        with open(f'{filename}.json', 'r') as f:
            print("reading")
            str = f.read()
            data = json.loads(str)
            data['dataset'] = smoothing_data_name
            with open(f'{smoothing_fname}.json', 'w') as f2:
                f2.write(json.dumps(data))
                print("done")
