import json
Model_Name = ['GCN', 'GAT', 'GraphSage', 'GIN', 'Monet', 'MLP', 'MLP_GATED', 'GatedGCN']
CONFIG_NAME = []
for model_name in Model_Name:
    CONFIG_NAME.append((f'SBMs_node_clustering_{model_name}_CLUSTER', 'SBM_CLUSTER'))
    CONFIG_NAME.append((f'SBMs_node_clustering_{model_name}_PATTERN', 'SBM_PATTERN'))

w_list = [0.9]
for name in CONFIG_NAME:
    filename, data_name = name
    for w in w_list:
        w_str = str(w).split(".")[1]
        print(f"writing [w]: {w_str}", filename)
        smoothing_fname = f'{filename}_w{w_str}'
        smoothing_data_name = f'{data_name}_w{w_str}'
        with open(f'{filename}.json', 'r') as f:
            print("reading")
            data = f.read()
            data = json.loads(data)
            data['dataset'] = smoothing_data_name
            with open(f'{smoothing_fname}.json', 'w') as f2:
                f2.write(json.dumps(data))
                print("done")
