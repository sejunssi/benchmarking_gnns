import json
Model_Name = ['GCN', 'GAT', 'GraphSage', 'GIN', 'Monet', 'MLP', 'MLP_GATED', 'GatedGCN']
CONFIG_NAME = []
for model_name in Model_Name:
    CONFIG_NAME.append((f'SBMs_node_clustering_{model_name}_CLUSTER', 'SBM_CLUSTER', model_name))
    CONFIG_NAME.append((f'SBMs_node_clustering_{model_name}_PATTERN', 'SBM_PATTERN', model_name))

for name in CONFIG_NAME:
    config_fname, data_name, model_name = name
    print(f"writing s", config_fname)
    smoothing_fname = f'{config_fname}_SMOOTH'
    smoothing_data_name = data_name
    smoothing_model_name = f'SMOOTH_{model_name}'
    with open(f'{config_fname}.json', 'r') as f:
        print("reading")
        data = f.read()
        data = json.loads(data)
        data['dataset'] = smoothing_data_name
        data['model'] = smoothing_model_name
        with open(f'{smoothing_fname}.json', 'w') as f2:
            f2.write(json.dumps(data))
            print("done")
