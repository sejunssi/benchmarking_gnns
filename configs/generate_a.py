import json
CONFIG_NAME = ['SBMs_node_clustering_GCN_CLUSTER']
a_list = [6, 10]
for filename in CONFIG_NAME:
    for a in a_list:
        fname = f'{filename}_a{a}'
        with open(f'{filename}.json', 'r') as f:
            str = f.read()
            data = json.loads(str)
            data['dataset'] = fname
            with open(f'{fname}.json', 'w') as f2:
                f2.write(json.dumps(data))
