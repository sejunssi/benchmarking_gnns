import pickle

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

NAME = ['SBM_CLUSTER',  'SBM_PATTERN']
a_list = [1,2,3,4,8]

for name in NAME:
    for a in a_list
        with open(f'{name}_a{a}.pkl', 'rb') as f:
            data = pickle.load(f)
            train = data[0]
            data[0]['node_label'] = []
            for data in data.dataset:
                data[0]['node_label'].extend(data['node_label'])
            with open(f'./smoothed_data/{name}_a{a}.pkl', 'w') as f:
                pickle.dump(data, f)