import pickle

class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

def read_pickle(dataname):
    with open(dataname, 'rb') as f:
        data = pickle.load(f)
        return data

def load_sample_data(train_sample_data_naem, test_sample_data_name, val_sample_data_name):
    trainset = read_pickle(train_sample_data_naem)
    testset = read_pickle(test_sample_data_name)
    valset = read_pickle(val_sample_data_name)
    dataset = [trainset, testset, valset]
    with open('SBM_CLUSTER_SAMPLE.pkl', 'wb') as f:
        pickle.dumps(dataset)

train_sample_data_naem = "SBM_CLUSTER_train.pkl"
test_sample_data_name = "SBM_CLUSTER_test.pkl"
val_sample_data_name = "SBM_CLUSTER_val.pkl"

load_sample_data(train_sample_data_naem, test_sample_data_name, val_sample_data_name)


