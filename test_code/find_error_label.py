import torch

def find_error_label(label):
    if label.dtype != torch.int64:
        print(label)
        with open('error_label', 'w') as f:
            f.write(str(list(label.cpu().detach().numpy())))
        try:
            label = label.long()
        except:
            # import pdb
            # pdb.set_trace()
            raise Exception("label is not long tensor")