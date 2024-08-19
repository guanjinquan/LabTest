import torch

def collate_fn_ensemble(input):  
    data = []
    target = []

    for x, y in input:
        data.append(x)
        target.append(y)
    
    data = torch.cat(data, dim=0)
    target = torch.Tensor(target).long()
    return (data, target)
