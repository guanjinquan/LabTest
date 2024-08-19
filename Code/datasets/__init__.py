from utils.config import parse_arguments
from datasets.data_utils import *
from torch.utils.data import DataLoader
from datasets.default_dataset import MyBaseDataset
from datasets.data_sampler import BalancedBatchSampler


def GetDataLoader(args=None, test_mode=False):
    
    train_set = GetDataset("train", "ALL", test_mode, args)
    valid_set = GetDataset("valid", "ALL", True, args)
    test_set = GetDataset("test", "ALL", True, args)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, 
        sampler=BalancedBatchSampler(train_set), num_workers=8, pin_memory=True, collate_fn=collate_fn_ensemble)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, collate_fn=collate_fn_ensemble)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
        num_workers=8, pin_memory=True, collate_fn=collate_fn_ensemble)

    return train_loader, valid_loader, test_loader


def GetDataset(type="train", data_type="ALL", test_mode=False, args=None):
    print("Using VanillaDataset!")
    return MyBaseDataset(type, data_type, test_mode, args)

