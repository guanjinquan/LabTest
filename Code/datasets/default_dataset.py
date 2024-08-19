from torch.utils.data import Dataset
import numpy as np
import random
import os
from utils.config import parse_arguments
import json
from datasets import default_augment


class MyBaseDataset(Dataset):
    def __init__(self, type="train", data_type="ALL", test_mode=None, args=None):
        super().__init__()
        assert test_mode is not None, "test_mode can't be None."
        
        # 所有成员变量归纳在这里
        self.args = parse_arguments() if args is None else args
        self.data_type = data_type
        self.type = type
        self.items = []
        self.transforms = \
            default_augment.TestTransforms() if test_mode else \
            default_augment.TrainTransforms()   
        
        # 计算items
        self._load_items()
        
        # 如果是train模式，random dataset
        if type == "train":
            for idx in range(1, len(self.items)):  # random shuffle
                idx2 = random.randint(0, idx)
                self.items[idx], self.items[idx2] = self.items[idx2], self.items[idx]
            
        # 打印数据集信息
        print(f"Dataset {self.type} {self.data_type} loaded. Length: {len(self.items)}")
             
    def _get_pids(self):
        pids = []
        for item in self.items:
            pids.append(item['pid'])
        return pids

    def _get_labels(self):
        ret = []
        for item in self.items:
            ret.append(item['label'])
        return ret
    
    def __getitem__(self, index):
        image = np.load(self.items[index]['path']).astype(np.uint8)
        label = self.items[index]['label']
        
        if self.data_type == "CORE":
            image = image[0:3]  # 只取前三张
        elif self.data_type == "EDGE":
            image = image[3:6]  # 只取后三张
        
        if self.transforms is not None:
            image = self.transforms(image)
            
        assert image.shape[0] == 6, f"error = {self.items[index]['pid']}"
        assert image.shape[1] == 3 and image.shape[2] == self.args.img_size and image.shape[3] == self.args.img_size, f"Invalid Shape : {image.shape} but config's img_size = {self.args.img_size}."
        
        return [image, label]  # image, labels, patient_id
    
    def __len__(self):
        return len(self.items)
    
    def _load_items(self):
        datainfo_file = self.args.datainfo_file
        datainfo_path = os.path.join(self.args.data_root, datainfo_file)
        with open(datainfo_path, 'r') as f:
            self.items = json.load(f)['datainfo']

        # filter items
        split_path = os.path.join(self.args.data_root, self.args.split_filename)
        with open(split_path, 'r') as f:
            split = json.load(f)

        target_pid = set(list(map(int, split[self.type])))
        self.items = list(filter(lambda x: x['pid'] in target_pid, self.items))
        assert self._check_datapath(), "Not all paths exist!"

        # debug模式
        if self.args.debug_mode:  # 正负类各50个
            temp_items = [[], []]
            for item in self.items:
                temp_items[item['label']].append(item)
            self.items = temp_items[0][:50] + temp_items[1][:50]
    
    
    def _check_datapath(self):
        for item in self.items:
            if not os.path.exists(item['path']):
                print("Path not exists!", item['path'], flush=True)
                print("May need to change img_size.", flush=True)
                return False
        return True    