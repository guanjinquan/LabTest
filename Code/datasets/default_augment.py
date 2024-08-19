from torchvision.transforms import Compose, Normalize, RandomResizedCrop, RandomVerticalFlip, RandomHorizontalFlip, RandomRotation, RandomAutocontrast, RandomAdjustSharpness
import torch
            

# metadata of images
MEAN_STD = \
    ([175.14728804175988, 110.57123792228117, 176.73598615775617], \
    [21.239463551725915, 39.15991384752335, 10.99100631656543])
            
            
class TrainTransforms:
    def __init__(self):
        self.transforms = Compose([
            Normalize(mean=MEAN_STD[0], std=MEAN_STD[1]),
            RandomResizedCrop(scale=(0.85, 1), size=(512, 512)),
            RandomVerticalFlip(p=0.5), 
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=(-45, 45)),
            RandomAutocontrast(p=0.5), 
            RandomAdjustSharpness(sharpness_factor=3, p=0.5)  
        ])
    
    def __call__(self, ensemble_data):
        ensemble_data = torch.from_numpy(ensemble_data).float()
        return self.transforms(ensemble_data)


class TestTransforms:
    def __init__(self):
        self.transforms = Normalize(mean=MEAN_STD[0], std=MEAN_STD[1])

    def __call__(self, ensemble_data):
        ensemble_data = torch.from_numpy(ensemble_data).float()
        return self.transforms(ensemble_data)