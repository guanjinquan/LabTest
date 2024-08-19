from models.backbones import *
from torch import nn 

class MyModel(nn.Module):
    ensemble_num = 6
    
    def __init__(self, backbone) -> None:
        super(MyModel, self).__init__()
        
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256), 
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64), 
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(64, 2),
        )
    
    # 这两个函数是为了获取backbone和head的参数，放入优化器的不同组中分别训练
    def get_backbone_params(self):
        return list(self.backbone.get_backbone_params())
    
    def get_others_params(self):
        backbones = set(self.get_backbone_params())
        return [p for p in self.parameters() if p not in backbones]
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def GetModel(args):
    if args.model == 'resnet50_imagenet':
        backbone = resnet_imagenet(args, 50)
    elif args.model == 'vit_small_p16_pathology':
        backbone = vit_small_p16_pathology(args)
    elif args.model == 'swin_imagenet':
        backbone = swin_imagenet(args)
    elif args.model == "vit_small_p16_imagenet":
        backbone = vit_small_p16_imagenet(args)
    elif args.model == "resnet50_pathology":
        backbone = resnet50_pathology(args)
    else:
        raise ValueError("model not supported")
    
    model = MyModel(backbone)    
    return model 