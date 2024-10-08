from torch.optim import Adam, AdamW
import torch

def GetOptimizer(args, model):
    if args.optimizer == 'Adam':
        return Adam([
            {'params': model.get_backbone_params(), 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
            {'params': model.get_others_params(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ])
    elif args.optimizer == 'AdamW':
        return AdamW([
            {'params': model.get_backbone_params(), 'lr': args.backbone_lr, 'weight_decay': args.weight_decay},
            {'params': model.get_others_params(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
        ])
    else:
        raise ValueError("optimizer not supported")
    
def GetScheduler(args, optim):
    if args.scheduler == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs, eta_min=1e-15)
    elif args.scheduler == 'CosineAnnealingLR_warmup':
        assert args.num_epochs % 2 == 0, "num_epochs must be even"
        return torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.num_epochs // 2, eta_min=1e-15)
    elif args.scheduler == 'OneCycleLR':
        return torch.optim.lr_scheduler.OneCycleLR(
                optim, 
                max_lr=[args.backbone_lr, args.learning_rate], 
                epochs=args.num_epochs, 
                steps_per_epoch=1, 
                anneal_strategy='cos'
            )
    else:
        raise ValueError("scheduler not supported") 
