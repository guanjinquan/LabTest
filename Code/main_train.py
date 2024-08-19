import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.config import parse_arguments
import numpy as np
import random
from trainer import Trainer


if __name__ == '__main__':
    args = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    os.chdir(os.path.dirname(__file__) + "/../")
    
    # set seed
    seed = int(args.seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
    # training
    trainer = Trainer(args=args)
    trainer.run()
      