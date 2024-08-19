import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Model')

    # path settings
    parser.add_argument('--data_root', type=str, default="./Data")
    parser.add_argument('--ckpt_path', type=str, default='./Checkpoints/', help='the path to save checkpoints')
    parser.add_argument('--log_path', type=str, default='./Results', help='the path to save log')
    
    # dataset settings 
    parser.add_argument('--datainfo_file', type=str, default="pathology_info.json") 
    parser.add_argument('--split_filename', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument("--debug_mode", type=bool, default=False)
    
    # models settings 
    parser.add_argument('--model', type=str)

    # trainer settings
    parser.add_argument("--runs_id", type=str)
    parser.add_argument("--acc_step", type=int)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--seed', type=int, default=109, help='random seed')
    parser.add_argument('--learning_rate', type=float, help='learning rate')
    parser.add_argument('--backbone_lr', type=float, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=6e-5, help='weight decay')
    parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch Size')
    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choose optimizer')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        help='choose scheduler')
    parser.add_argument('--loss', type=str, default='FocalLoss',
                        help='choose loss function')
    
    
    args = parser.parse_args()
    return args
