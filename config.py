import argparse


def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="/media/userdisk1/yf/ISIC2019/ISIC_2019_Training_Input")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--dataset", type=str, default="ISIC2019")
    parser.add_argument("--attack_mode", type=str, default="all2one")

    parser.add_argument("--bs", type=int, default=64)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, default=[50,100,150,200])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=200)
    parser.add_argument("--num_workers", type=float, default=12)

    parser.add_argument("--target_label", type=int, default=3)
    parser.add_argument("--pc", type=float, default=0.1)
    parser.add_argument("--cross_ratio", type=float, default=1)  # num_cross = int(num_bd * opt.cross_ratio)

    parser.add_argument("--random_rotation", type=int, default=90)
    
    # Fourier attack
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--target_img", type=str, default="./coco_val75/000000002157.jpg")
    parser.add_argument("--cross_dir", type=str, default="./coco_test1000")

    parser.add_argument('--split_idx', type=int, default=0) # multifold cross validation
    parser.add_argument('--experiment_idx',type=str,default='0')# name the experiment
    parser.add_argument('--test_model', type=str, default='0') # path of test model (used by eval.py)
    return parser
