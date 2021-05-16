from solver import Solver
import os
import argparse

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: 
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(config):
    solver = Solver(config)
    solver.run()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--latent", type=int, default=128)
    parser.add_argument("--ld", type=float, default=1.)
    parser.add_argument("--model_save_path", type=str, default="./../saved_models")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument('--block', type=str, default='se', choices=['se', 'rese', 'res', 'basic'])
    parser.add_argument('--withJE', type=str2bool, default=True)
    parser.add_argument('--data_path', type=str, default='./../../tf2-harmonic-cnn/dataset')
    parser.add_argument('--encoder_type', type=str, default='HC', choices=['HC', 'SC', 'MS'])
    config = parser.parse_args()


    print(config)
    main(config)
