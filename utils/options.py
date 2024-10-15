import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--root', type=str, default='/Users/yitongta/Desktop/Data Center Processing/Adversarial-Attack/German')
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    parser.add_argument('--lr', type=int, default=0.001, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    parser.add_argument('--num_of_class', type=int, default=10)
    parser.add_argument('--model', type=str, default='MLP', help="model stucture")
    parser.add_argument('--save_name', type=str, default='unknow', help="model save name")
    




    args = parser.parse_args()
    return args