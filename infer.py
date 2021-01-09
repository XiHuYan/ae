import argparse
from os.path import join
import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader
from tensorboardX import SummaryWriter

from config import Config
from utils import *
from datasets import *
from model import *
from loss import *

config = Config()
# data_root = '/home/yanxuhua/data/singlecell'
# middle_layer_size = [256, 128, 256]  # [n_input_features] + middle_layer_size + [n_output_features]

def get_test_data(dataset_name):
    x = np.loadtxt(join(config.data_root, dataset_name+'.txt'))
    # y = np.loadtxt(join(data_root, dataset_name+'_label.txt'))
    x = preprocess(x)
    # return x, y
    return x

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--exp_id',
        required=True,
        help='experiment id')
    argparser.add_argument(
        '--datasets',
        required=True,
        help='dataset_name')
    argparser.add_argument(
        '--gpu_id',
        default='0',
        help='which gpu to use')

    args = argparser.parse_args()
    log_dir = 'logs/%s/%s' % (args.datasets, args.exp_id)
    res_dir = 'results/%s/%s' % (args.datasets, args.exp_id)
    create_dirs([res_dir])
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    X = get_test_data(args.datasets)

    in_dim = X.shape[-1]
    model_layer_sizes = [in_dim] + config.middle_layer_size + [in_dim]

    autoencoder = AutoEncoder(model_layer_sizes)
    ckpts = find_last(log_dir)
    autoencoder.load_state_dict(torch.load(ckpts))
    autoencoder.eval()

    if torch.cuda.is_available():
        autoencoder.to('cuda')  

    y_enc = []
    with torch.no_grad():
        for cell in tqdm(X):
            cell = torch.Tensor(cell).view(1, -1).to('cuda')
            enc, dec = autoencoder(cell)
            enc = enc.cpu().numpy()
            y_enc.append(enc)

    y_enc = np.array(y_enc)
    np.save(join(res_dir, 'encoding.npy'), y_enc)




