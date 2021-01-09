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

# exp_id = 'v1.0'

# middle_layer_size = [256, 128, 256]  # [n_input_features] + middle_layer_size + [n_output_features]

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
        '--lr',
        default=1e-4,
        type=float,
        help='learning rate')
    argparser.add_argument(
        '--eps',
        default=80,
        type=int,
        help='reset training epochs of training')
    argparser.add_argument(
        '--bs',
        default=4,
        type=int,
        help='batch_size for training only')
    argparser.add_argument(
        '--gpu_id',
        default='0',
        help='which gpu to use')

    args = argparser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    logs_dir = 'logs/%s/%s' % (args.datasets, args.exp_id)
    create_dirs([logs_dir])


    # train dataset
    dataset = SingleCell(
        config.data_root,
        args.datasets,
    )

    data_loader = dataloader.DataLoader(
        dataset=dataset,
        batch_size=args.bs,
        shuffle=True,
    )

    data_iter = iter(data_loader)  # data_loader is iterable
    sample_batch = next(data_iter)

    in_dim = sample_batch.size(-1)
    model_layer_sizes = [in_dim] + config.middle_layer_size + [in_dim]

    autoencoder = AutoEncoder(model_layer_sizes)
    if torch.cuda.is_available():
        autoencoder.to('cuda')  

    decLoss = nn.MSELoss()
    encLoss = SquareRegularizeLoss(p=config.p)
    encLoss.to('cuda')
    decLoss.to('cuda')
    optimizer = optim.Adam(autoencoder.parameters(), lr=args.lr)

    best_loss = 1e10
    writer = SummaryWriter(logs_dir)
    for epoch in range(args.eps):
        epoch_loss, epoch_encloss, epoch_decloss = 0, 0, 0
        with tqdm(data_loader, unit="batch") as tepoch:
            for batch in tepoch:
                if torch.cuda.is_available():
                    batch = batch.to('cuda')

                enc_out, dec_out = autoencoder(batch)

                optimizer.zero_grad()
                loss_decoder = decLoss(dec_out, batch)
                loss_encoder = encLoss(enc_out)
                loss = loss_encoder + loss_decoder

                # loss = decLoss(dec_out, batch)

                loss.backward()
                optimizer.step()

                tepoch.set_postfix(loss=loss.item())

                epoch_encloss += loss_encoder.data.cpu().numpy()*batch.size(0)
                epoch_decloss += loss_decoder.data.cpu().numpy()*batch.size(0)
                epoch_loss += loss.data.cpu().numpy()*batch.size(0)
        epoch_loss /= len(dataset)
        epoch_encloss /= len(dataset)
        epoch_decloss /= len(dataset)

        writer.add_scalar('Loss/encLoss', epoch_encloss, epoch)
        writer.add_scalar('Loss/decloss', epoch_decloss, epoch)
        writer.add_scalar('Loss/total',   epoch_loss,    epoch)

        if epoch_loss<best_loss:
            best_loss = epoch_loss
            torch.save(autoencoder.state_dict(), join(logs_dir, 'model_{:03d}_loss_{:.3f}.pth'.format(epoch, best_loss)))


        