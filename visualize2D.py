import argparse
from os.path import join
import os
import time
import numpy as np
from tqdm import tqdm
from sklearn.manifold import TSNE

from config import Config

config = Config()
def color_plot(x, label, save_path):
    # cdict = {1: 'red', 2: 'blue', 3: 'green', 4:'black', 5:'orange', 6:'yellow', 7:'purple'}

    fig, ax = plt.subplots()
    for g in np.unique(label):
        ix = np.where(label == g)
        ax.scatter(x[ix,0], x[ix,1], label=g, s=20)
    ax.legend()
    plt.show()
    plt.savefig(join(save_path, 'tSNE_plot.png'), dpi=1024) 

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

    args = argparser.parse_args()
    log_dir = 'logs/%s/%s' % (args.datasets, args.exp_id)
    res_dir = 'results/%s/%s' % (args.datasets, args.exp_id)
    create_dirs([res_dir])


	# X = np.loadtxt(join(data_root, dataset_name+'.txt'))
	y = np.loadtxt(join(config.data_root, args.datasets+'_label.txt'), dtype='int')

	x = np.load(join(res_dir, 'encoding.npy'))
	x = x.squeeze()

	x_emb = TSNE(n_components=2).fit_transform(x)

	color_plot(x_emb, y, res_dir)
