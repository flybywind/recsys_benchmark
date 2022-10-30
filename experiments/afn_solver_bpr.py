import argparse
import torch
import os
from random import randint
from typing import Tuple, Optional
import sys
sys.path.append('../src')
from models import AFN
from utils import get_folder_path
from solvers import BaseSolver


MODEL_TYPE = 'CTR'
LOSS_TYPE = 'BPR'
MODEL = 'AFN'
GRAPH_TYPE = 'hete' # for dataset parser

parser = argparse.ArgumentParser()


def convert_tuple_int(a:str) -> Tuple[int]:
    try:
        a = [int(x) for x in a.split(",")]
        return a
    except ValueError:
        msg = "Given param ({}) not valid! Expected format, 1,2 ".format(a)
        raise argparse.ArgumentTypeError(msg)


def convert_str_bool(a:str) -> bool:
    try:
        a = a.lower()
        return a == 'true' or a == 't' or a == '1'
    except ValueError:
        msg = "Given param ({}) not valid! Expected format, True/T or False/F ".format(a)
        raise argparse.ArgumentTypeError(msg)


def convert_opt_float(a:str) -> Optional[float]:
    try:
        a = a.lower()
        if a == 'none':
            return None
        return float(a)
    except ValueError:
        msg = "Given param ({}) not valid! Expected format, float or none ".format(a)
        raise argparse.ArgumentTypeError(msg)


# Dataset params
parser.add_argument('--dataset', type=str, default='Movielens', help='')		#Movielens, Yelp
parser.add_argument('--dataset_name', type=str, default='latest-small', help='')	#25m, latest-small
parser.add_argument('--if_use_features', type=str, default='false', help='')
parser.add_argument('--num_core', type=int, default=10, help='')
parser.add_argument('--num_feat_core', type=int, default=10, help='')
parser.add_argument('--sampling_strategy', type=str, default='random', help='') # unseen(for latest-small), random(for Yelp,25m)
parser.add_argument('--entity_aware', type=str, default='false', help='')

# Model params
parser.add_argument('--ltl_hidden_size', type=int, default=256, help='')
parser.add_argument('--afn_dnn_hidden_units', type=convert_tuple_int, default="256,128", help='')
parser.add_argument('--emb_dim', type=int, default=64, help='')
parser.add_argument('--max_norm', type=convert_opt_float, default='None')
parser.add_argument('--l2_reg_embedding', type=float, default=0.00001, help='')
parser.add_argument('--l2_reg_dnn', type=float, default=0.0, help='')
parser.add_argument('--l2_reg_shadow', type=convert_opt_float, default=None, help='')
parser.add_argument('--init_std', type=float, default=0.0001, help='embedding matrix init std')
parser.add_argument('--seed', type=int, default=randint(-10000, 10000), help='')
parser.add_argument('--dnn_dropout', type=float, default=0., help='')
parser.add_argument('--activation', type=str, default='relu', help='')
parser.add_argument('--dtype', type=str, default='float32', help='')

# Train params
parser.add_argument('--init_eval', type=convert_str_bool, default='true', help='')
parser.add_argument('--num_negative_samples', type=int, default=4, help='')
parser.add_argument('--num_neg_candidates', type=int, default=99, help='')

parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--gpu_idx', type=str, default='0', help='')
parser.add_argument('--runs', type=int, default=5, help='')             #5(for MovieLens), 3(for Yelp)
parser.add_argument('--epochs', type=int, default=30, help='')          #30(for MovieLens), 20(only for Yelp)
parser.add_argument('--batch_size', type=int, default=1024, help='')    #1024(for others), 4096(only for 25m)
parser.add_argument('--num_workers', type=int, default=12, help='')
parser.add_argument('--opt', type=str, default='adam', help='')
parser.add_argument('--lr', type=float, default=0.001, help='')
parser.add_argument('--weight_decay', type=float, default=0.001, help='')
parser.add_argument('--early_stopping', type=int, default=20, help='')
parser.add_argument('--save_epochs', type=str, default='5,10,15,20,25', help='')
parser.add_argument('--save_every_epoch', type=int, default=26, help='')        #26(for MovieLens), 16(only for Yelp)

args = parser.parse_args()


# Setup data and weights file path
data_folder, weights_folder, logger_folder = \
    get_folder_path(model=MODEL, dataset=args.dataset + args.dataset_name, loss_type=LOSS_TYPE)

# Setup device
device = args.device
if args.device == 'cuda':
    if not torch.cuda.is_available():
        print("WARN: cuda not available, use cpu instead")
        device = 'cpu'
    else:
        device = 'cuda:{}'.format(args.gpu_idx)

# Setup args
dataset_args = {
    'root': data_folder, 'dataset': args.dataset, 'name': args.dataset_name,
    'if_use_features': args.if_use_features.lower() == 'true', 'num_negative_samples': args.num_negative_samples,
    'num_core': args.num_core, 'num_feat_core': args.num_feat_core,
    'cf_loss_type': LOSS_TYPE, 'type': GRAPH_TYPE,
    'sampling_strategy': args.sampling_strategy, 'entity_aware': args.entity_aware.lower() == 'true',
    'model': MODEL
}
model_args = {
    'model_type': MODEL_TYPE,
    'dnn_dropout': args.dnn_dropout,
    'ltl_hidden_size': args.ltl_hidden_size,
    'afn_dnn_hidden_units': args.afn_dnn_hidden_units,
    'emb_dim': args.emb_dim,
    'max_norm': args.max_norm,
    'l2_reg_embedding': args.l2_reg_embedding,
    'l2_reg_dnn': args.l2_reg_dnn,
    'l2_reg_shadow': args.l2_reg_shadow,
    'init_std': args.init_std,
    'activation': args.activation,
    'dtype': args.dtype
}
path_args = model_args.copy()
train_args = {
    'init_eval': args.init_eval,
    'num_negative_samples': args.num_negative_samples, 'num_neg_candidates': args.num_neg_candidates,
    'opt': args.opt,
    'runs': args.runs,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'weight_decay': args.weight_decay,  'device': device,
    'lr': args.lr,
    'num_workers': args.num_workers,
    'weights_folder': os.path.join(weights_folder, str(path_args)[:255]),
    'logger_folder': os.path.join(logger_folder, str(path_args)[:255]),
    'save_epochs': [int(i) for i in args.save_epochs.split(',')], 'save_every_epoch': args.save_every_epoch,
    'metapath_test': args.metapath_test.lower() == 'true'
}
print('dataset params: {}'.format(dataset_args))
print('task params: {}'.format(model_args))
print('train params: {}'.format(train_args))

if __name__ == '__main__':
    solver = BaseSolver(AFN, dataset_args, model_args, train_args)
    solver.run()
