# import os library
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import additional libraries
import argparse
import warnings
import datetime as dt
import numpy as np

# import pytorch libraries
import torch as th

# import project libraries
from UtilsHandler import UtilsHandler
# from ExperimentHandler import AutoencoderExperiment
from ExperimentHandler import GraphAutoencoderExperiment

# define main function
def main():

    # init client argument parser
    parser = argparse.ArgumentParser(description='deepApple Experiments')

    # general experiment parameter
    parser.add_argument('-exp_timestamp', help='', nargs='?', type=str,  default=dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
    parser.add_argument('-experiment', help='', nargs='?', type=str,  default='graph_autoencoder')
    parser.add_argument('-data_dir', help='', nargs='?', type=str, default='./100_datasets')
    parser.add_argument('-base_dir', help='', nargs='?', type=str, default='./200_experiments')
    parser.add_argument('-exp_postfix', type=str, default='poc', help='postfix of experimental runs.')

    # data parameter
    parser.add_argument('-dataset', help='', nargs='?', type=str,  default='sap') # ey, serpro, sap
    parser.add_argument('-sample_eval', help='', nargs='?', type=str, default='False')
    parser.add_argument('-sample_size', help='', nargs='?', type=int, default=1001)
    parser.add_argument('-min_line_items', help='', nargs='?', type=int, default=2)
    parser.add_argument('-max_line_items', help='', nargs='?', type=int, default=40)

    # model architecture parameter
    parser.add_argument('-seed', type=int, default=1111, help='seed value for deterministic results.')
    parser.add_argument('-data_dim', type=int, default=10, help='the dimension of the data embeddings.')
    parser.add_argument('-encoder_dim', nargs='+', default=[64, 32, 16, 8, 4, 2], help='the dimensions of the encoder gnn layers.')
    parser.add_argument('-decoder_dim', nargs='+', default=[2, 4, 8, 16, 32, 64], help='the dimensions of the decoder fc layers.')
    parser.add_argument('-hidden_dim', type=int, default=64, help='the dimension of the first gnn layer.')
    parser.add_argument('-embed_dim', type=int, default=2, help='the dimension of the graph embeddings.')

    # model training parameter
    parser.add_argument('-iterations', type=int, default=101, help='the number of training iterations.')
    parser.add_argument('-eval_iteration', type=int, default=100, help='the eval training iteration.')
    parser.add_argument('-batch_size', type=int, default=128, help='the batch size.')
    parser.add_argument('-loss', type=str, default='mse', help='the training and validation loss.')
    parser.add_argument('-learning_rate', type=float, default=0.0001, help='the learning rate.')
    parser.add_argument('-kl_div_alpha', type=float, default=0.0, help='the kl-divergence loss regularizer.')
    parser.add_argument('-device', type=str, default='cpu', help='the compute device.')
    parser.add_argument('-wandb', type=str, default='True', help='enable wandb logging.')

    # model anomaly detection paramter
    parser.add_argument('-algo', type=str, default='hdbscan', help='the anomaly detection algorithm.') # lof, svm, iforest, hdbscan
    parser.add_argument('-min_cluster_size', type=int, default=100, help='the hdbscan min cluster size.')
    parser.add_argument('-min_samples', type=int, default=1, help='the hdbscan min samples.')
    parser.add_argument('-kernel', type=str, default='rbf', help='the one-class svm kernel.')
    parser.add_argument('-degree', type=int, default=3, help='the one-class svm degree of the polynomial kernel function.')
    parser.add_argument('-gamma', type=str, default='svm', help='the one-class svm kernel coefficient.')
    parser.add_argument('-n_neighbors', type=int, default=6, help='the number of LOF neighbors.')
    parser.add_argument('-leaf_size', type=int, default=30, help='the leaf size of LOF tree creation.')

    # ignore potential warnings - mostly due to package updates
    warnings.filterwarnings('ignore')

    # parse client arguments
    parameter = vars(parser.parse_args())

    # init new handlers
    uha = UtilsHandler.UtilsHandler()

    # parse boolean args as boolean
    parameter['sample_eval'] = uha.str2bool(parameter['sample_eval'])
    parameter['wandb'] = uha.str2bool(parameter['wandb'])

    # parse integer array args as integer
    parameter['encoder_dim'] = [int(ele) for ele in parameter['encoder_dim']]
    parameter['decoder_dim'] = [int(ele) for ele in parameter['decoder_dim']]

    # set deterministic seeds of the client training experiments
    np.random.seed(int(parameter['seed']))  # set numpy seed
    th.manual_seed(int(parameter['seed']))  # set pytorch seed CPU
    th.cuda.manual_seed(int(parameter['seed']))  # set pytorch seed GPU

    # case: autoencoder experiment
    if parameter['experiment'] == 'autoencoder':

        # todo: implement experiment
        pass

    # case: graph autoencoder experiment
    elif parameter['experiment'] == 'graph_autoencoder':

        # init graph autoencoder experiment
        exp = GraphAutoencoderExperiment.GraphAutoencoderExperiment()

        # run graph autoencoder experiment
        exp.run_experiement(parameter)

# run main function
if __name__ == '__main__':

    # call main function
    main()