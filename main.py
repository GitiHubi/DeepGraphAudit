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
    parser.add_argument('-exp_mode', type=str, default='dynamic', help='postfix of experimental runs.')  # complete dynamic

    # data parameter
    parser.add_argument('-dataset', help='', nargs='?', type=str,  default='ey') # ey, serpro, sap
    parser.add_argument('-sample_eval', help='', nargs='?', type=str, default='False')
    parser.add_argument('-sample_size', help='', nargs='?', type=int, default=2002)
    parser.add_argument('-min_line_items', help='', nargs='?', type=int, default=2)
    parser.add_argument('-max_line_items', help='', nargs='?', type=int, default=20)

    # model architecture parameter
    parser.add_argument('-seed', type=int, default=1111, help='seed value for deterministic results.')
    parser.add_argument('-encoder_dim', nargs='+', default=[16, 8, 4, 2], help='the dimensions of the encoder gnn layers.') # [128, 64, 32, 16, 8, 4, 2]
    parser.add_argument('-decoder_dim', nargs='+', default=[2, 4, 8, 16], help='the dimensions of the decoder fc layers.') # [2, 4, 8, 16, 32, 64, 128]
    parser.add_argument('-encoder_type', type=str, default='embed', help='the type of feature encoding') # onehot, embed
    parser.add_argument('-bottleneck', type=str, default='linear', help='the bottleneck non-linearity.') # linear, tanh, lrelu
    parser.add_argument('-feat_embed_dim', type=int, default=2, help='the feature dimension of the graph embeddings.')
    parser.add_argument('-lat_embed_dim', type=int, default=2, help='the dimension of the graph embeddings.')

    # model training parameter
    parser.add_argument('-train_iterations', type=int, default=100001, help='the number of training iterations.')
    parser.add_argument('-train_batch_size', type=int, default=1, help='the training batch size.')
    parser.add_argument('-loss', type=str, default='mse', help='the training and validation loss.') #mse, bce
    parser.add_argument('-learning_rate', type=float, default=0.01, help='the learning rate.')
    parser.add_argument('-learning_rate_steps', type=int, default=2, help='the learning rate steps.')
    parser.add_argument('-weight_decay', nargs='?', type=float, default=1e-6, help='the optimizer weight decay.')
    parser.add_argument('-kl_div_alpha', type=float, default=0.0, help='the kl-divergence loss regularizer.')
    parser.add_argument('-beta', type=float, default=0.5, help='the loss regularizer.')
    parser.add_argument('-device', type=str, default='cpu', help='the compute device.')

    # model evaluation parameter
    parser.add_argument('-valid_iterations', type=int, default=5000, help='the eval training iteration.')
    parser.add_argument('-valid_batch_size', type=int, default=1, help='the evaluation batch size.')
    parser.add_argument('-wandb', type=str, default='True', help='enable wandb logging.')

    # anomaly detection parameter
    parser.add_argument('-algo', type=str, default='hdbscan', help='the anomaly detection algorithm.') # lof, svm, iforest, hdbscan
    parser.add_argument('-kernel', type=str, default='rbf', help='the one-class svm kernel.')
    parser.add_argument('-degree', type=int, default=3, help='the one-class svm degree of the polynomial kernel function.')
    parser.add_argument('-gamma', type=str, default='svm', help='the one-class svm kernel coefficient.')
    parser.add_argument('-n_neighbors', type=int, default=6, help='the number of LOF neighbors.')
    parser.add_argument('-leaf_size', type=int, default=30, help='the leaf size of LOF tree creation.')

    # hdbscan grid search parameter
    parser.add_argument('-grid_min_cluster_size', nargs='+', default=[10, 50, 100, 200, 300, 400, 500, 600], help='')
    parser.add_argument('-grid_min_samples', nargs='+', default=[5, 10, 30, 50, 60, 100], help='')
    parser.add_argument('-grid_metric', nargs='+', default=['euclidean', 'manhattan'], help='')
    parser.add_argument('-grid_cluster_selection_method', nargs='+', default=['eom', 'leaf'], help='')

    # ignore potential warnings - mostly due to package updates
    warnings.filterwarnings('ignore')

    # parse client arguments
    experiment_parameter = vars(parser.parse_args())

    # init new handlers
    uha = UtilsHandler.UtilsHandler()

    # parse boolean args as boolean
    experiment_parameter['sample_eval'] = uha.str2bool(experiment_parameter['sample_eval'])
    experiment_parameter['wandb'] = uha.str2bool(experiment_parameter['wandb'])

    # parse integer array args as integer
    experiment_parameter['encoder_dim'] = [int(ele) for ele in experiment_parameter['encoder_dim']]
    experiment_parameter['decoder_dim'] = [int(ele) for ele in experiment_parameter['decoder_dim']]
    experiment_parameter['grid_min_cluster_size'] = [int(ele) for ele in experiment_parameter['grid_min_cluster_size']]
    experiment_parameter['grid_min_samples'] = [int(ele) for ele in experiment_parameter['grid_min_samples']]

    # parse string array as string
    experiment_parameter['grid_metric'] = [str(ele) for ele in experiment_parameter['grid_metric']]
    experiment_parameter['grid_cluster_selection_method'] = [str(ele) for ele in experiment_parameter['grid_cluster_selection_method']]

    # determine compute device
    # determine hardware device
    experiment_parameter['device'] = th.device("cuda:0" if th.cuda.is_available() else "cpu").type

    # set deterministic seeds of the client training experiments
    np.random.seed(int(experiment_parameter['seed']))  # set numpy seed
    th.manual_seed(int(experiment_parameter['seed']))  # set pytorch seed CPU
    th.cuda.manual_seed(int(experiment_parameter['seed']))  # set pytorch seed GPU

    # case: autoencoder experiment
    if experiment_parameter['experiment'] == 'autoencoder':

        # todo: implement experiment
        pass

    # case: graph autoencoder experiment
    elif experiment_parameter['experiment'] == 'graph_autoencoder':

        # case: ey dataset
        if experiment_parameter['dataset'] == 'ey':

            # init dataset statistics
            data_parameter = dict()

            # collect dataset statistics
            data_parameter['dataset'] = experiment_parameter['dataset']

            # set dataset parameter
            data_parameter['je_identifier_field'] = 'Y_JE_IDENTIFIER'
            data_parameter['je_line_item_field'] = 'Y_JE_ITEM_IDENTIFIER'
            data_parameter['je_gl_account_field'] = 'Y_GL_ACCOUNT_NUMBER'
            data_parameter['je_debit_credit_field'] = 'Y_DEBIT_CREDIT'

            # set journal entry attribute information
            data_parameter['je_header_attributes'] = ['JEIdentifier', 'Source', 'PreparerID']
            data_parameter['je_segment_attributes_categorical'] = ['AccountType', 'AccountClass', 'GLAccountNumber', 'GLAccountName']
            data_parameter['je_segment_attributes_numerical'] = ['Amount']
            data_parameter['je_attributes'] = data_parameter['je_header_attributes'] + data_parameter['je_segment_attributes_categorical'] + data_parameter['je_segment_attributes_numerical']

            # set journal entry feature information
            data_parameter['je_header_features'] = ['Y_SOURCE', 'Y_PREPARER_ID']
            data_parameter['je_segment_features_categorical'] = ['Y_JE_ITEM_IDENTIFIER', 'Y_DEBIT_CREDIT', 'Y_GL_ACCOUNT_TYPE', 'Y_GL_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME']  # Y_JE_ITEM_IDENTIFIER and Y_DEBIT_CREDIT are derived features.
            data_parameter['je_segment_features_numerical'] = ['Y_DMBTR']
            data_parameter['je_features'] = data_parameter['je_header_features'] + data_parameter['je_segment_features_categorical'] + data_parameter['je_segment_features_numerical']
            data_parameter['je_segment_features'] = data_parameter['je_segment_features_categorical'] + data_parameter['je_segment_features_numerical']

            # set the visualization features
            data_parameter['hover_attributes'] = ['Y_JE_IDENTIFIER', 'Y_JE_ITEM_IDENTIFIER', 'Y_GL_ACCOUNT_TYPE', 'Y_GL_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NAME', 'Y_PREPARER_ID', 'Y_SOURCE', 'Y_DMBTR']
            data_parameter['visual_attributes'] = ['Y_BUZEI', 'Y_PREPARER_ID', 'Y_SOURCE', 'Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NAME', 'Y_DMBTR']

        # case: sap dataset
        elif experiment_parameter['dataset'] == 'sap':

            # init dataset statistics
            data_parameter = dict()

            # collect dataset statistics
            data_parameter['dataset'] = experiment_parameter['dataset']

            # set dataset parameter
            data_parameter['je_identifier_field'] = 'Y_BELNR'
            data_parameter['je_line_item_field'] = 'Y_JE_ITEM_IDENTIFIER'
            data_parameter['je_gl_account_field'] = 'Y_HKONT'
            data_parameter['je_debit_credit_field'] = 'Y_DEBIT_CREDIT'

            # set journal entry attribute information
            data_parameter['je_header_attributes'] = ['DocumentNr', 'DocType', 'DocTypeDescr', 'UserName Post', 'TransactionDescription']
            data_parameter['je_segment_attributes_categorical'] = ['PostingKey', 'GL Accountnr', 'GL AccountDescr', 'ProfitCenter', 'CostCenter']
            data_parameter['je_segment_attributes_numerical'] = ['AmountinUSD']
            data_parameter['je_attributes'] = data_parameter['je_header_attributes'] + data_parameter['je_segment_attributes_categorical'] + data_parameter['je_segment_attributes_numerical']

            # set journal entry feature information
            data_parameter['je_header_features'] = ['Y_BLART', 'Y_USNAM', 'Y_TCODE']
            data_parameter['je_segment_features_categorical'] = ['Y_JE_ITEM_IDENTIFIER', 'Y_DEBIT_CREDIT', 'Y_BSCHL', 'Y_HKONT', 'Y_HKONT_TEXT', 'Y_PRCTR', 'Y_KOSTL']
            data_parameter['je_segment_features_numerical'] = ['Y_DMBTR']
            data_parameter['je_features'] = data_parameter['je_header_features'] + data_parameter['je_segment_features_categorical'] + data_parameter['je_segment_features_numerical']
            data_parameter['je_segment_features'] = data_parameter['je_segment_features_categorical'] + data_parameter['je_segment_features_numerical']

            # set the visualization features
            data_parameter['hover_attributes'] = ['Y_BELNR', 'Y_JE_ITEM_IDENTIFIER', 'Y_BLART', 'Y_USNAM', 'Y_TCODE', 'Y_BSCHL', 'Y_HKONT_TEXT', 'Y_DMBTR', 'Y_PRCTR', 'Y_KOSTL']
            data_parameter['visual_attributes'] = ['Y_JE_ITEM_IDENTIFIER', 'Y_USNAM', 'Y_BLART', 'Y_TCODE', 'Y_BSCHL', 'Y_HKONT_TEXT', 'Y_DMBTR', 'Y_PRCTR', 'Y_KOSTL']

        # init graph autoencoder experiment
        exp = GraphAutoencoderExperiment.GraphAutoencoderExperiment()

        # run graph autoencoder experiment
        exp.run_experiement(parameter=experiment_parameter, data_parameter=data_parameter)

# run main function
if __name__ == '__main__':

    # call main function
    main()