# import python libraries
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import additional libraries
import datetime as dt

# import project libraries
from ExperimentHandler import GraphAutoencoderExperimentDynamic
from UtilsHandler import UtilsHandler
from LoggingHandler import LoggingHandler

# class GridSearchHandler
class GridSearchHandler(object):

    def __init__(self):

        # init utils handler
        self.uha = UtilsHandler.UtilsHandler()

        # init logging handler
        self.lha = LoggingHandler.LoggingHandler()

    # run grid search experiment
    def run_grid_search_experiment(self, parameter, data_parameter):

        # init grid search sub directory name
        sub_directory_name = '{}_grid_exp_ae_graph_ds_{}_{}'.format(str(parameter['exp_timestamp']), str(parameter['dataset']), str(parameter['exp_postfix']))

        # create grid search sub directory
        self.uha.create_experiment_sub_directory(parent_dir=parameter['base_dir'], folder_name=sub_directory_name)

        # reset base dir with sub directory
        parameter['base_dir'] = os.path.join(parameter['base_dir'], sub_directory_name)

        # init experiment logging
        file_name = '{}_grid_experiment_log_ds_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['dataset']), str(parameter['exp_postfix']))
        self.lha.init_experiment_log(parameter=parameter, directory=parameter['base_dir'], file_name=file_name)

        # iterate over experiment betas
        for beta in parameter['grid_beta']:

            # set current experiment beta
            parameter['beta'] = beta

            # iterate over feature embedding dimensions
            for feat_embed_dim in parameter['grid_feat_embed_dim']:

                # set current experiment beta
                parameter['feat_embed_dim'] = feat_embed_dim

                # iterate over experiment seeds
                for seed in parameter['grid_seed']:

                    # set current experiment seed
                    parameter['seed'] = seed

                    # log configuration processing
                    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
                    print('[INFO {}] DeepAppleGraph :: Start GNN grid search experiment, beta: {}, feat-embed-dim: {}, seed: {}.'.format(now, str(parameter['beta']), str(parameter['feat_embed_dim']), str(parameter['seed'])))

                    # init dynamic GNN experiment
                    exp = GraphAutoencoderExperimentDynamic.GraphAutoencoderExperimentDynamic()

                    # run dynamic GNN experiment
                    experiment_statistics = exp.run_experiement(parameter=parameter, data_parameter=data_parameter)

                    # log configuration processing
                    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
                    print('[INFO {}] DeepAppleGraph :: Complete GNN grid search experiment, beta: {}, feat-embed-dim: {}, seed: {}.'.format(now, str(parameter['beta']), str(parameter['feat_embed_dim']), str(parameter['seed'])))

                    # reset current experiment encoder and decoder dim
                    parameter['encoder_dim'] = parameter['encoder_dim'][1:]
                    parameter['decoder_dim'] = parameter['decoder_dim'][:-1]

                    # log experiment results
                    file_name = '{}_grid_experiment_log_ds_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['dataset']), str(parameter['exp_postfix']))
                    self.lha.save_experiment_log(parameter=parameter, experiment_statistics=experiment_statistics, directory=parameter['base_dir'], file_name=file_name)

        # return experiment statistics
        return experiment_statistics