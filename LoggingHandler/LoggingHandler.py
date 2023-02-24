# import python libraries
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import wandb logging
import wandb

# import python libraries
import numpy as np
import pandas as pd
import datetime as dt

# class DataHandler
class LoggingHandler(object):

    def __init__(self):

        # init experiment log
        self.experiment_log = None

        # init wandb log
        self.wandb_log = {}

        # init wandb run
        self.wandb_run = {}

    # init experiment logging
    def init_experiment_log(self, parameter, directory, file_name):

        # init experiment statistics
        summary_cols = [
            'timestamp'
            , 'seed'
            , 'train_iterations'
            , 'batch_size'
            , 'learning_rate_start'
            , 'learning_rate_iteration'
            , 'beta'
            , 'encoder_dim'
            , 'encoder_bottleneck'
            , 'decoder_dim'
            , 'decoder_bottleneck'
            , 'lat_embed_dim'
            , 'no_accounts'
            , 'feat_embed_dim'
            , 'no_features'
            , 'train_loss'
            , 'train_adj_loss'
            , 'train_fea_loss'
            , 'valid_loss'
            , 'valid_adj_loss'
            , 'valid_fea_loss'
            , 'algo'
            , 'best_min_cluster_size'
            , 'best_min_samples'
            , 'best_metric'
            , 'best_cluster_selection_method'
            , 'no_global_anomalies'
            , 'no_local_anomalies'
        ]
        self.experiment_log = pd.DataFrame(columns=summary_cols)

        # save current experiment statistics
        self.experiment_log.to_csv(os.path.join(directory, file_name), sep=',', encoding='utf-8')

    # save experiment statistics
    def save_experiment_log(self, parameter, experiment_statistics, directory, file_name):

        # collect experiment statistics
        exp_stats = {
            'timestamp': str(dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S'))
            , 'seed': parameter['seed']
            , 'train_iterations': parameter['train_iterations']
            , 'batch_size': parameter['train_batch_size']
            , 'learning_rate_start': parameter['learning_rate']
            , 'learning_rate_iteration': experiment_statistics['learning_rate']
            , 'beta': parameter['beta']
            , 'encoder_dim': parameter['encoder_dim']
            , 'encoder_bottleneck': parameter['encoder_bottleneck']
            , 'decoder_dim': parameter['decoder_dim']
            , 'decoder_bottleneck': parameter['decoder_bottleneck']
            , 'lat_embed_dim': parameter['lat_embed_dim']
            , 'no_accounts': int(experiment_statistics['no_accounts'])
            , 'feat_embed_dim': parameter['feat_embed_dim']
            , 'no_features': int(experiment_statistics['no_features'])
            , 'train_loss': np.round(experiment_statistics['average_train_loss'], 6)
            , 'train_adj_loss': np.round(experiment_statistics['average_adj_train_loss'], 6)
            , 'train_fea_loss': np.round(experiment_statistics['average_fea_train_loss'], 6)
            , 'valid_loss': np.round(experiment_statistics['average_valid_loss'], 6)
            , 'valid_adj_loss': np.round(experiment_statistics['average_adj_valid_loss'], 6)
            , 'valid_fea_loss': np.round(experiment_statistics['average_fea_valid_loss'], 6)
            , 'algo': parameter['algo']
            , 'best_min_cluster_size': parameter['min_cluster_size']
            , 'best_min_samples': parameter['min_samples']
            , 'best_metric': parameter['metric']
            , 'best_cluster_selection_method': parameter['cluster_selection_method']
            , 'no_clusters': int(experiment_statistics['no_clusters'])
            , 'no_global_anomalies': int(experiment_statistics['no_global_anomalies'])
            , 'no_local_anomalies': int(experiment_statistics['no_local_anomalies'])
        }

        # determine and collect training summary statistics of current epoch
        self.experiment_log = self.experiment_log.append(exp_stats, ignore_index=True)

        # save current experiment statistics
        self.experiment_log.to_csv(os.path.join(directory, file_name), sep=',', encoding='utf-8')

    # init wandb logging
    def init_wandb_run(self, project, parameter):

        # init weights and biases log
        self.wandb_run = wandb.init(project=project, group='{}'.format(str(parameter['base_dir'].split('/')[-1])), name='{}_ae_graph_sd_{}_ds_{}_{}'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['dataset']), str(parameter['exp_postfix'])))

        # init wandb parameters
        wandb_parameter = dict()

        # populate experiment meta parameters
        wandb_parameter['exp_timestamp'] = parameter['exp_timestamp']
        wandb_parameter['dataset'] = parameter['dataset']
        wandb_parameter['exp_postfix'] = parameter['exp_postfix']

        # model architecture parameter
        wandb_parameter['seed'] = parameter['seed']
        wandb_parameter['encoder_dim'] = parameter['encoder_dim']
        wandb_parameter['encoder_bottleneck'] = parameter['encoder_bottleneck']
        wandb_parameter['encoder_type'] = parameter['encoder_type']
        wandb_parameter['decoder_dim'] = parameter['decoder_dim']
        wandb_parameter['decoder_bottleneck'] = parameter['decoder_bottleneck']
        wandb_parameter['feat_embed_dim'] = parameter['feat_embed_dim']
        wandb_parameter['lat_embed_dim'] = parameter['lat_embed_dim']

        # model training parameter
        wandb_parameter['train_iterations'] = parameter['train_iterations']
        wandb_parameter['train_batch_size'] = parameter['train_batch_size']
        wandb_parameter['loss'] = parameter['loss']
        wandb_parameter['learning_rate'] = parameter['learning_rate']
        wandb_parameter['learning_rate_steps'] = parameter['learning_rate_steps']
        wandb_parameter['weight_decay'] = parameter['weight_decay']
        wandb_parameter['beta'] = parameter['beta']

        # model validation parameter
        wandb_parameter['valid_iterations'] = parameter['valid_iterations']
        wandb_parameter['valid_batch_size'] = parameter['valid_batch_size']

        # anomaly detection parameter
        wandb_parameter['algo'] = parameter['algo']
        wandb_parameter['grid_min_cluster_size'] = parameter['grid_min_cluster_size']
        wandb_parameter['grid_min_samples'] = parameter['grid_min_samples']
        wandb_parameter['grid_metric'] = parameter['grid_metric']
        wandb_parameter['grid_cluster_selection_method'] = parameter['grid_cluster_selection_method']

        # add wandb experiment configuration
        self.wandb_run.config.update(wandb_parameter)

    # update wandb logging
    def update_wandb_run(self, statistics):

        # fill wandb log dict
        self.wandb_log['001_model_training/001_avg_train_loss'] = statistics['average_train_loss']
        self.wandb_log['001_model_training/002_avg_adj_train_loss'] = statistics['average_adj_train_loss']
        self.wandb_log['001_model_training/003_avg_fea_train_loss'] = statistics['average_fea_train_loss']
        self.wandb_log['001_model_training/004_learning_rate'] = statistics['learning_rate']

        # log training progress
        self.wandb_run.log(self.wandb_log)

    # close wandb logging
    def close_wandb_run(self, parameter, statistics):

        # log dataset statistics
        self.wandb_run.summary['no_accounts'] = statistics['no_accounts']
        self.wandb_run.summary['no_features'] = statistics['no_features']

        # log best clustering parameters
        self.wandb_run.summary['best_min_cluster_size'] = parameter['min_cluster_size']
        self.wandb_run.summary['best_min_samples'] = parameter['min_samples']
        self.wandb_run.summary['best_metric'] = parameter['metric']
        self.wandb_run.summary['best_cluster_selection_method'] = parameter['cluster_selection_method']

        # log clustering results
        self.wandb_run.summary['no_clusters'] = statistics['no_clusters']
        self.wandb_run.summary['no_global_anomalies'] = statistics['no_global_anomalies']
        self.wandb_run.summary['no_local_anomalies'] = statistics['no_local_anomalies']

        # finish wandb run
        self.wandb_run.finish()