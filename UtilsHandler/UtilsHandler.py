# import python libraries
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import libraries
import json as js
import argparse
import torch

# class UtilsHandler
class UtilsHandler(object):

    def __init__(self):

        pass

    # create experiment directories
    def create_experiment_directory(self, param, parent_dir, architecture='ae_graph'):

        # case: baseline architecture
        if architecture == 'ae_baseline':

            # create experiment directory name
            experiment_directory_name = '{}_exp_{}_ds_{}_sd_{}_itr_{}_bt_{}_lr_{}_lt_{}_enc_{}_dec_{}_bn_{}_{}'.format(
                str(param['exp_timestamp']), str(architecture), str(param['dataset']), str(param['seed']),
                str(param['iterations']), str(param['train_batch_size']), str(param['learning_rate']), str(param['embed_dim']),
                str(len(param['encoder_dim'])), str(len(param['decoder_dim'])), str(param['bottleneck']), str(param['exp_postfix'])
            )

        # case: gnn architecture
        if architecture == 'ae_graph':

            # create experiment directory name
            experiment_directory_name = '{}_exp_{}_ds_{}_sd_{}_itr_{}_bt_{}_lr_{}_lt_{}_enc_{}_dec_{}_bn_{}_{}'.format(
                str(param['exp_timestamp']), str(architecture), str(param['dataset']), str(param['seed']),
                str(param['iterations']), str(param['train_batch_size']), str(param['learning_rate']), str(param['embed_dim']),
                str(len(param['encoder_dim'])), str(len(param['decoder_dim'])), str(param['bottleneck']), str(param['exp_postfix'])
            )

        # case: potential additional architecture
        elif architecture == 'not_implemented':

            # todo
            pass

        # create experiment directory name
        exp_dir = os.path.join(parent_dir, experiment_directory_name)

        # case experiment directory does not exist
        if (not os.path.exists(exp_dir)):

            # create new experiment directory
            os.makedirs(exp_dir)

        # create meta data, signal data, and backtest data sub directories
        par_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='00_param')
        sta_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='01_statistics')
        res_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='02_results')
        vis_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='03_visuals')
        log_sub_dir = self.create_experiment_sub_directory(parent_dir=exp_dir, folder_name='04_logging')

        # return new experiment directory and sub directories
        return exp_dir, par_sub_dir, sta_sub_dir, res_sub_dir, vis_sub_dir, log_sub_dir

    # create experiment sub-directories
    def create_experiment_sub_directory(self, parent_dir, folder_name):

        # create sub directory name
        sub_dir = os.path.join(parent_dir, folder_name)

        # case experiment sub directory does not exist
        if (not os.path.exists(sub_dir)):

            # create new sub directory
            os.makedirs(sub_dir)

        # return new sub directory
        return sub_dir

    # save client model checkpoint
    def save_client_model_checkpoint(self, filename, iteration, model, optimizer, chpt_dir):

        # create checkpoint
        checkpoint = {'iteration': iteration,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()
                      }

        # create model checkpoint file path
        filepath = os.path.join(chpt_dir, str(filename))

        # save model checkpoint
        torch.save(checkpoint, filepath)

    # convert str to bool
    def str2bool(self, value):

        # case: already boolean
        if type(value) == bool:

            # return actual boolean
            return value

        # convert value to lower cased string
        # case: true acronyms
        elif value.lower() in ('yes', 'true', 't', 'y', '1'):

            # return true boolean
            return True

        # case: false acronyms
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):

            # return false boolean
            return False

        # case: no valid acronym detected
        else:

            # raise error
            raise argparse.ArgumentTypeError('[ERROR] Boolean value expected.')

    # save experiment parameter
    def save_experiment_parameter(self, param, parameter_dir):

        # create filename
        filename = str('{}_DeepAppleGraph_exp_parameter.txt'.format(str(param['exp_timestamp'])))

        # write experimental config to file
        with open(os.path.join(parameter_dir, filename), 'w') as outfile:

            # dump experiment parameters
            js.dump(param, outfile)