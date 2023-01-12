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
import pandas as pd
from tqdm import tqdm

# import pytorch libraries
import torch as th
from torch.utils.data import DataLoader

# import project libraries
from UtilsHandler import UtilsHandler
from DataHandler import DataHandler
from DataHandler.AccountingDataset import AccountingDataset
from DataHandler.AccountingGNNDataset import AccountingGNNDataset
from ModelHandler import Autoencoder
from ModelHandler import GNNAutoencoder
from VisualisationHandler import VisualisationHandler

# define client main
def main():

    # init client argument parser
    parser = argparse.ArgumentParser(description='deepApple Experiments')

    # general experiment parameter
    parser.add_argument('-exp_timestamp', help='', nargs='?', type=str,  default=dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
    parser.add_argument('-data_dir', help='', nargs='?', type=str, default='./100_datasets')
    parser.add_argument('-base_dir', help='', nargs='?', type=str, default='./200_experiments')
    parser.add_argument('-exp_postfix', type=str, default='poc', help='postfix of experimental runs.')

    # data parameter
    parser.add_argument('-dataset', help='', nargs='?', type=str,  default='ernstyoung')
    parser.add_argument('-sample_eval', help='', nargs='?', type=str, default='False')
    parser.add_argument('-sample_size', help='', nargs='?', type=int, default=10000)

    # model architecture parameter
    parser.add_argument('-seed', type=int, default=1111, help='seed value for deterministic results.')
    parser.add_argument('-encoder_dim', help='', nargs='+', default=[64, 32, 16, 8, 4, 2]) # [8, 4, 2], [5096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
    parser.add_argument('-bottleneck', help='', nargs='?', type=str, default='linear')
    parser.add_argument('-decoder_dim', help='', nargs='+', default=[2, 4, 8, 16, 32, 64]) # [2, 4, 8], [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 5096]

    # model training parameter
    parser.add_argument('-iterations', type=int, default=100001, help='the number of training iterations.')
    parser.add_argument('-eval_iteration', type=int, default=10000, help='the eval training iteration.')
    parser.add_argument('-batch_size', type=int, default=64, help='the batch size.')
    parser.add_argument('-learning_rate', type=float, default=0.0001, help='the learning rate.')
    parser.add_argument('-device', type=str, default='cpu', help='the compute device.')

    # ignore potential warnings - mostly due to package updates
    warnings.filterwarnings('ignore')

    # parse client arguments
    parameter = vars(parser.parse_args())

    # init new handlers
    uha = UtilsHandler.UtilsHandler()
    dha = DataHandler.DataHandler()
    vha = VisualisationHandler.VisualizationHandler()

    # parse boolean args as boolean
    parameter['sample_eval'] = uha.str2bool(parameter['sample_eval'])

    # set deterministic seeds of the client training experiments
    np.random.seed(int(parameter['seed']))  # set numpy seed
    th.manual_seed(int(parameter['seed']))  # set pytorch seed CPU
    th.cuda.manual_seed(int(parameter['seed']))  # set pytorch seed GPU

    # parse string args as int
    parameter['encoder_dim'] = [int(ele) for ele in parameter['encoder_dim']]
    parameter['decoder_dim'] = [int(ele) for ele in parameter['decoder_dim']]

    # create experiment directory
    parameter['exp_dir'], parameter['par_sub_dir'], parameter['sta_sub_dir'], parameter['res_sub_dir'], parameter['vis_sub_dir'], parameter['log_sub_dir'] = uha.create_experiment_directory(param=parameter, parent_dir=parameter['base_dir'], architecture='ae_baseline')

    # load the EY training data
    processed_entries, aggregated_entries, prepared_entries, entries_statistics = dha.get_ernstyoung_data_range(parameter=parameter)

    # log aggregated categorical entries
    file_name = '{}_sampled_processed_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
    processed_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

    # log aggregated categorical entries
    file_name = '{}_sampled_aggregated_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
    aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

    # log aggregated categorical entries
    file_name = '{}_sampled_prepared_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
    prepared_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

    # convert the EY training data to pytorch tensor
    prepared_tensor_entries = AccountingDataset(dataset=prepared_entries.values.astype(np.float32))

    #### start training routine

    # init the EY data loader
    train_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=True)

    # update encoder and decoder network dimensions
    parameter['encoder_dim'].insert(0, prepared_entries.shape[1])
    parameter['decoder_dim'].insert(len(parameter['encoder_dim']), prepared_entries.shape[1])

    # init the autoencoder model
    model = Autoencoder.Autoencoder(
        encoder_layer=parameter['encoder_dim'],
        bottleneck=parameter['bottleneck'],
        decoder_layer=parameter['decoder_dim'],
    ).to(parameter['device'])

    # init aggregated categorical autoencoder loss
    rec_criterion = th.nn.MSELoss().to(parameter['device'])

    # init training optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=parameter['learning_rate'])

    # set model in train mode
    model.train()

    # init the training loss
    average_train_loss = 0.0

    # push aggregated losses to compute device
    rec_criterion_cat = rec_criterion.to(parameter['device'])

    # init and wrap range of training iterations
    training_iterations = tqdm(range(0, parameter['iterations']))

    # case: mini-batches are still available
    for i in training_iterations:

        # get next training batch
        inputs, targets = next(iter(train_loader))

        # push the inputs and targets to compute device
        inputs, targets = inputs.to(parameter['device']), targets.to(parameter['device'])

        # reset optimizer gradients
        optimizer.zero_grad()

        # run model forward pass
        latent, reconstructions = model(inputs)

        # compute and add categorical reconstruction loss
        train_batch_loss = rec_criterion_cat(input=reconstructions, target=inputs)

        # compute and collect average reconstruction loss
        average_train_loss += train_batch_loss.cpu().detach().item()

        # log training progress
        now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
        training_iterations.set_description(
            (
                '[INFO {}] DeepAppleGraph :: iteration: {}, train-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(train_batch_loss.cpu().detach().item(), 6)))
            )
        )

        # run backward pass
        train_batch_loss.backward()

        # run optimizer step
        optimizer.step()

        # case: eval iteration
        if i % parameter['eval_iteration'] == 0:

            # save client model checkpoint
            file_name = '{}_ae_baseline_model_checkpoint_itr_{}.pth'.format(parameter['exp_timestamp'], str(i).zfill(6))
            uha.save_client_model_checkpoint(filename=file_name, iteration=i, model=model, optimizer=optimizer, chpt_dir=parameter['log_sub_dir'])

    #### start evaluation routine

    # init the EY data loader
    eval_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=False)

    # set model in evaluation mode
    model.eval()

    # init validation embeddings
    valid_embeddings = []

    # init number of batch iterations
    batch_count = 0

    # case: mini-batches are still available
    for batch, _ in eval_loader:

        # push the inputs to compute device
        batch = batch.to(parameter['device'])

        # run model forward pass
        valid_embeddings_batch, _ = model(batch)

        # case: initial batch
        if batch_count == 0:

            # collect validation embeddings
            valid_embeddings = valid_embeddings_batch.cpu().detach().numpy()

        # case: non-initial batch
        else:

            # collect validation embeddings
            valid_embeddings = np.vstack((valid_embeddings, valid_embeddings_batch.cpu().detach().numpy()))

        # increase batch iteration
        batch_count += 1

    # update journal entries with embedding
    aggregated_entries['z1'] = valid_embeddings[:, 0]
    aggregated_entries['z2'] = valid_embeddings[:, 1]

    # log aggregated categorical entries
    file_name = '{}_sampled_aggregated_entries_embedded_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
    aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

    # set visualization handler directory
    vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

    # visualize learned embeddings
    filename = '{}_je_embedding_distribution_it_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['iterations']).zfill(6))
    title = 'AE Baseline - JE Embedding Distribution\niterations: {}, train-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)))
    vha.plot_embeddings_2d(data=aggregated_entries, z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

    # visualize learned embeddings
    filename = '{}_je_embedding_distribution_it_{}_at_PREPARER.png'.format(str(parameter['exp_timestamp']), str(parameter['iterations']).zfill(6))
    title = 'AE Baseline - JE Embedding Distribution\nattribute: Y_PREPARER_ID, iterations: {}, train-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)))
    vha.plot_embeddings_2d_attribute(data=aggregated_entries, attribute='Y_PREPARER_ID', z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

    # visualize learned embeddings
    filename = '{}_je_embedding_distribution_it_{}_at_SOURCE.png'.format(str(parameter['exp_timestamp']), str(parameter['iterations']).zfill(6))
    title = 'AE Baseline - JE Embedding Distribution\nattribute: Y_SOURCE, iterations: {}, train-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)))
    vha.plot_embeddings_2d_attribute(data=aggregated_entries, attribute='Y_SOURCE', z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

# define client main
def main_gnn():

    # init client argument parser
    parser = argparse.ArgumentParser(description='deepApple Experiments')

    # general experiment parameter
    parser.add_argument('-exp_timestamp', help='', nargs='?', type=str,  default=dt.datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S'))
    parser.add_argument('-data_dir', help='', nargs='?', type=str, default='./100_datasets')
    parser.add_argument('-base_dir', help='', nargs='?', type=str, default='./200_experiments')
    parser.add_argument('-exp_postfix', type=str, default='poc', help='postfix of experimental runs.')

    # data parameter
    parser.add_argument('-dataset', help='', nargs='?', type=str,  default='ernstyoung')
    parser.add_argument('-sample_eval', help='', nargs='?', type=str, default='False')
    parser.add_argument('-sample_size', help='', nargs='?', type=int, default=10000)

    # model architecture parameter
    parser.add_argument('-seed', type=int, default=1111, help='seed value for deterministic results.')
    parser.add_argument('-hidden_dim', type=int, default=64, help='the dimension of the first gnn layer.')
    parser.add_argument('-embed_dim', type=int, default=2, help='the dimension of the graph embeddings.')
    parser.add_argument('-kl_div_alpha', type=float, default=0.0, help='the kl-divergence loss regularizer.')

    # model training parameter
    parser.add_argument('-iterations', type=int, default=1001, help='the number of training iterations.')
    parser.add_argument('-eval_iteration', type=int, default=100, help='the eval training iteration.')
    parser.add_argument('-batch_size', type=int, default=128, help='the batch size.')
    parser.add_argument('-learning_rate', type=float, default=0.0001, help='the learning rate.')
    parser.add_argument('-device', type=str, default='cpu', help='the compute device.')

    # ignore potential warnings - mostly due to package updates
    warnings.filterwarnings('ignore')

    # parse client arguments
    parameter = vars(parser.parse_args())

    # init new handlers
    uha = UtilsHandler.UtilsHandler()
    dha = DataHandler.DataHandler()
    vha = VisualisationHandler.VisualizationHandler()

    # parse boolean args as boolean
    parameter['sample_eval'] = uha.str2bool(parameter['sample_eval'])

    # set deterministic seeds of the client training experiments
    np.random.seed(int(parameter['seed']))  # set numpy seed
    th.manual_seed(int(parameter['seed']))  # set pytorch seed CPU
    th.cuda.manual_seed(int(parameter['seed']))  # set pytorch seed GPU

    # create experiment directory
    parameter['exp_dir'], parameter['par_sub_dir'], parameter['sta_sub_dir'], parameter['res_sub_dir'], parameter['vis_sub_dir'], parameter['log_sub_dir'] = uha.create_experiment_directory(param=parameter, parent_dir=parameter['base_dir'], architecture='ae_graph')

    # load the EY training data
    posting_ids, adj_matrices, feat_matrices, aggregated_entries, entries_statistics = dha.get_ernstyoung_gnn_data_range(parameter=parameter)

    # log aggregated categorical entries
    file_name = '{}_sampled_processed_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
    #processed_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

    # log aggregated categorical entries
    file_name = '{}_sampled_aggregated_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
    #aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

    # log aggregated categorical entries
    file_name = '{}_sampled_prepared_entries_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['exp_postfix']))
    #prepared_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

    # convert the EY training data to pytorch tensor
    prepared_tensor_entries = AccountingGNNDataset(adj_matrices=adj_matrices, feat_matrices=feat_matrices)

    #### start training routine

    # init the EY data loader
    train_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=True)

    # init the autoencoder model
    model = GNNAutoencoder.GNNAutoencoder(
        input_dim=entries_statistics['no_accounts'],
        hidden_dim=parameter['hidden_dim'],
        embed_dim=parameter['embed_dim'],
        output_dim=entries_statistics['no_accounts'] * (entries_statistics['no_accounts'] + 1) // 2,
        device=parameter['device']
    ).to(parameter['device'])

    # init aggregated categorical autoencoder loss
    rec_criterion = th.nn.BCELoss().to(parameter['device'])

    # init training optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=parameter['learning_rate'])

    # set model in train mode
    model.train()

    # init the training loss
    average_train_loss = 0.0

    # push aggregated losses to compute device
    rec_criterion = rec_criterion.to(parameter['device'])

    # init and wrap range of training iterations
    training_iterations = tqdm(range(0, parameter['iterations']))

    # case: mini-batches are still available
    for i in training_iterations:

        # get next training batch
        adj_matrices_batch, feat_matrices_batch = next(iter(train_loader))

        # push the inputs and targets to compute device
        adj_matrices_batch, feat_matrices_batch = adj_matrices_batch.to(parameter['device']), feat_matrices_batch.to(parameter['device'])

        # reset optimizer gradients
        optimizer.zero_grad()

        # clamp reconstructed matrix
        adj_matrices_batch = th.clamp(adj_matrices_batch, min=0.0, max=1.0)

        # run model forward pass
        _, mu, sigma, feat_matrices_recon, adj_matrices_recon = model(feat_matrices_batch, adj_matrices_batch)

        # compute categorical reconstruction loss
        train_batch_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch)

        # compute kl-divergence loss (https://en.wikipedia.org/wiki/Kullback–Leibler_divergence -> Normal Distribution)
        train_batch_kl_div_loss = parameter['kl_div_alpha'] * (-0.5 * th.sum(1 + sigma - mu.pow(2) - sigma.exp()))

        # compute train batch loss
        train_batch_loss = train_batch_rec_loss + train_batch_kl_div_loss

        # compute and collect average reconstruction loss
        average_train_loss += train_batch_loss.cpu().detach().item()

        # log training progress
        now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
        training_iterations.set_description(
            (
                '[INFO {}] DeepAppleGraph :: iteration: {}, train-loss: {}, train-rec-loss: {}, train-kld-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(train_batch_loss.cpu().detach().item(), 6)), str(np.round(train_batch_rec_loss.cpu().detach().item(), 6)), str(np.round(train_batch_kl_div_loss.cpu().detach().item(), 6)))
            )
        )

        # run backward pass
        train_batch_loss.backward()

        # run optimizer step
        optimizer.step()

        # case: eval iteration
        if i % parameter['eval_iteration'] == 0:

            # save client model checkpoint
            file_name = '{}_ae_gnn_model_checkpoint_itr_{}.pth'.format(parameter['exp_timestamp'], str(i).zfill(6))
            uha.save_client_model_checkpoint(filename=file_name, iteration=i, model=model, optimizer=optimizer, chpt_dir=parameter['log_sub_dir'])

    #### start evaluation routine

    # init the EY data loader
    eval_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=False)

    # init aggregated categorical autoencoder loss
    rec_criterion = th.nn.BCELoss(reduce=False).to(parameter['device'])

    # set model in evaluation mode
    model.eval()

    # init validation reconstruction losses
    valid_losses = []

    # init validation embeddings
    valid_embeddings = []

    # init number of batch iterations
    batch_count = 0

    # case: mini-batches are still available
    for adj_matrices_batch, feat_matrices_batch in eval_loader:

        # push the inputs to compute device
        adj_matrices_batch = adj_matrices_batch.to(parameter['device'])
        feat_matrices_batch = feat_matrices_batch.to(parameter['device'])

        # run model forward pass
        valid_embeddings_batch, _, _, feat_matrices_recon, adj_matrices_recon = model(feat_matrices_batch, adj_matrices_batch)

        # compute and add categorical reconstruction loss
        train_batch_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch).mean(axis=1)

        # case: initial batch
        if batch_count == 0:

            # collect validation losses
            valid_losses = train_batch_loss.cpu().detach().numpy()

            # collect validation embeddings
            valid_embeddings = valid_embeddings_batch.cpu().detach().numpy()

        # case: non-initial batch
        else:

            # collect validation losses
            valid_losses = np.vstack((valid_losses, train_batch_loss.cpu().detach().numpy()))

            # collect validation embeddings
            valid_embeddings = np.vstack((valid_embeddings, valid_embeddings_batch.cpu().detach().numpy()))

        # increase batch iteration
        batch_count += 1

    # update journal entries with embedding
    aggregated_entries['error'] = valid_losses.mean(axis=1)
    aggregated_entries['z1'] = valid_embeddings[:, 0]
    aggregated_entries['z2'] = valid_embeddings[:, 1]

    # log aggregated categorical entries
    file_name = '{}_aggregated_entries_embedded_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
    aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

    # set visualization handler directory
    vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

    # visualize learned embeddings
    filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
    title = 'GNN AE Baseline - JE Embedding Distribution\niterations: {}, train-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)))
    vha.plot_embeddings_2d(data=aggregated_entries, z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

    # visualize learned embeddings in specific interval
    filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_interval.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
    title = 'GNN AE Baseline - JE Embedding Distribution\niterations: {}, train-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)))
    vha.plot_embeddings_2d_interval(data=aggregated_entries, z1_col_name='z1', z2_col_name='z2', c_col_name='error', filename=filename, title=title, xlim=[-5.0, 5.0], ylim=[-5.0, 5.0]) # xlim=[18.5, 20.1], ylim=[-10.2, -11.5]

# run main function
if __name__ == '__main__':

    # call main function
    # main()
    main_gnn()