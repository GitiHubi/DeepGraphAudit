# import python libraries
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import additional python libraries
import datetime as dt
import numpy as np
from tqdm import tqdm

# import wandb logging
import wandb

# import pytorch libraries
import torch as th
from torch.utils.data import DataLoader

# import project libraries
from UtilsHandler import UtilsHandler
from DataHandler import DataHandler
from DataHandler.AccountingGNNDataset import AccountingGNNDataset
from ModelHandler import GNNAutoencoder
from VisualisationHandler import VisualisationHandler

# class GraphAutoencoderExperiment
class GraphAutoencoderExperiment(object):

    # init class constructor
    def __init__(self):

        # init utils handler
        self.uha = UtilsHandler.UtilsHandler()

        # init data handler
        self.dha = DataHandler.DataHandler()

        # init visualization handler
        self.vha = VisualisationHandler.VisualizationHandler()

    # run graph autoencoder experiment
    def run_experiement(self, parameter):

        # case: wandb logging enabled
        if parameter['wandb']:

            # init weights and biases log
            run = wandb.init(project='GraphExperiments', group='{}'.format(str(parameter['base_dir'].split('/')[-1])), name='{}_ae_graph_sd_{}_ds_{}_{}'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['dataset']), str(parameter['exp_postfix'])))

            # add wandb experiment configuration
            run.config.update(parameter)

            # init wandb log dictionary
            wandb_logging = {}

        # create experiment directory
        parameter['exp_dir'], parameter['par_sub_dir'], parameter['sta_sub_dir'], parameter['res_sub_dir'], parameter['vis_sub_dir'], parameter['log_sub_dir'] = self.uha.create_experiment_directory(param=parameter, parent_dir=parameter['base_dir'], architecture='ae_graph')

        # load the EY training data
        posting_ids, adj_matrices, feat_matrices, aggregated_entries, entries_statistics = self.dha.get_ernstyoung_gnn_data_range(parameter=parameter)

        # convert the EY training data to pytorch tensor
        prepared_tensor_entries = AccountingGNNDataset(adj_matrices=adj_matrices, feat_matrices=feat_matrices)

        #### start training routine

        # init the EY data loader
        train_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=True, drop_last=False)

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

        # run the model training
        model, average_train_loss = self.run_model_training(parameter=parameter, model=model, rec_criterion=rec_criterion, train_loader=train_loader, optimizer=optimizer, wandb_logging=wandb_logging, run=run)

        #### start evaluation routine

        # init the EY data loader
        eval_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=False, drop_last=False)

        # init aggregated categorical autoencoder loss
        rec_criterion = th.nn.BCELoss().to(parameter['device'])

        # init aggregated categorical autoencoder loss
        rec_criterion_details = th.nn.BCELoss(reduce=False).to(parameter['device'])

        # run the model evaluation
        aggregated_entries, average_valid_loss = self.run_model_validation(parameter=parameter, model=model, rec_criterion=rec_criterion, rec_criterion_details=rec_criterion_details, eval_loader=eval_loader, aggregated_entries=aggregated_entries, wandb_logging=wandb_logging, run=run)

        # log aggregated categorical entries
        file_name = '{}_aggregated_entries_embedded_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # set visualization handler directory
        self.vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

        # run the model visualization
        self.run_model_visualization(parameter=parameter, aggregated_entries=aggregated_entries, average_train_loss=average_train_loss, average_valid_loss=average_valid_loss)

        # case: wandb logging enabled
        if parameter['wandb']:

            # finish wandb run
            run.finish()

    # run the model training
    def run_model_training(self, parameter, model, rec_criterion, train_loader, optimizer, run, wandb_logging):

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
            # adj_matrices_batch = th.clamp(adj_matrices_batch, min=0.0, max=1.0)

            # run model forward pass
            _, mu, sigma, feat_matrices_recon, adj_matrices_recon = model(feat_matrices_batch, adj_matrices_batch)

            # compute categorical reconstruction loss
            train_batch_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch)

            # compute kl-divergence loss
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

            # case: wandb logging enabled
            if parameter['wandb']:

                # fill wandb log dict
                wandb_logging['001_model_training/avg_train_loss'] = average_train_loss / (i + 1)

                # log training progress
                run.log(wandb_logging)

            # case: eval iteration
            if i % parameter['eval_iteration'] == 0:

                # save client model checkpoint
                file_name = '{}_ae_gnn_model_checkpoint_itr_{}.pth'.format(parameter['exp_timestamp'], str(i).zfill(6))
                self.uha.save_client_model_checkpoint(filename=file_name, iteration=i, model=model, optimizer=optimizer, chpt_dir=parameter['log_sub_dir'])

        # return model training results
        return model, average_train_loss

    # run the model evaluation
    def run_model_validation(self, parameter, model, rec_criterion, rec_criterion_details, eval_loader, aggregated_entries, run, wandb_logging):

        # set model in evaluation mode
        model.eval()

        # init validation reconstruction losses
        average_valid_loss = 0.0
        valid_losses = []

        # init validation embeddings
        valid_embeddings = []

        # init and wrap range of training iterations
        validation_iterations = tqdm(range(0, len(eval_loader)))

        # case: mini-batches are still available
        for i, (adj_matrices_batch, feat_matrices_batch) in enumerate(eval_loader):

            # update validation iterations
            validation_iterations.update(i)

            # push the inputs to compute device
            adj_matrices_batch = adj_matrices_batch.to(parameter['device'])
            feat_matrices_batch = feat_matrices_batch.to(parameter['device'])

            # clamp reconstructed matrix
            # adj_matrices_batch = th.clamp(adj_matrices_batch, min=0.0, max=1.0)

            # run model forward pass
            valid_embeddings_batch, _, _, feat_matrices_recon, adj_matrices_recon = model(feat_matrices_batch, adj_matrices_batch)

            # compute and add categorical reconstruction loss
            valid_batch_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch)

            # compute and add categorical reconstruction loss
            valid_batch_loss_details = rec_criterion_details(input=adj_matrices_recon, target=adj_matrices_batch).mean(axis=1)

            # compute and collect average reconstruction loss
            average_valid_loss += valid_batch_rec_loss.cpu().detach().item()

            # log validation progress
            now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
            validation_iterations.set_description(
                (
                    '[INFO {}] DeepAppleGraph :: iteration: {}, valid-loss: {}, train-rec-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(valid_batch_rec_loss.cpu().detach().item(), 6)), str(np.round(valid_batch_rec_loss.cpu().detach().item(), 6)))
                )
            )

            # case: initial batch
            if i == 0:

                # collect validation losses
                valid_losses = valid_batch_loss_details.cpu().detach().numpy()

                # collect validation embeddings
                valid_embeddings = valid_embeddings_batch.cpu().detach().numpy()

            # case: non-initial batch
            else:

                # collect validation losses
                valid_losses = np.vstack((valid_losses, valid_batch_loss_details.cpu().detach().numpy()))

                # collect validation embeddings
                valid_embeddings = np.vstack((valid_embeddings, valid_embeddings_batch.cpu().detach().numpy()))

                # case: wandb logging enabled
            if parameter['wandb']:

                # fill wandb log dict
                wandb_logging['002_model_validation/avg_eval_loss'] = average_valid_loss / (i + 1)

                # log training progress
                run.log(wandb_logging)

        # update journal entries with embedding
        aggregated_entries['error'] = valid_losses.mean(axis=1)
        aggregated_entries['z1'] = valid_embeddings[:, 0]
        aggregated_entries['z2'] = valid_embeddings[:, 1]

        # close validation iteration
        validation_iterations.close()

        # return model evaluation results
        return aggregated_entries, average_valid_loss

    # run the model and result visualization
    def run_model_visualization(self, parameter, aggregated_entries, average_train_loss, average_valid_loss):

        # visualize learned embeddings
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)), str(np.round((average_valid_loss / parameter['iterations']), 6)))
        self.vha.plot_embeddings_2d(data=aggregated_entries, z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

        # visualize learned embeddings in specific interval
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_interval.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)), str(np.round((average_valid_loss / parameter['iterations']), 6)))
        self.vha.plot_embeddings_2d_interval(data=aggregated_entries, z1_col_name='z1', z2_col_name='z2', c_col_name='error', filename=filename, title=title, xlim=[-5.0, 5.0], ylim=[-5.0, 5.0]) # xlim=[18.5, 20.1], ylim=[-10.2, -11.5]

        # visualize learned embeddings interactively
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)), str(np.round((average_valid_loss / parameter['iterations']), 6)))
        self.vha.plot_embeddings_2d_interactive(data=aggregated_entries, z1_col_name='z1', z2_col_name='z2', c_col_name='error', filename=filename, title=title)