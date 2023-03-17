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

# import pytorch libraries
import torch as th
from torch.utils.data import DataLoader

# import project libraries
from UtilsHandler import UtilsHandler
from DataHandler import DataHandler
from AnomalyHandler import AnomalyHandler
from EvaluationHandler import EvaluationHandler
from DataHandler.AccountingGNNDatasetDynamic import AccountingGNNDatasetDynamic
from ModelHandler import GNNAutoencoderDynamic
from VisualisationHandler import VisualisationHandler
from LoggingHandler import LoggingHandler

# class GraphAutoencoderExperiment
class GraphAutoencoderExperimentDynamic(object):

    # init class constructor
    def __init__(self):

        # init utils handler
        self.uha = UtilsHandler.UtilsHandler()

        # init data handler
        self.dha = DataHandler.DataHandler()

        # init evaluation handler
        self.eha = EvaluationHandler.EvaluationHandler()

        # init anomaly handler
        self.aha = AnomalyHandler.AnomalyHandler()

        # init visualization handler
        self.vha = VisualisationHandler.VisualisationHandler()

        # init logging handler
        self.lha = LoggingHandler.LoggingHandler()

    # run graph autoencoder experiment
    def run_experiement(self, parameter, data_statistics):

        # case: wandb logging enabled
        if parameter['wandb']:

            # init wandb logging
            self.lha.init_wandb_run(project='DeepAppleGNN_{}_{}'.format(str(parameter['exp_series']).zfill(3), str(parameter['dataset'])), parameter=parameter)

        # create experiment directory
        parameter['exp_dir'], parameter['par_sub_dir'], parameter['sta_sub_dir'], parameter['res_sub_dir'], parameter['vis_sub_dir'], parameter['log_sub_dir'], parameter['gra_sub_dir'] = self.uha.create_experiment_directory(param=parameter, parent_dir=parameter['base_dir'], architecture='ae_graph')

        # save experiment parameters
        self.uha.save_experiment_parameter(param=parameter, parameter_dir=parameter['par_sub_dir'])

        # init experiment logging
        file_name = '{}_experiment_log_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        self.lha.init_experiment_log(parameter=parameter, directory=parameter['sta_sub_dir'], file_name=file_name)

        # init dataset statistics
        experiment_statistics = {}

        # case: ey dataset
        if parameter['dataset'] == 'ey':

            # load the EY training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries_belnr_hkont, aggregated_entries_belnr, data_statistics = self.dha.get_gnn_data_range_ey(parameter=parameter, statistics=data_statistics)

        # case: serpro dataset
        elif parameter['dataset'] == 'serpro':

            # load the Serpro training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries, data_statistics = self.dha.get_gnn_data_range_serpro(parameter=parameter, statistics=data_statistics)

        # case: sap dataset
        elif parameter['dataset'] == 'sap':

            # load the Serpro training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries_belnr_hkont, aggregated_entries_belnr, data_statistics = self.dha.get_gnn_data_range_sap(parameter=parameter, statistics=data_statistics)

        # log aggregated entries
        file_name = '{}_aggregated_entries_all_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        aggregated_entries_belnr_hkont.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # log selected aggregated entries
        file_name = '{}_aggregated_entries_selected_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        aggregated_entries_belnr.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # determine the number of journal entries, accounts and features
        experiment_statistics['no_journal_entries'] = data_statistics['no_journal_entries']
        experiment_statistics['no_accounts'] = data_statistics['no_posting_accounts']
        experiment_statistics['no_features'] = data_statistics['no_posting_features'] * parameter['feat_embed_dim']

        # update the encoder input dim depending on the number of features
        parameter['encoder_dim'].insert(0, experiment_statistics['no_features'])
        parameter['decoder_dim'].insert(len(parameter['decoder_dim']), experiment_statistics['no_features'])

        # convert the training data to pytorch tensor
        prepared_tensor_entries = AccountingGNNDatasetDynamic(adj_matrices=adj_matrices, feat_matrices=feat_matrices)

        #### start training routine

        # init the train data loader -> batch size == 1 due to dynamic adjacency matrices
        train_loader = DataLoader(prepared_tensor_entries, batch_size=1, shuffle=True, drop_last=False)

        # init the eval data loader
        eval_loader = DataLoader(prepared_tensor_entries, batch_size=1, shuffle=False, drop_last=False)

        # init the graph autoencoder model
        model = GNNAutoencoderDynamic.GNNAutoencoderDynamic(
            statistics=data_statistics
            , feat_embed_dim=parameter['feat_embed_dim']
            , encoder_dim=parameter['encoder_dim']
            , encoder_bottleneck=parameter['encoder_bottleneck']
            , decoder_dim=parameter['decoder_dim']
            , decoder_bottleneck=parameter['decoder_bottleneck']
            , device=parameter['device']
        ).to(parameter['device'])

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DeepAppleGraph :: GNN Autoencoder Model {}.'.format(now, str(model)))

        # case: BCE loss training
        if parameter['loss'] == 'bce':

            # init aggregated BCE autoencoder loss
            rec_criterion = th.nn.BCELoss(reduce=True, reduction='mean').to(parameter['device'])

            # init detailed BCE autoencoder loss
            rec_criterion_details = th.nn.BCELoss(reduce=False).to(parameter['device'])

        # case: MSE loss training
        elif parameter['loss'] == 'mse':

            # init aggregated MSE autoencoder loss
            rec_criterion = th.nn.MSELoss(reduce=True, reduction='mean').to(parameter['device'])

            # init detailed MSE autoencoder loss
            rec_criterion_details = th.nn.MSELoss(reduce=False).to(parameter['device'])

        # init training optimizer
        optimizer = th.optim.Adam(model.parameters(), lr=parameter['learning_rate'], weight_decay=parameter['weight_decay'])

        # init learning rate scheduler
        scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=int(parameter['train_iterations'] / parameter['learning_rate_steps']), gamma=0.1)

        # run the model training
        model, experiment_statistics = self.run_model_training(parameter=parameter, data_statistics=data_statistics, experiment_statistics=experiment_statistics, model=model, rec_criterion=rec_criterion, rec_criterion_details=rec_criterion_details, train_loader=train_loader, eval_loader=eval_loader, optimizer=optimizer, scheduler=scheduler, aggregated_entries=aggregated_entries_belnr)

        #### start evaluation routine

        # run the model evaluation
        aggregated_entries_belnr, experiment_statistics = self.run_model_validation(parameter=parameter, data_statistics=data_statistics, experiment_statistics=experiment_statistics, model=model, rec_criterion=rec_criterion, rec_criterion_details=rec_criterion_details, eval_loader=eval_loader, aggregated_entries=aggregated_entries_belnr)

        # determine model evaluation measures
        experiment_statistics = self.eha.compute_error_pr_auc_score(statistics=data_statistics, results=experiment_statistics, entries=aggregated_entries_belnr)
        experiment_statistics = self.eha.compute_error_roc_auc_score(statistics=data_statistics, results=experiment_statistics, entries=aggregated_entries_belnr)

        # run anomaly detection routine
        experiment_statistics, aggregated_entries_belnr = self.aha.run_anomaly_detection(parameter=parameter, results=experiment_statistics, entries=aggregated_entries_belnr)

        # determine model evaluation measures
        experiment_statistics = self.eha.compute_density_pr_auc_score(statistics=data_statistics, results=experiment_statistics, entries=aggregated_entries_belnr)
        experiment_statistics = self.eha.compute_error_pr_auc_score(statistics=data_statistics, results=experiment_statistics, entries=aggregated_entries_belnr)

        # determine current learning rate
        experiment_statistics['learning_rate'] = optimizer.state_dict()['param_groups'][0]['lr']

        # log aggregated entries
        file_name = '{}_aggregated_entries_selected_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        aggregated_entries_belnr.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # log experiment results
        file_name = '{}_experiment_log_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        self.lha.save_experiment_log(parameter=parameter, experiment_statistics=experiment_statistics, directory=parameter['sta_sub_dir'], file_name=file_name)

        # set visualization handler directory
        self.vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

        # run the model visualization
        self.run_model_visualization(parameter=parameter, data_statistics=data_statistics, experiment_statistics=experiment_statistics, data=aggregated_entries_belnr, average_train_loss=experiment_statistics['average_train_loss'], average_valid_loss=experiment_statistics['average_valid_loss'], iteration=parameter['train_iterations'])

        # case: wandb logging enabled
        if parameter['wandb']:

            # finalize and close wandb logging
            self.lha.close_wandb_run(parameter=parameter, statistics=experiment_statistics)

        # return the experiment statistics
        return experiment_statistics

    # run the model training
    def run_model_training(self, parameter, data_statistics, experiment_statistics, model, rec_criterion, rec_criterion_details, train_loader, eval_loader, optimizer, scheduler, aggregated_entries):

        # set model in train mode
        model.train()

        # init the training loss
        average_train_iteration_loss = 0.0
        average_adj_train_iteration_loss = 0.0
        average_fea_train_iteration_loss = 0.0

        # init experiment loss statistics
        experiment_statistics['average_train_loss'] = 0.0
        experiment_statistics['average_fea_train_loss'] = 0.0
        experiment_statistics['average_adj_train_loss'] = 0.0

        # init experiment error statistics
        experiment_statistics['error_roc_auc_all'] = 0.0
        experiment_statistics['error_roc_auc_global'] = 0.0
        experiment_statistics['error_roc_auc_local'] = 0.0

        experiment_statistics['error_pr_auc_all'] = 0.0
        experiment_statistics['error_pr_auc_global'] = 0.0
        experiment_statistics['error_pr_auc_local'] = 0.0

        # init experiment density statistics
        experiment_statistics['density_roc_auc_all'] = 0.0
        experiment_statistics['density_roc_auc_global'] = 0.0
        experiment_statistics['density_roc_auc_local'] = 0.0

        experiment_statistics['density_pr_auc_all'] = 0.0
        experiment_statistics['density_pr_auc_global'] = 0.0
        experiment_statistics['density_pr_auc_local'] = 0.0

        # push aggregated losses to compute device
        rec_criterion = rec_criterion.to(parameter['device'])

        # init and wrap range of training iterations
        training_iterations = tqdm(range(0, parameter['train_iterations']))

        # iterate over validation iterations
        for i in training_iterations:

            # reset optimizer gradients
            optimizer.zero_grad()

            # init the train batch loss
            train_batch_feat_rec_loss = th.tensor([0.0], requires_grad=True).to(parameter['device'])
            train_batch_adj_rec_loss = th.tensor([0.0], requires_grad=True).to(parameter['device'])

            # iterate over mini-batch elements
            for k in range(0, parameter['train_batch_size']):

                # get next training batch
                adj_matrices_batch, feat_matrices_batch = next(iter(train_loader))

                # push the inputs and targets to compute device
                adj_matrices_batch, feat_matrices_batch = adj_matrices_batch.to(parameter['device']), feat_matrices_batch.to(parameter['device'])

                # determine feature embeddings
                feat_matrices_batch = model.embedd_features_batch(feat_matrices_batch)

                # run model forward pass
                _, mu, sigma, feat_matrices_recon, adj_matrices_recon = model(feat_matrices_batch, adj_matrices_batch)

                # init feature reconstruction loss
                #train_batch_feat_rec_loss = th.zeros(1).to(parameter['device'])

                # iterate over dataset features
                #for k, feature in enumerate(data_statistics['je_features']):

                    # determine original and reconstructed feature encodings
                    #feat_matrices_batch_encodings = feat_matrices_batch[:, :, k * parameter['feat_embed_dim']: (k+1) * parameter['feat_embed_dim']]
                    #feat_matrices_recon_encodings = feat_matrices_recon[:, :, k * parameter['feat_embed_dim']: (k+1) * parameter['feat_embed_dim']]

                    # compute feature vector loss
                    # train_batch_feat_rec_loss += rec_criterion(input=feat_matrices_recon_encodings, target=feat_matrices_batch_encodings)

                # determine feature matrix normalization factor
                feat_norm = feat_matrices_batch.shape[1] * feat_matrices_batch.shape[1] # / float((feat_matrices_batch.shape[1] * feat_matrices_batch.shape[1] - feat_matrices_batch.sum()) * 2)

                # determine individual sample feature loss
                train_sample_feat_rec_loss = rec_criterion(input=feat_matrices_recon, target=feat_matrices_batch) / feat_norm

                # compute feature vector loss
                train_batch_feat_rec_loss = train_batch_feat_rec_loss + train_sample_feat_rec_loss

                # determine adjacency matrix normalization factor
                adj_norm = adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] # / float((adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] - adj_matrices_batch.sum()) * 2)

                # determine individual sample adjacency loss
                train_sample_adj_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch) / adj_norm

                # compute adjacency matrix loss
                train_batch_adj_rec_loss = train_batch_adj_rec_loss + train_sample_adj_rec_loss

            # compute train batch loss
            train_batch_loss = (parameter['beta'] * train_batch_feat_rec_loss + (1.0 - parameter['beta']) * train_batch_adj_rec_loss) / parameter['train_batch_size']

            # determine average train batch loss
            average_train_batch_loss = train_batch_loss / parameter['train_batch_size']
            average_fea_train_batch_loss = train_batch_feat_rec_loss / parameter['train_batch_size']
            average_adj_train_batch_loss = train_batch_adj_rec_loss / parameter['train_batch_size']

            # log training progress
            now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
            training_iterations.set_description(
                (
                    '[INFO {}] DeepAppleGraph :: train iteration: {}, lr: {}, av-batch-loss: {}, av-batch-feat-loss: {}, av-batch-adj-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(optimizer.state_dict()['param_groups'][0]['lr'], 6)), str(np.round(average_train_batch_loss.cpu().detach().item(), 8)), str(np.round(average_fea_train_batch_loss.cpu().detach().item(), 8)), str(np.round(average_adj_train_batch_loss.cpu().detach().item(), 8)))
                )
            )

            # run backward pass
            train_batch_loss.backward()

            # run optimizer step
            optimizer.step()

            # update learning rate
            scheduler.step()

            # compute and collect average reconstruction loss
            average_train_iteration_loss += average_train_batch_loss.cpu().detach().item()
            average_fea_train_iteration_loss += average_fea_train_batch_loss.cpu().detach().item()
            average_adj_train_iteration_loss += average_adj_train_batch_loss.cpu().detach().item()

            # determine finale average losses per iteration
            experiment_statistics['average_train_loss'] = average_train_iteration_loss / (i + 1)
            experiment_statistics['average_fea_train_loss'] = average_fea_train_iteration_loss / (i + 1)
            experiment_statistics['average_adj_train_loss'] = average_adj_train_iteration_loss / (i + 1)
            experiment_statistics['learning_rate'] = optimizer.state_dict()['param_groups'][0]['lr']

            # case: eval iteration
            if (i % parameter['valid_iterations'] == 0) and (i > 0):

                # run model evaluation
                aggregated_entries, experiment_statistics = self.run_model_validation(parameter, data_statistics, experiment_statistics, model, rec_criterion, rec_criterion_details, eval_loader, aggregated_entries)

                # determine model error evaluation measures
                experiment_statistics = self.eha.compute_error_roc_auc_score(statistics=data_statistics, results=experiment_statistics, entries=aggregated_entries)
                experiment_statistics = self.eha.compute_error_pr_auc_score(statistics=data_statistics, results=experiment_statistics, entries=aggregated_entries)

                # run anomaly detection routine
                experiment_statistics, aggregated_entries = self.aha.run_anomaly_detection(parameter=parameter, results=experiment_statistics, entries=aggregated_entries)

                # determine model density evaluation measures
                experiment_statistics = self.eha.compute_density_roc_auc_score(statistics=data_statistics, results=experiment_statistics, entries=aggregated_entries)
                experiment_statistics = self.eha.compute_density_pr_auc_score(statistics=data_statistics, results=experiment_statistics, entries=aggregated_entries)

                # determine current learning rate
                experiment_statistics['learning_rate'] = optimizer.state_dict()['param_groups'][0]['lr']

                # log experiment results
                file_name = '{}_experiment_log_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
                self.lha.save_experiment_log(parameter=parameter, experiment_statistics=experiment_statistics, directory=parameter['sta_sub_dir'], file_name=file_name)

                # set visualization handler directory
                self.vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

                # run the result visualization
                self.run_model_visualization(parameter=parameter, data_statistics=data_statistics, experiment_statistics=experiment_statistics, data=aggregated_entries, average_train_loss=experiment_statistics['average_train_loss'], average_valid_loss=experiment_statistics['average_valid_loss'], iteration=i)

                # save client model checkpoint
                file_name = '{}_ae_gnn_model_checkpoint_itr_{}.pth'.format(parameter['exp_timestamp'], str(i).zfill(6))
                # self.uha.save_client_model_checkpoint(filename=file_name, iteration=i, model=model, optimizer=optimizer, chpt_dir=parameter['log_sub_dir'])

            # case: wandb logging enabled
            if parameter['wandb']:

                # log training progress
                self.lha.update_wandb_run(statistics=experiment_statistics)

        # return model training results
        return model, experiment_statistics

    # run the model evaluation
    def run_model_validation(self, parameter, data_statistics, experiment_statistics, model, rec_criterion, rec_criterion_details, eval_loader, aggregated_entries):

        # set model in evaluation mode
        model.eval()

        # init the validation loss
        average_valid_iteration_loss = 0.0
        average_adj_valid_iteration_loss = 0.0
        average_fea_valid_iteration_loss = 0.0

        # init detailed validation reconstruction losses
        valid_losses = np.zeros([0])

        # init validation embeddings
        valid_embeddings = np.zeros([0])

        # determine number of valid iterations
        no_batches = np.ceil(len(eval_loader.dataset.adj_matrices) / parameter['valid_batch_size'])

        # init and wrap range of training iterations
        validation_iterations = tqdm(range(0, int(no_batches)))

        # init dataloader iterator -> ToDo: do quality assurance around this hack :D
        eval_iterator = iter(eval_loader)

        # init regular batch size
        batch_size = parameter['valid_batch_size']

        # disable gradient computation
        with th.no_grad():

            # iterate over validation iterations
            for i in validation_iterations:

                # init the valid batch loss
                valid_batch_feat_rec_loss = th.tensor([0.0], requires_grad=True).to(parameter['device'])
                valid_batch_adj_rec_loss = th.tensor([0.0], requires_grad=True).to(parameter['device'])

                # case: last (probably incomplete) batch reached
                if (i == no_batches-1) and (len(eval_loader.dataset.adj_matrices) % parameter['valid_batch_size'] != 0):

                    # update regular batch size
                    batch_size = len(eval_loader.dataset.adj_matrices) % parameter['valid_batch_size']

                # iterate over mini-batch elements
                for k in range(0, batch_size):

                    # get next training batch
                    adj_matrices_batch, feat_matrices_batch = next(eval_iterator)

                    # push the inputs to compute device
                    adj_matrices_batch, feat_matrices_batch = adj_matrices_batch.to(parameter['device']), feat_matrices_batch.to(parameter['device'])

                    # determine feature embeddings
                    feat_matrices_batch = model.embedd_features_batch(feat_matrices_batch)

                    # run model forward pass
                    valid_embeddings_batch, _, _, feat_matrices_recon, adj_matrices_recon = model(feat_matrices_batch, adj_matrices_batch)

                    # determine feature matrix normalization factor
                    feat_norm = feat_matrices_batch.shape[1] * feat_matrices_batch.shape[1] # / float((feat_matrices_batch.shape[1] * feat_matrices_batch.shape[1] - feat_matrices_batch.sum()) * 2)

                    # compute feature vector loss
                    valid_sample_feat_rec_loss = rec_criterion(input=feat_matrices_recon, target=feat_matrices_batch) / feat_norm

                    # compute feature vector loss
                    valid_batch_feat_rec_loss = valid_batch_feat_rec_loss + valid_sample_feat_rec_loss

                    # determine adjacency matrix normalization factor
                    adj_norm = adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] # / float((adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] - adj_matrices_batch.sum()) * 2)

                    # compute adjacency matrix loss
                    valid_sample_adj_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch) / adj_norm

                    # compute adjacency matrix loss
                    valid_batch_adj_rec_loss = valid_batch_adj_rec_loss + valid_sample_adj_rec_loss

                    # compute feature vector loss details -> Todo: think about removing
                    valid_batch_feat_rec_loss_details = rec_criterion_details(input=feat_matrices_recon, target=feat_matrices_batch).mean(axis=1).mean(axis=1) / feat_norm

                    # compute adjacency matrix loss details -> Todo: think about removing
                    valid_batch_adj_rec_loss_details = rec_criterion_details(input=adj_matrices_recon, target=adj_matrices_batch).mean(axis=1).mean(axis=1) / adj_norm

                    # compute and add adjacency and feature reconstruction loss
                    valid_batch_loss_details = parameter['beta'] * valid_batch_feat_rec_loss_details + (1.0 - parameter['beta']) * valid_batch_adj_rec_loss_details

                    # case: initial batch
                    if valid_losses.shape[0] == 0:

                        # collect validation losses
                        valid_losses = valid_batch_loss_details.cpu().detach().numpy()

                        # collect validation embeddings
                        valid_embeddings = valid_embeddings_batch.cpu().detach().numpy()

                    # case: non-initial batch
                    else:

                        # collect validation losses
                        valid_losses = np.hstack((valid_losses, valid_batch_loss_details.cpu().detach().numpy()))

                        # collect validation embeddings
                        valid_embeddings = np.vstack((valid_embeddings, valid_embeddings_batch.cpu().detach().numpy()))

                # compute valid batch loss
                valid_batch_loss = parameter['beta'] * valid_batch_feat_rec_loss + (1.0 - parameter['beta']) * valid_batch_adj_rec_loss

                # determine average train batch loss
                average_valid_batch_loss = valid_batch_loss / parameter['valid_batch_size']
                average_fea_valid_batch_loss = valid_batch_feat_rec_loss / parameter['valid_batch_size']
                average_adj_valid_batch_loss = valid_batch_adj_rec_loss / parameter['valid_batch_size']

                # log validation progress
                now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
                validation_iterations.set_description(
                    (
                        '[INFO {}] DeepAppleGraph :: valid iteration: {}, av-batch-loss: {}, av-batch-feat-loss: {}, av-batch-adj-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(average_valid_batch_loss.cpu().detach().item(), 8)), str(np.round(average_fea_valid_batch_loss.cpu().detach().item(), 8)), str(np.round(average_adj_valid_batch_loss.cpu().detach().item(), 8)))
                    )
                )

                # update iterations
                validation_iterations.update(1)

            # close iterations
            validation_iterations.close()

            # compute and collect average reconstruction loss
            average_valid_iteration_loss += average_valid_batch_loss.cpu().detach().item()
            average_fea_valid_iteration_loss += average_fea_valid_batch_loss.cpu().detach().item()
            average_adj_valid_iteration_loss += average_adj_valid_batch_loss.cpu().detach().item()

            # determine final average batch validation losses
            experiment_statistics['average_valid_loss'] = average_valid_iteration_loss / (i + 1)
            experiment_statistics['average_fea_valid_loss'] = average_fea_valid_iteration_loss / (i + 1)
            experiment_statistics['average_adj_valid_loss'] = average_adj_valid_iteration_loss / (i + 1)

            # update journal entries with embedding
            aggregated_entries['Y_REC_ERROR'] = valid_losses
            aggregated_entries['z1'] = valid_embeddings[:, 0]
            aggregated_entries['z2'] = valid_embeddings[:, 1]

        # return model evaluation results
        return aggregated_entries, experiment_statistics

    # run the model and result visualization
    def run_model_visualization(self, parameter, data_statistics, experiment_statistics, data, average_train_loss, average_valid_loss, iteration):

        # visualize learned embeddings
        filename = '{}_1_1_je_embedding_sd_{}_it_{}_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)))
        self.vha.plot_embeddings_2d(data=data, z1_col_name='z1', z2_col_name='z2', c_col_name=data_statistics['je_class_name_field'], filename=filename, title=title)

        # visualize reconstruction error
        filename = '{}_1_2_je_embedding_error_sd_{}_it_{}_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)))
        self.vha.plot_embeddings_2d_error(data=data, z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title)

        # visualize anomaly detection
        filename = '{}_1_3_je_embedding_anomalies_sd_{}_it_{}_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)))
        self.vha.plot_embeddings_2d_anomalies(data=data, z1_col_name='z1', z2_col_name='z2', c_col_name='Y_ANOMALY_CLASS', filename=filename, title=title)

        # visualize learned embeddings in specific interval
        #filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_interval.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        #title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)))
        #self.vha.plot_embeddings_2d_interval(data=data, z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title, xlim=[-5.0, 5.0], ylim=[-5.0, 5.0]) # xlim=[18.5, 20.1], ylim=[-10.2, -11.5]

        # case: visualize ey dataset
        if parameter['dataset'] == 'ey':

            # visualize learned embeddings interactively -> Todo Fix
            filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
            title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}'.format(str(parameter['dataset']).upper(), str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)))
            # self.vha.plot_embeddings_2d_interactive(data=data, hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title)
            # Open - self.vha.plot_ey_embeddings_2d_interactive(data=data, attributes=data_statistics['visual_attributes'], hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title)

        # case: visualize sap dataset
        elif parameter['dataset'] == 'sap':

            # visualize learned embeddings interactively -> Todo Fix
            filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
            title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}'.format(str(parameter['dataset']).upper(), str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)))
            # self.vha.plot_embeddings_2d_interactive(data=data, hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title)
            # Open - self.vha.plot_sap_embeddings_2d_interactive(data=data, attributes=data_statistics['visual_attributes'], hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title)

        # visualize learned embeddings interactively
        filename = '{}_2_1_je_embedding_sd_{}_it_{}_{}_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}<br>Anomaly-Algorithm: {}, Clusters: {}, Global-Anomalies: {}, Local-Anomalies: {}'.format(str(parameter['dataset']).upper(), str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)), str(parameter['algo']).upper(), str(experiment_statistics['no_detected_clusters']), str(experiment_statistics['no_detected_global_anomalies']), str(experiment_statistics['no_detected_local_anomalies']))
        self.vha.plot_embeddings_2d_interactive(data=data, hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name=data_statistics['je_class_name_field'], filename=filename, title=title)

        # visualize learned embeddings interactively
        filename = '{}_2_2_je_embedding_error_sd_{}_it_{}_{}_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}<br>Anomaly-Algorithm: {}, Clusters: {}, Global-Anomalies: {}, Local-Anomalies: {}'.format(str(parameter['dataset']).upper(), str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)), str(parameter['algo']).upper(), str(experiment_statistics['no_detected_clusters']), str(experiment_statistics['no_detected_global_anomalies']), str(experiment_statistics['no_detected_local_anomalies']))
        self.vha.plot_embeddings_2d_error_interactive(data=data, hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title)

        # visualize learned embeddings interactively
        filename = '{}_2_3_je_embedding_anomalies_sd_{}_it_{}_{}_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}<br>Anomaly-Algorithm: {}, Clusters: {}, Global-Anomalies: {}, Local-Anomalies: {}'.format(str(parameter['dataset']).upper(), str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)), str(parameter['algo']).upper(), str(experiment_statistics['no_detected_clusters']), str(experiment_statistics['no_detected_global_anomalies']), str(experiment_statistics['no_detected_local_anomalies']))
        self.vha.plot_embeddings_2d_anomalies_interactive(data=data, hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_ANOMALY_CLASS', filename=filename, title=title)

        # visualize learned embeddings interactively
        #filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_anomalies_cluster_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        #title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}<br>Anomaly-Algorithm: {}, Global-Anomalies: {}, Local-Anomalies: {}'.format(str(parameter['dataset']).upper(), str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)), str(parameter['algo']).upper(), str(data[data['Y_ANOMALY_CLASS'] == -1].shape[0]), str(data[data['Y_ANOMALY_CLASS'] == -2].shape[0]))
        #self.vha.plot_embeddings_2d_anomalies_cluster_interactive(data=data, hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_ANOMALY_CLASS', filename=filename, title=title)