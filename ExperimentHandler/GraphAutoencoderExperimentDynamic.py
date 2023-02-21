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
import pandas as pd
from tqdm import tqdm

# import pytorch libraries
import torch as th
from torch.utils.data import DataLoader

# import scikit learn libraries
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

# import hdbscan library
import hdbscan

# import project libraries
from UtilsHandler import UtilsHandler
from DataHandler import DataHandler
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

        # init visualization handler
        self.vha = VisualisationHandler.VisualizationHandler()

        # init logging handler
        self.lha = LoggingHandler.LoggingHandler()

    # run graph autoencoder experiment
    def run_experiement(self, parameter, data_parameter):

        # case: wandb logging enabled
        if parameter['wandb']:

            # init wandb logging
            self.lha.init_wandb_run(project='DeepAppleGNN', parameter=parameter)

        # create experiment directory
        parameter['exp_dir'], parameter['par_sub_dir'], parameter['sta_sub_dir'], parameter['res_sub_dir'], parameter['vis_sub_dir'], parameter['log_sub_dir'] = self.uha.create_experiment_directory(param=parameter, parent_dir=parameter['base_dir'], architecture='ae_graph')

        # save experiment parameters
        self.uha.save_experiment_parameter(param=parameter, parameter_dir=parameter['par_sub_dir'])

        # init experiment logging
        file_name = '{}_experiment_log_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        self.lha.init_experiment_log(parameter=parameter, file_name=file_name)

        # init dataset statistics
        experiment_statistics = {}

        # case: ey dataset
        if parameter['dataset'] == 'ey':

            # load the EY training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries_belnr_hkont, aggregated_entries_belnr, data_parameter = self.dha.get_gnn_data_range_ey(parameter=parameter, statistics=data_parameter)

        # case: serpro dataset
        elif parameter['dataset'] == 'serpro':

            # load the Serpro training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries, data_parameter = self.dha.get_gnn_data_range_serpro(parameter=parameter, statistics=data_parameter)

        # case: sap dataset
        elif parameter['dataset'] == 'sap':

            # load the Serpro training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries_belnr_hkont, aggregated_entries_belnr, data_parameter = self.dha.get_gnn_data_range_sap(parameter=parameter, statistics=data_parameter)

        # log aggregated entries
        file_name = '{}_aggregated_entries_all_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        aggregated_entries_belnr_hkont.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # log selected aggregated entries
        file_name = '{}_aggregated_entries_selected_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        aggregated_entries_belnr.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # determine the number of accounts and features
        experiment_statistics['no_accounts'] = data_parameter['no_posting_accounts']
        experiment_statistics['no_features'] = data_parameter['no_posting_features'] * parameter['feat_embed_dim']

        # update the encoder input dim depending on the number of features
        parameter['encoder_dim'].insert(0, experiment_statistics['no_features'])
        parameter['decoder_dim'].insert(len(parameter['decoder_dim']), experiment_statistics['no_features'])

        # convert the training data to pytorch tensor
        prepared_tensor_entries = AccountingGNNDatasetDynamic(adj_matrices=adj_matrices, feat_matrices=feat_matrices)

        #### start training routine

        # init the train data loader
        train_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['train_batch_size'], shuffle=True, drop_last=False)

        # init the eval data loader
        eval_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['valid_batch_size'], shuffle=False, drop_last=False)

        # init the graph autoencoder model
        model = GNNAutoencoderDynamic.GNNAutoencoderDynamic(
            statistics=data_parameter
            , feat_embed_dim=parameter['feat_embed_dim']
            , encoder_dim=parameter['encoder_dim']
            , bottleneck=parameter['bottleneck']
            , decoder_dim=parameter['decoder_dim']
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
        model, experiment_statistics = self.run_model_training(parameter=parameter, data_statistics=data_parameter, experiment_statistics=experiment_statistics, model=model, rec_criterion=rec_criterion, rec_criterion_details=rec_criterion_details, train_loader=train_loader, eval_loader=eval_loader, optimizer=optimizer, scheduler=scheduler, aggregated_entries=aggregated_entries_belnr)

        #### start evaluation routine

        # run the model evaluation
        aggregated_entries_belnr, experiment_statistics = self.run_model_validation(parameter=parameter, data_statistics=data_parameter, experiment_statistics=experiment_statistics, model=model, rec_criterion=rec_criterion, rec_criterion_details=rec_criterion_details, eval_loader=eval_loader, aggregated_entries=aggregated_entries_belnr)

        #### start anomaly detection routine
        aggregated_entries_belnr, global_anomalies, local_anomalies = self.run_anomaly_detection(parameter, aggregated_entries_belnr)

        # determine number of clusters
        experiment_statistics['no_clusters'] = int(len(aggregated_entries_belnr['Y_ANOMALY_CLASS'].unique())-2)

        # determine number of global and local anomalies
        experiment_statistics['no_global_anomalies'] = int(global_anomalies.shape[0])
        experiment_statistics['no_local_anomalies'] = int(local_anomalies.shape[0])

        # log aggregated entries
        file_name = '{}_aggregated_entries_selected_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        aggregated_entries_belnr.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # determine current learning rate
        experiment_statistics['learning_rate'] = optimizer.state_dict()['param_groups'][0]['lr']

        # log experiment results
        file_name = '{}_experiment_log_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
        self.lha.save_experiment_log(parameter=parameter, experiment_statistics=experiment_statistics, file_name=file_name)

        # set visualization handler directory
        self.vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

        # run the model visualization
        self.run_model_visualization(parameter=parameter, data_statistics=data_parameter, experiment_statistics=experiment_statistics, data=aggregated_entries_belnr, average_train_loss=experiment_statistics['average_train_loss'], average_valid_loss=experiment_statistics['average_valid_loss'], iteration=parameter['train_iterations'])

        # case: wandb logging enabled
        if parameter['wandb']:

            # finalize and close wandb logging
            self.lha.close_wandb_run(parameter=parameter, statistics=experiment_statistics)

    # run the model training
    def run_model_training(self, parameter, data_statistics, experiment_statistics, model, rec_criterion, rec_criterion_details, train_loader, eval_loader, optimizer, scheduler, aggregated_entries):

        # set model in train mode
        model.train()

        # init the training loss
        average_train_loss = 0.0
        average_adj_train_loss = 0.0
        average_fea_train_loss = 0.0

        # push aggregated losses to compute device
        rec_criterion = rec_criterion.to(parameter['device'])

        # init and wrap range of training iterations
        training_iterations = tqdm(range(0, parameter['train_iterations']))

        # case: mini-batches are still available
        for i in training_iterations:

            # get next training batch
            adj_matrices_batch, feat_matrices_batch = next(iter(train_loader))

            # push the inputs and targets to compute device
            adj_matrices_batch, feat_matrices_batch = adj_matrices_batch.to(parameter['device']), feat_matrices_batch.to(parameter['device'])

            # reset optimizer gradients
            optimizer.zero_grad()

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

            # determine adjacency matrix normalization factor
            # norm = adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] / float((adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] - adj_matrices_batch.sum()) * 2)

            # compute feature vector loss
            train_batch_feat_rec_loss = rec_criterion(input=feat_matrices_recon, target=feat_matrices_batch)

            # compute adjacency matrix loss
            train_batch_adj_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch)

            # compute train batch loss
            train_batch_loss = parameter['beta'] * train_batch_feat_rec_loss + (1.0 - parameter['beta']) * train_batch_adj_rec_loss

            # compute and collect average reconstruction loss
            average_train_loss += train_batch_loss.cpu().detach().item()
            average_adj_train_loss += train_batch_adj_rec_loss.cpu().detach().item()
            average_fea_train_loss += train_batch_feat_rec_loss.cpu().detach().item()

            # log training progress
            now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
            training_iterations.set_description(
                (
                    '[INFO {}] DeepAppleGraph :: iteration: {}, lr: {}, train-loss: {}, train-feat-loss: {}, train-adj-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(optimizer.state_dict()['param_groups'][0]['lr'], 6)), str(np.round(train_batch_loss.cpu().detach().item(), 8)), str(np.round(train_batch_feat_rec_loss.cpu().detach().item(), 8)), str(np.round(train_batch_adj_rec_loss.cpu().detach().item(), 8)))
                )
            )

            # run backward pass
            train_batch_loss.backward()

            # run optimizer step
            optimizer.step()

            # update learning rate
            scheduler.step()

            # determine finale average losses per iteration
            experiment_statistics['average_train_loss'] = average_train_loss / (i + 1)
            experiment_statistics['average_adj_train_loss'] = average_adj_train_loss / (i + 1)
            experiment_statistics['average_fea_train_loss'] = average_fea_train_loss / (i + 1)
            experiment_statistics['learning_rate'] = optimizer.state_dict()['param_groups'][0]['lr']

            # case: wandb logging enabled
            if parameter['wandb']:

                # log training progress
                self.lha.update_wandb_run(statistics=experiment_statistics)

            # case: eval iteration
            if (i % parameter['valid_iterations'] == 0) and (i > 0):

                # run model evaluation
                aggregated_entries, experiment_statistics = self.run_model_validation(parameter, data_statistics, experiment_statistics, model, rec_criterion, rec_criterion_details, eval_loader, aggregated_entries)

                # determine current learning rate
                experiment_statistics['learning_rate'] = optimizer.state_dict()['param_groups'][0]['lr']

                # run anomaly detection routine
                selected_aggregated_entries, global_anomalies, local_anomalies = self.run_anomaly_detection(parameter, aggregated_entries)

                # determine number of clusters
                experiment_statistics['no_clusters'] = int(len(selected_aggregated_entries['Y_ANOMALY_CLASS'].unique())-2)

                # determine number of global and local anomalies
                experiment_statistics['no_global_anomalies'] = int(global_anomalies.shape[0])
                experiment_statistics['no_local_anomalies'] = int(local_anomalies.shape[0])

                # log experiment results
                file_name = '{}_experiment_log_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['train_iterations']).zfill(6), str(parameter['exp_postfix']))
                self.lha.save_experiment_log(parameter=parameter, experiment_statistics=experiment_statistics, file_name=file_name)

                # set visualization handler directory
                self.vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

                # run the result visualization
                self.run_model_visualization(parameter=parameter, data_statistics=data_statistics, experiment_statistics=experiment_statistics, data=selected_aggregated_entries, average_train_loss=experiment_statistics['average_train_loss'], average_valid_loss=experiment_statistics['average_valid_loss'], iteration=i)

                # save client model checkpoint
                file_name = '{}_ae_gnn_model_checkpoint_itr_{}.pth'.format(parameter['exp_timestamp'], str(i).zfill(6))
                # self.uha.save_client_model_checkpoint(filename=file_name, iteration=i, model=model, optimizer=optimizer, chpt_dir=parameter['log_sub_dir'])

        # return model training results
        return model, experiment_statistics

    # run the model evaluation
    def run_model_validation(self, parameter, data_statistics, experiment_statistics, model, rec_criterion, rec_criterion_details, eval_loader, aggregated_entries):

        # set model in evaluation mode
        model.eval()

        # init validation reconstruction losses
        average_valid_loss = 0.0
        average_adj_valid_loss = 0.0
        average_fea_valid_loss = 0.0

        # init detailed validation reconstruction losses
        valid_losses = []

        # init validation embeddings
        valid_embeddings = []

        # init and wrap range of training iterations
        validation_iterations = tqdm(total=len(eval_loader))

        # disable gradient computation
        with th.no_grad():

            # case: mini-batches are still available
            for i, (adj_matrices_batch, feat_matrices_batch) in enumerate(eval_loader):

                # push the inputs to compute device
                adj_matrices_batch = adj_matrices_batch.to(parameter['device'])
                feat_matrices_batch = feat_matrices_batch.to(parameter['device'])

                # determine feature embeddings
                feat_matrices_batch = model.embedd_features_batch(feat_matrices_batch)

                # run model forward pass
                valid_embeddings_batch, _, _, feat_matrices_recon, adj_matrices_recon = model(feat_matrices_batch, adj_matrices_batch)

                ### compute batch reconstruction loss

                # init feature vector reconstruction loss
                # valid_batch_feat_rec_loss = th.zeros(1).to(parameter['device'])

                # iterate over dataset features
                #for k, feature in enumerate(data_statistics['je_features']):

                    # determine original and reconstructed feature encodings
                    #feat_matrices_batch_encodings = feat_matrices_batch[:, :, k * parameter['feat_embed_dim']: (k+1) * parameter['feat_embed_dim']]
                    #feat_matrices_recon_encodings = feat_matrices_recon[:, :, k * parameter['feat_embed_dim']: (k+1) * parameter['feat_embed_dim']]

                    # compute feature vector loss
                    #valid_batch_feat_rec_loss += rec_criterion(input=feat_matrices_recon_encodings, target=feat_matrices_batch_encodings)

                # determine adjacency matrix normalization factor
                # norm = adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] / float((adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] - adj_matrices_batch.sum()) * 2)

                # compute feature vector loss
                valid_batch_feat_rec_loss = rec_criterion(input=feat_matrices_recon, target=feat_matrices_batch)

                # compute adjacency matrix loss
                valid_batch_adj_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch)

                # compute and add adjacency and feature reconstruction loss
                valid_batch_loss = valid_batch_feat_rec_loss.cpu().detach().item() + valid_batch_adj_rec_loss.cpu().detach().item()

                # collect average validation losses
                average_valid_loss += valid_batch_loss
                average_adj_valid_loss += valid_batch_adj_rec_loss.cpu().detach().item()
                average_fea_valid_loss += valid_batch_feat_rec_loss.cpu().detach().item()

                ### compute detailed reconstruction losses

                # init feature vector reconstruction loss
                # valid_batch_feat_rec_loss_details = th.zeros(feat_matrices_batch_encodings.shape[0]).to(parameter['device'])

                # iterate over dataset features
                #for k, feature in enumerate(data_statistics['je_features']):

                    # determine original and reconstructed feature encodings
                    #feat_matrices_batch_encodings = feat_matrices_batch[:, :, k * parameter['feat_embed_dim']: (k+1) * parameter['feat_embed_dim']]
                    #feat_matrices_recon_encodings = feat_matrices_recon[:, :, k * parameter['feat_embed_dim']: (k+1) * parameter['feat_embed_dim']]

                    # compute feature vector loss
                    #valid_batch_feat_rec_loss_details += rec_criterion_details(input=feat_matrices_recon_encodings, target=feat_matrices_batch_encodings).mean(axis=1).mean(axis=1)

                # determine adjacency matrix normalization factor
                # norm = adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] / float((adj_matrices_batch.shape[1] * adj_matrices_batch.shape[1] - adj_matrices_batch.sum()) * 2)

                # compute feature vector loss
                valid_batch_feat_rec_loss_details = rec_criterion_details(input=feat_matrices_recon, target=feat_matrices_batch).mean(axis=1).mean(axis=1)

                # compute adjacency matrix loss
                valid_batch_adj_rec_loss_details = rec_criterion_details(input=adj_matrices_recon, target=adj_matrices_batch).mean(axis=1).mean(axis=1)

                # compute and add adjacency and feature reconstruction loss
                valid_batch_loss_details = valid_batch_feat_rec_loss_details + valid_batch_adj_rec_loss_details

                # log validation progress
                now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
                validation_iterations.set_description(
                    (
                        '[INFO {}] DeepAppleGraph :: iteration: {}, valid-loss: {}, valid-feat-loss: {}, valid-adj-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(valid_batch_loss / (i + 1), 6)), str(np.round(average_fea_valid_loss / (i + 1), 6)), str(np.round(average_adj_valid_loss / (i + 1), 6)))
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
                    valid_losses = np.hstack((valid_losses, valid_batch_loss_details.cpu().detach().numpy()))

                    # collect validation embeddings
                    valid_embeddings = np.vstack((valid_embeddings, valid_embeddings_batch.cpu().detach().numpy()))

                # update iterations
                validation_iterations.update(1)

        # close iterations
        validation_iterations.close()

        # update journal entries with embedding
        aggregated_entries['Y_REC_ERROR'] = valid_losses
        aggregated_entries['z1'] = valid_embeddings[:, 0]
        aggregated_entries['z2'] = valid_embeddings[:, 1]

        # determine final average batch validation losses
        experiment_statistics['average_valid_loss'] = average_valid_loss / (i + 1)
        experiment_statistics['average_adj_valid_loss'] = average_adj_valid_loss / (i + 1)
        experiment_statistics['average_fea_valid_loss'] = average_fea_valid_loss / (i + 1)

        # return model evaluation results
        return aggregated_entries, experiment_statistics

    # run the anomaly detection
    def run_anomaly_detection(self, parameter, aggregated_entries):

        # case: one-class svm anomaly detection
        if parameter['algo'] == 'svm':

            # init the one-class anomaly detection model
            svm_model = OneClassSVM(kernel=parameter['kernel'], degree=parameter['degree'], gamma=parameter['gamma'])

            # determine anomaly detection prediction
            predictions = svm_model.fit_predict(aggregated_entries[['z1', 'z2']])

            # determine anomaly detection score
            scores = svm_model.score_samples(aggregated_entries[['z1', 'z2']])

        # case: local outlier factor anomaly detection
        elif parameter['algo'] == 'lof':

            # init the local outlier factor model
            lof_model = LocalOutlierFactor(n_neighbors=parameter['n_neighbors'], leaf_size=parameter['leaf_size'])

            # determine anomaly detection prediction
            predictions = lof_model.fit_predict(aggregated_entries[['z1', 'z2']])

            # determine anomaly detection score
            scores = lof_model.negative_outlier_factor_

        # case: isolation forest anomaly detection
        elif parameter['algo'] == 'iforest':

            # init the local outlier factor model
            iforest_model = IsolationForest(random_state=0, n_estimators=100, max_samples=256)

            # determine anomaly detection prediction
            predictions = iforest_model.fit_predict(aggregated_entries[['z1', 'z2']])

            # determine anomaly detection score
            scores = iforest_model.score_samples(aggregated_entries[['z1', 'z2']])

        # case: isolation hdbscan anomaly detection
        elif parameter['algo'] == 'hdbscan':

            # grid search best hdbscan parameters
            best_parameters = self.grid_search_hdbscan_parameter(parameter=parameter, aggregated_entries=aggregated_entries)

            # update experiment parameter
            parameter['min_cluster_size'] = best_parameters['min_cluster_size']
            parameter['min_samples'] = best_parameters['min_samples']
            parameter['metric'] = best_parameters['metric']
            parameter['cluster_selection_method'] = best_parameters['cluster_selection_method']

            # init the hdbscan model with grid searched parameters
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=parameter['min_cluster_size'], min_samples=parameter['min_samples'], metric=parameter['metric'], cluster_selection_method=parameter['cluster_selection_method'])

            # determine hdbscan clustering prediction
            predictions = hdbscan_model.fit_predict(aggregated_entries[['z1', 'z2']])

            # determine anomaly detection score
            scores = hdbscan_model.outlier_scores_

            # replace potential nan values of the hdbscan anomaly score
            scores = np.nan_to_num(scores)

            # determine the anomaly score mean and standard deviation
            scores_mean = scores.mean()
            scores_std = scores.std()

            # normalize anomaly detection score
            scores = (scores - scores_mean) / scores_std

            # determine top quantile anomaly threshold
            anomaly_threshold = pd.Series(scores).quantile(0.99)

            # determine anomaly detection prediction
            predictions[np.where((scores >= anomaly_threshold) & (predictions != -1))[0]] = -2

        # collect local outlier factor anomaly prediction results
        aggregated_entries['Y_ANOMALY_CLASS'] = predictions

        # collect local outlier factor anomaly scores results
        aggregated_entries['Y_ANOMALY_SCORE'] = scores

        # collect local outlier factor anomaly prediction results
        aggregated_entries['Y_ANOMALY_LABEL'] = 'Regular'
        aggregated_entries['Y_ANOMALY_LABEL'] = np.where(aggregated_entries['Y_ANOMALY_CLASS'] == -1, 'Global Anomaly', aggregated_entries['Y_ANOMALY_LABEL'])
        aggregated_entries['Y_ANOMALY_LABEL'] = np.where(aggregated_entries['Y_ANOMALY_CLASS'] == -2, 'Local Anomaly', aggregated_entries['Y_ANOMALY_LABEL'])

        # determine global anomalies
        global_anomalies = aggregated_entries[aggregated_entries['Y_ANOMALY_CLASS'] == -1]

        # determine local anomalies
        local_anomalies = aggregated_entries[aggregated_entries['Y_ANOMALY_CLASS'] == -2]

        # return anomaly detection results
        return aggregated_entries, global_anomalies, local_anomalies

    def grid_search_hdbscan_parameter(self, parameter, aggregated_entries):

        # init the hdbscan model
        hdb = hdbscan.HDBSCAN(gen_min_span_tree=True).fit(aggregated_entries[['z1', 'z2']])

        # specify the grid hdbscan grid search paramters
        param_dist = {'min_samples': parameter['grid_min_samples'],
                      'min_cluster_size': parameter['grid_min_cluster_size'],
                      'cluster_selection_method': parameter['grid_cluster_selection_method'],
                      'metric': parameter['grid_metric']
                      }

        # init grid search optimization criterion
        validity_scorer = make_scorer(hdbscan.validity.validity_index, greater_is_better=True)

        # init hdbscan parameter grid search
        random_search = RandomizedSearchCV(hdb, param_distributions=param_dist, n_iter=20, scoring=validity_scorer, random_state=parameter['seed'], verbose=0)

        # run hdbscan parameter grid search
        random_search.fit(aggregated_entries[['z1', 'z2']])

        # get hdbscan parameters
        best_parameters = random_search.best_params_

        # return best parameters
        return best_parameters

    # run the model and result visualization
    def run_model_visualization(self, parameter, data_statistics, experiment_statistics, data, average_train_loss, average_valid_loss, iteration):

        # visualize learned embeddings
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)))
        self.vha.plot_embeddings_2d(data=data, z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

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
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_anomalies_score_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}<br>Anomaly-Algorithm: {}, Clusters: {}, Global-Anomalies: {}, Local-Anomalies: {}'.format(str(parameter['dataset']).upper(), str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)), str(parameter['algo']).upper(), str(experiment_statistics['no_clusters']), str(experiment_statistics['no_global_anomalies']), str(experiment_statistics['no_local_anomalies']))
        self.vha.plot_embeddings_2d_anomalies_score_interactive(data=data, hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_ANOMALY_SCORE', filename=filename, title=title)

        # visualize learned embeddings interactively
        #filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_anomalies_cluster_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(iteration).zfill(6), str(parameter['exp_postfix']))
        #title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}<br>Anomaly-Algorithm: {}, Global-Anomalies: {}, Local-Anomalies: {}'.format(str(parameter['dataset']).upper(), str(iteration).zfill(6), str(np.round((average_train_loss / iteration), 6)), str(np.round((average_valid_loss / iteration), 6)), str(parameter['algo']).upper(), str(data[data['Y_ANOMALY_CLASS'] == -1].shape[0]), str(data[data['Y_ANOMALY_CLASS'] == -2].shape[0]))
        #self.vha.plot_embeddings_2d_anomalies_cluster_interactive(data=data, hover=data_statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_ANOMALY_CLASS', filename=filename, title=title)