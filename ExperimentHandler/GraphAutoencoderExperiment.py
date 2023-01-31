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

# import wandb logging
import wandb

# import pytorch libraries
import torch as th
from torch.utils.data import DataLoader

# import scikit learn libraries
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# import hdbscan library
import hdbscan

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
            run = wandb.init(project='DeepAppleGraph', group='{}'.format(str(parameter['base_dir'].split('/')[-1])), name='{}_ae_graph_sd_{}_ds_{}_{}'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['dataset']), str(parameter['exp_postfix'])))

            # add wandb experiment configuration
            run.config.update(parameter)

            # init wandb log dictionary
            wandb_logging = {}

        # create experiment directory
        parameter['exp_dir'], parameter['par_sub_dir'], parameter['sta_sub_dir'], parameter['res_sub_dir'], parameter['vis_sub_dir'], parameter['log_sub_dir'] = self.uha.create_experiment_directory(param=parameter, parent_dir=parameter['base_dir'], architecture='ae_graph')

        # save experiment parameters
        self.uha.save_experiment_parameter(param=parameter, parameter_dir=parameter['par_sub_dir'])

        # init dataset statistics
        dataset_statistics = {}
        experiment_statistics = {}

        # case: e&y dataset
        if parameter['dataset'] == 'ey':

            # load the EY training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries, selected_aggregated_entries, statistics = self.dha.get_gnn_data_range_ey(parameter=parameter, statistics=dataset_statistics)

        # case: serpro dataset
        elif parameter['dataset'] == 'serpro':

            # load the Serpro training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries, dataset_statistics = self.dha.get_gnn_data_range_serpro(parameter=parameter, statistics=dataset_statistics)

        # case: sap dataset
        elif parameter['dataset'] == 'sap':

            # load the Serpro training data
            posting_ids, adj_matrices, feat_matrices, aggregated_entries, selected_aggregated_entries, dataset_statistics = self.dha.get_gnn_data_range_sap(parameter=parameter, statistics=dataset_statistics)

        # log aggregated entries
        file_name = '{}_aggregated_entries_all_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # log selected aggregated entries
        file_name = '{}_aggregated_entries_selected_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        selected_aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # init experiment statistics
        summary_cols = [
            'timestamp'
            , 'seed'
            , 'iterations'
            , 'batch_size'
            , 'learning_rate'
            , 'encoder_dim'
            , 'decoder_dim'
            , 'embed_dim'
            , 'bottleneck'
            , 'train_loss'
            , 'train_adj_loss'
            , 'train_fea_loss'
            , 'valid_loss'
            , 'valid_adj_loss'
            , 'valid_fea_loss'
            , 'algo'
            , 'min_cluster_size'
            , 'min_samples'
            , 'no_global_anomalies'
            , 'no_local_anomalies'
        ]
        experiment_results = pd.DataFrame(columns=summary_cols)

        # determine the number of accounts and features
        no_accounts = adj_matrices[0].shape[1]
        no_features = feat_matrices[0].shape[1]

        # update the encoder input dim depending on the number of features
        parameter['encoder_dim'].insert(0, no_features)
        parameter['decoder_dim'].insert(len(parameter['decoder_dim']), no_accounts * no_features)
        parameter['decoder_dim'].insert(len(parameter['decoder_dim']), no_accounts * no_accounts)

        # convert the EY training data to pytorch tensor
        prepared_tensor_entries = AccountingGNNDataset(adj_matrices=adj_matrices, feat_matrices=feat_matrices)

        #### start training routine

        # init the EY data loader
        train_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=True, drop_last=False)

        # init the graph convolutional autoencoder model
        model = GNNAutoencoder.GNNAutoencoder(
            # token_no=statistics['token_no'],
            # data_dim=parameter['data_dim'],
            encoder_dim=parameter['encoder_dim'],
            bottleneck=parameter['bottleneck'],
            decoder_dim=parameter['decoder_dim'],
            device=parameter['device']
        ).to(parameter['device'])

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DeepAppleGraph :: GNN Autoencoder Model {}.'.format(now, str(model)))

        # case: BCE loss training
        if parameter['loss'] == 'bce':

            # init aggregated BCE autoencoder loss
            rec_criterion = th.nn.BCELoss().to(parameter['device'])

        # case: MSE loss training
        elif parameter['loss'] == 'mse':

            # init aggregated MSE autoencoder loss
            rec_criterion = th.nn.MSELoss().to(parameter['device'])

        # init training optimizer
        optimizer = th.optim.Adam(model.parameters(), lr=parameter['learning_rate'])

        # run the model training
        model, experiment_statistics = self.run_model_training(parameter=parameter, experiment_statistics=experiment_statistics, model=model, rec_criterion=rec_criterion, train_loader=train_loader, optimizer=optimizer, wandb_logging=wandb_logging, run=run)

        #### start evaluation routine

        # init the EY data loader
        eval_loader = DataLoader(prepared_tensor_entries, batch_size=parameter['batch_size'], shuffle=False, drop_last=False)

        # case: BCE loss training
        if parameter['loss'] == 'bce':

            # init aggregated BCE autoencoder loss
            rec_criterion = th.nn.BCELoss().to(parameter['device'])

            # init detailed BCE autoencoder loss
            rec_criterion_details = th.nn.BSELoss(reduce=False).to(parameter['device'])

        # case: MSE loss training
        elif parameter['loss'] == 'mse':

            # init aggregated MSE autoencoder loss
            rec_criterion = th.nn.MSELoss().to(parameter['device'])

            # init detailed MSE autoencoder loss
            rec_criterion_details = th.nn.MSELoss(reduce=False).to(parameter['device'])

        # run the model evaluation
        selected_aggregated_entries, experiment_statistics = self.run_model_validation(parameter=parameter, experiment_statistics=experiment_statistics, model=model, rec_criterion=rec_criterion, rec_criterion_details=rec_criterion_details, eval_loader=eval_loader, aggregated_entries=selected_aggregated_entries, wandb_logging=wandb_logging, run=run)

        #### start anomaly detection routine
        selected_aggregated_entries, global_anomalies, local_anomalies = self.run_anomaly_detection(parameter, selected_aggregated_entries)

        # log aggregated entries
        file_name = '{}_aggregated_entries_selected_sd_{}_it_{}_{}.csv'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        selected_aggregated_entries.to_csv(os.path.join(parameter['res_sub_dir'], file_name), sep=',', encoding='utf-8')

        # set visualization handler directory
        self.vha.set_plot_dir(plot_dir=parameter['vis_sub_dir'])

        # run the model visualization
        self.run_model_visualization(parameter=parameter, statistics=statistics, data=selected_aggregated_entries, average_train_loss=experiment_statistics['average_train_loss'], average_valid_loss=experiment_statistics['average_valid_loss'])

        # collect experiment statistics
        exp_stats = {
            'timestamp': str(dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S'))
            , 'seed': parameter['seed']
            , 'iterations': parameter['iterations']
            , 'batch_size': parameter['batch_size']
            , 'learning_rate': parameter['learning_rate']
            , 'encoder_dim': parameter['encoder_dim']
            , 'decoder_dim': parameter['decoder_dim']
            , 'embed_dim': parameter['embed_dim']
            , 'bottleneck': parameter['bottleneck']
            , 'train_loss': np.round(experiment_statistics['average_train_loss'], 6)
            , 'train_adj_loss': np.round(experiment_statistics['average_adj_train_loss'], 6)
            , 'train_fea_loss': np.round(experiment_statistics['average_fea_train_loss'], 6)
            , 'valid_loss': np.round(experiment_statistics['average_valid_loss'], 6)
            , 'valid_adj_loss': np.round(experiment_statistics['average_adj_valid_loss'], 6)
            , 'valid_fea_loss': np.round(experiment_statistics['average_fea_valid_loss'], 6)
            , 'algo': parameter['algo']
            , 'min_cluster_size': parameter['min_cluster_size']
            , 'min_samples':  parameter['min_samples']
            , 'no_global_anomalies': int(global_anomalies.shape[0])
            , 'no_local_anomalies': int(local_anomalies.shape[0])
        }

        # determine and collect training summary statistics of current epoch
        experiment_results = experiment_results.append(exp_stats, ignore_index=True)

        # save current experiment statistics
        file_name = '{}_experiment_results_sd_{}_ep_{}.csv'.format(parameter['exp_timestamp'], parameter['seed'], parameter['iterations'])
        experiment_results.to_csv(os.path.join(parameter['sta_sub_dir'], file_name), sep=',', encoding='utf-8')

        # case: wandb logging enabled
        if parameter['wandb']:

            # finish wandb run
            run.finish()

    # run the model training
    def run_model_training(self, parameter, experiment_statistics, model, rec_criterion, train_loader, optimizer, run, wandb_logging):

        # set model in train mode
        model.train()

        # init the training loss
        experiment_statistics['average_train_loss'] = 0.0
        experiment_statistics['average_adj_train_loss'] = 0.0
        experiment_statistics['average_fea_train_loss'] = 0.0

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

            # compute feature vector loss
            train_batch_feat_rec_loss = rec_criterion(input=feat_matrices_recon, target=feat_matrices_batch)

            # compute adjacency matrix loss
            train_batch_adj_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch)

            # compute kl-divergence loss
            # train_batch_kl_div_loss = parameter['kl_div_alpha'] * (-0.5 * th.sum(1 + sigma - mu.pow(2) - sigma.exp()))

            # compute train batch loss
            train_batch_loss = train_batch_feat_rec_loss + train_batch_adj_rec_loss

            # compute and collect average reconstruction loss
            experiment_statistics['average_train_loss'] += train_batch_loss.cpu().detach().item()
            experiment_statistics['average_adj_train_loss'] += train_batch_adj_rec_loss.cpu().detach().item()
            experiment_statistics['average_fea_train_loss'] += train_batch_feat_rec_loss.cpu().detach().item()

            # log training progress
            now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
            training_iterations.set_description(
                (
                    '[INFO {}] DeepAppleGraph :: iteration: {}, train-loss: {}, train-feat-loss: {}, train-adj-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(train_batch_loss.cpu().detach().item(), 8)), str(np.round(train_batch_feat_rec_loss.cpu().detach().item(), 8)), str(np.round(train_batch_adj_rec_loss.cpu().detach().item(), 8)))
                )
            )

            # run backward pass
            train_batch_loss.backward()

            # run optimizer step
            optimizer.step()

            # case: wandb logging enabled
            if parameter['wandb']:

                # fill wandb log dict
                wandb_logging['001_model_training/avg_train_loss'] = experiment_statistics['average_train_loss'] / (i + 1)
                wandb_logging['001_model_training/avg_adj_train_loss'] = experiment_statistics['average_adj_train_loss'] / (i + 1)
                wandb_logging['001_model_training/avg_fea_train_loss'] = experiment_statistics['average_fea_train_loss'] / (i + 1)

                # log training progress
                run.log(wandb_logging)

            # case: eval iteration
            if i % parameter['eval_iteration'] == 0:

                # save client model checkpoint
                file_name = '{}_ae_gnn_model_checkpoint_itr_{}.pth'.format(parameter['exp_timestamp'], str(i).zfill(6))
                self.uha.save_client_model_checkpoint(filename=file_name, iteration=i, model=model, optimizer=optimizer, chpt_dir=parameter['log_sub_dir'])

        # determine finale average losses per iteration
        experiment_statistics['average_train_loss'] /= parameter['iterations']
        experiment_statistics['average_adj_train_loss'] /= parameter['iterations']
        experiment_statistics['average_fea_train_loss'] /= parameter['iterations']

        # return model training results
        return model, experiment_statistics

    # run the model evaluation
    def run_model_validation(self, parameter, experiment_statistics, model, rec_criterion, rec_criterion_details, eval_loader, aggregated_entries, run, wandb_logging):

        # set model in evaluation mode
        model.eval()

        # init validation reconstruction losses
        experiment_statistics['average_valid_loss'] = 0.0
        experiment_statistics['average_adj_valid_loss'] = 0.0
        experiment_statistics['average_fea_valid_loss'] = 0.0

        # init detailed validation reconstruction losses
        valid_losses = []

        # init validation embeddings
        valid_embeddings = []

        # init and wrap range of training iterations
        validation_iterations = tqdm(total=len(eval_loader))

        # case: mini-batches are still available
        for i, (adj_matrices_batch, feat_matrices_batch) in enumerate(eval_loader):

            # push the inputs to compute device
            adj_matrices_batch = adj_matrices_batch.to(parameter['device'])
            feat_matrices_batch = feat_matrices_batch.to(parameter['device'])

            # clamp reconstructed matrix
            # adj_matrices_batch = th.clamp(adj_matrices_batch, min=0.0, max=1.0)

            # run model forward pass
            valid_embeddings_batch, _, _, feat_matrices_recon, adj_matrices_recon = model(feat_matrices_batch, adj_matrices_batch)

            ### compute batch reconstruction loss

            # compute feature vector loss
            valid_batch_feat_rec_loss = rec_criterion(input=feat_matrices_recon, target=feat_matrices_batch)

            # compute adjacency matrix loss
            valid_batch_adj_rec_loss = rec_criterion(input=adj_matrices_recon, target=adj_matrices_batch)

            # compute and add adjacency and feature reconstruction loss
            valid_batch_loss = valid_batch_feat_rec_loss.cpu().detach().item() + valid_batch_adj_rec_loss.cpu().detach().item()

            experiment_statistics['average_valid_loss'] += valid_batch_loss
            experiment_statistics['average_adj_valid_loss'] += valid_batch_adj_rec_loss.cpu().detach().item()
            experiment_statistics['average_fea_valid_loss'] += valid_batch_feat_rec_loss.cpu().detach().item()

            ### compute detailed reconstruction losses

            # compute and add categorical reconstruction loss
            valid_batch_feat_rec_loss_details = rec_criterion_details(input=feat_matrices_recon, target=feat_matrices_batch).mean(axis=1).mean(axis=1)

            # compute and add categorical reconstruction loss
            valid_batch_adj_rec_loss_details = rec_criterion_details(input=adj_matrices_recon, target=adj_matrices_batch).mean(axis=1).mean(axis=1)

            # compute and add adjacency and feature reconstruction loss
            valid_batch_loss_details = valid_batch_feat_rec_loss_details + valid_batch_adj_rec_loss_details

            # log validation progress
            now = dt.datetime.utcnow().strftime('%m/%d/%Y %H:%M:%S')
            validation_iterations.set_description(
                (
                    '[INFO {}] DeepAppleGraph :: iteration: {}, valid-loss: {}, valid-feat-loss: {}, valid-adj-loss: {}'.format(str(now), str(i).zfill(2), str(np.round(valid_batch_loss, 6)), str(np.round(valid_batch_feat_rec_loss.cpu().detach().item(), 6)), str(np.round(valid_batch_adj_rec_loss.cpu().detach().item(), 6)))
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

            # case: wandb logging enabled
            if parameter['wandb']:

                # fill wandb log dict
                wandb_logging['001_model_training/avg_valid_loss'] = experiment_statistics['average_valid_loss'] / (i + 1)
                wandb_logging['001_model_training/avg_adj_valid_loss'] = experiment_statistics['average_adj_valid_loss'] / (i + 1)
                wandb_logging['001_model_training/avg_fea_valid_loss'] = experiment_statistics['average_fea_valid_loss'] / (i + 1)

                # log training progress
                run.log(wandb_logging)

            # update validation iterations
            validation_iterations.update(1)

        # update journal entries with embedding
        aggregated_entries['Y_REC_ERROR'] = valid_losses
        aggregated_entries['z1'] = valid_embeddings[:, 0]
        aggregated_entries['z2'] = valid_embeddings[:, 1]

        # close validation iteration
        validation_iterations.close()

        # determine final average losses per iteration
        experiment_statistics['average_valid_loss'] /= parameter['iterations']
        experiment_statistics['average_adj_valid_loss'] /= parameter['iterations']
        experiment_statistics['average_fea_valid_loss'] /= parameter['iterations']

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

            # init the hdbscan model
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=parameter['min_cluster_size'], min_samples=parameter['min_samples'])

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

        # determine the anomaly score mean and standard deviation
        # scores_mean = scores.mean()
        # scores_std = scores.std()

        # collect local outlier factor anomaly prediction results
        aggregated_entries['Y_ANOMALY_CLASS'] = predictions

        # collect local outlier factor anomaly scores results
        aggregated_entries['Y_ANOMALY_SCORE'] = scores

        # determine global anomalies
        global_anomalies = aggregated_entries[aggregated_entries['Y_ANOMALY_CLASS'] == -1]

        # determine local anomalies
        local_anomalies = aggregated_entries[aggregated_entries['Y_ANOMALY_CLASS'] == -2]

        # return anomaly detection results
        return aggregated_entries, global_anomalies, local_anomalies

    # run the model and result visualization
    def run_model_visualization(self, parameter, statistics, data, average_train_loss, average_valid_loss):

        # visualize learned embeddings
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)), str(np.round((average_valid_loss / parameter['iterations']), 6)))
        self.vha.plot_embeddings_2d(data=data, z1_col_name='z1', z2_col_name='z2', filename=filename, title=title)

        # visualize learned embeddings in specific interval
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_interval.png'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        title = 'GNN Autoencoder - Journal Entry Embedding Distribution\niterations: {}, avg-train-loss: {}, avg-valid-loss: {}'.format(str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)), str(np.round((average_valid_loss / parameter['iterations']), 6)))
        self.vha.plot_embeddings_2d_interval(data=data, z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title, xlim=[-5.0, 5.0], ylim=[-5.0, 5.0]) # xlim=[18.5, 20.1], ylim=[-10.2, -11.5]

        # visualize learned embeddings interactively
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}'.format(str(parameter['dataset']).upper(), str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)), str(np.round((average_valid_loss / parameter['iterations']), 6)))
        self.vha.plot_embeddings_2d_interactive(data=data, hover=statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_REC_ERROR', filename=filename, title=title)

        # visualize learned embeddings interactively
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_anomalies_score_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}<br>Anomaly-Algorithm: {}, Global-Anomalies: {}, Local-Anomalies: {}'.format(str(parameter['dataset']).upper(), str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)), str(np.round((average_valid_loss / parameter['iterations']), 6)), str(parameter['algo']).upper(), str(data[data['Y_ANOMALY_CLASS'] == -1].shape[0]), str(data[data['Y_ANOMALY_CLASS'] == -2].shape[0]))
        self.vha.plot_embeddings_2d_anomalies_score_interactive(data=data, hover=statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_ANOMALY_SCORE', filename=filename, title=title)

        # visualize learned embeddings interactively
        filename = '{}_je_embedding_distribution_sd_{}_it_{}_{}_anomalies_cluster_interactive.html'.format(str(parameter['exp_timestamp']), str(parameter['seed']), str(parameter['iterations']).zfill(6), str(parameter['exp_postfix']))
        title = '<b>GNN Autoencoder - Journal Entry Embedding Distribution</b><br>Dataset: {}, Train-Iterations: {}, Avg-Train-Loss: {}, Avg-Valid-Loss: {}<br>Anomaly-Algorithm: {}, Global-Anomalies: {}, Local-Anomalies: {}'.format(str(parameter['dataset']).upper(), str(parameter['iterations']).zfill(6), str(np.round((average_train_loss / parameter['iterations']), 6)), str(np.round((average_valid_loss / parameter['iterations']), 6)), str(parameter['algo']).upper(), str(data[data['Y_ANOMALY_CLASS'] == -1].shape[0]), str(data[data['Y_ANOMALY_CLASS'] == -2].shape[0]))
        self.vha.plot_embeddings_2d_anomalies_cluster_interactive(data=data, hover=statistics['hover_attributes'], z1_col_name='z1', z2_col_name='z2', c_col_name='Y_ANOMALY_CLASS', filename=filename, title=title)