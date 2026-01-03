# import python libraries
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import sklearn libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd



# class EvaluationHandler
class EvaluationHandler(object):

    def __init__(self):

        pass

    # precision recall area under the curve score
    def compute_error_pr_auc_score(self, statistics, results, entries):

        # init error pr auc scores
        results['error_pr_auc_all'] = 0.0
        results['error_pr_auc_global'] = 0.0
        results['error_pr_auc_local'] = 0.0

        # init min-max normalization
        scaler = MinMaxScaler()

        # min-max normalize reconstruction error
        rec_error_scaled_all = scaler.fit_transform(entries['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_all = entries[statistics['je_class_field']].copy(deep=True).to_numpy()

        # determine binary class labels
        class_labels_all[class_labels_all == 2] = 1

        # compute overall average precision score
        results['error_pr_auc_all'] = average_precision_score(y_true=class_labels_all, y_score=rec_error_scaled_all)

        # determine regular journal entries and global anomalies
        entries_regular_global = entries[entries[statistics['je_class_field']] != 2]

        # min-max normalize reconstruction error
        rec_error_scaled_global = scaler.fit_transform(
            entries_regular_global['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_global = entries_regular_global[statistics['je_class_field']].copy(deep=True).to_numpy()

        # compute overall average precision score
        results['error_pr_auc_global'] = average_precision_score(y_true=class_labels_global,
                                                                 y_score=rec_error_scaled_global)

        # determine regular journal entries and global anomalies
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]

        # min-max normalize reconstruction error
        rec_error_scaled_local = scaler.fit_transform(
            entries_regular_local['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_local = entries_regular_local[statistics['je_class_field']].copy(deep=True).to_numpy()

        # reset class label
        class_labels_local[class_labels_local == 2] = 1

        # compute overall average precision score
        results['error_pr_auc_local'] = average_precision_score(y_true=class_labels_local,
                                                                y_score=rec_error_scaled_local)

        # return overall, local and global pr-auc scores
        return results

    # roc area under the curve score
    def compute_error_roc_auc_score(self, statistics, results, entries):

        # init error pr auc scores
        results['error_roc_auc_all'] = 0.0
        results['error_roc_auc_global'] = 0.0
        results['error_roc_auc_local'] = 0.0

        # init min-max normalization
        scaler = MinMaxScaler()

        # min-max normalize reconstruction error
        rec_error_scaled_all = scaler.fit_transform(entries['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_all = entries[statistics['je_class_field']].copy(deep=True).to_numpy()

        # determine binary class labels
        class_labels_all[class_labels_all == 2] = 1

        # compute overall average precision score
        results['error_roc_auc_all'] = roc_auc_score(y_true=class_labels_all, y_score=rec_error_scaled_all)

        # determine regular journal entries and global anomalies
        entries_regular_global = entries[entries[statistics['je_class_field']] != 2]

        # min-max normalize reconstruction error
        rec_error_scaled_global = scaler.fit_transform(
            entries_regular_global['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_global = entries_regular_global[statistics['je_class_field']].copy(deep=True).to_numpy()

        # compute overall average precision score
        results['error_roc_auc_global'] = roc_auc_score(y_true=class_labels_global, y_score=rec_error_scaled_global)

        # determine regular journal entries and global anomalies
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]

        # min-max normalize reconstruction error
        rec_error_scaled_local = scaler.fit_transform(
            entries_regular_local['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_local = entries_regular_local[statistics['je_class_field']].copy(deep=True).to_numpy()

        # reset class label
        class_labels_local[class_labels_local == 2] = 1

        # compute overall average precision score
        results['error_roc_auc_local'] = roc_auc_score(y_true=class_labels_local, y_score=rec_error_scaled_local)

        # return overall, local and global roc-auc scores
        return results

    # precision recall area under the curve score
    def compute_density_pr_auc_score(self, statistics, results, entries):

        # init density pr auc scores
        results['density_pr_auc_all'] = 0.0
        results['density_pr_auc_global'] = 0.0
        results['density_pr_auc_local'] = 0.0

        # init min-max normalization
        scaler = MinMaxScaler()

        # min-max normalize reconstruction error
        rec_error_scaled_all = scaler.fit_transform(entries['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_all = entries[statistics['je_class_field']].copy(deep=True).to_numpy()

        # determine binary class labels
        class_labels_all[class_labels_all == 2] = 1

        # compute overall average precision score
        results['density_pr_auc_all'] = average_precision_score(y_true=class_labels_all, y_score=rec_error_scaled_all)

        # determine regular journal entries and global anomalies
        entries_regular_global = entries[entries[statistics['je_class_field']] != 2]

        # min-max normalize reconstruction error
        rec_error_scaled_global = scaler.fit_transform(
            entries_regular_global['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_global = entries_regular_global[statistics['je_class_field']].copy(deep=True).to_numpy()

        # compute overall average precision score
        results['density_pr_auc_global'] = average_precision_score(y_true=class_labels_global,
                                                                   y_score=rec_error_scaled_global)

        # determine regular journal entries and global anomalies
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]

        # min-max normalize reconstruction error
        rec_error_scaled_local = scaler.fit_transform(
            entries_regular_local['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_local = entries_regular_local[statistics['je_class_field']].copy(deep=True).to_numpy()

        # reset class label
        class_labels_local[class_labels_local == 2] = 1

        # compute overall average precision score
        results['density_pr_auc_local'] = average_precision_score(y_true=class_labels_local,
                                                                  y_score=rec_error_scaled_local)

        # return overall, local and global pr-auc scores
        return results

    # roc area under the curve score
    def compute_density_roc_auc_score(self, statistics, results, entries):

        # init density pr auc scores
        results['density_roc_auc_all'] = 0.0
        results['density_roc_auc_global'] = 0.0
        results['density_roc_auc_local'] = 0.0

        # init min-max normalization
        scaler = MinMaxScaler()

        # min-max normalize reconstruction error
        rec_error_scaled_all = scaler.fit_transform(entries['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_all = entries[statistics['je_class_field']].copy(deep=True).to_numpy()

        # determine binary class labels
        class_labels_all[class_labels_all == 2] = 1

        # compute overall average precision score
        results['density_roc_auc_all'] = roc_auc_score(y_true=class_labels_all, y_score=rec_error_scaled_all)

        # determine regular journal entries and global anomalies
        entries_regular_global = entries[entries[statistics['je_class_field']] != 2]

        # min-max normalize reconstruction error
        rec_error_scaled_global = scaler.fit_transform(
            entries_regular_global['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_global = entries_regular_global[statistics['je_class_field']].copy(deep=True).to_numpy()

        # compute overall average precision score
        results['density_roc_auc_global'] = roc_auc_score(y_true=class_labels_global, y_score=rec_error_scaled_global)

        # determine regular journal entries and global anomalies
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]

        # min-max normalize reconstruction error
        rec_error_scaled_local = scaler.fit_transform(
            entries_regular_local['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_local = entries_regular_local[statistics['je_class_field']].copy(deep=True).to_numpy()

        # reset class label
        class_labels_local[class_labels_local == 2] = 1

        # compute overall average precision score
        results['density_roc_auc_local'] = roc_auc_score(y_true=class_labels_local, y_score=rec_error_scaled_local)

        # return overall, local and global pr-auc scores
        return results

    def compute_classification_metrics(self, statistics: object, results: object, entries: object) -> object:
        """
        Calcula métricas de classificação para detecção de anomalias
        seguindo o mesmo padrão dos métodos existentes
        """

        # Inicialização padrão igual aos outros métodos
        results['od_accuracy_all'] = 0.0
        results['od_precision_all'] = 0.0
        results['od_recall_all'] = 0.0
        results['od_f1_score_all'] = 0.0

        results['od_accuracy_global'] = 0.0
        results['od_precision_global'] = 0.0
        results['od_recall_global'] = 0.0
        results['od_f1_score_global'] = 0.0

        results['od_accuracy_local'] = 0.0
        results['od_precision_local'] = 0.0
        results['od_recall_local'] = 0.0
        results['od_f1_score_local'] = 0.0

        # metrics for reconstruction error
        results['re_accuracy_all'] = 0.0
        results['re_precision_all'] = 0.0
        results['re_recall_all'] = 0.0
        results['re_f1_score_all'] = 0.0

        results['re_accuracy_global'] = 0.0
        results['re_precision_global'] = 0.0
        results['re_recall_global'] = 0.0
        results['re_f1_score_global'] = 0.0

        results['re_accuracy_local'] = 0.0
        results['re_precision_local'] = 0.0
        results['re_recall_local'] = 0.0
        results['re_f1_score_local'] = 0.0

        # Mesmo pré-processamento de normalização
        scaler = MinMaxScaler()
        anomaly_scores = scaler.fit_transform(entries['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()
        y_rank = pd.Series(anomaly_scores).rank(ascending=False, method='average')
        y_pred = (y_rank <= 200).astype(int)

        rec_error= scaler.fit_transform(entries['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()
        y_rank_re = pd.Series(rec_error).rank(ascending=False, method='average')
        y_pred_re = (y_rank_re <= 200).astype(int)


        # Mesmo tratamento de labels que nos métodos existentes
        class_labels = entries[statistics['je_class_field']].copy(deep=True).to_numpy()
        class_labels[class_labels == 2] = 1  # Unificação de classes anômalas

        #compute the metrics for the OD global
        entries_regular_global = entries[entries[statistics['je_class_field']] != 2]
        y_pred_global = y_pred[entries_regular_global.index]
        y_labels_global = class_labels[entries_regular_global.index]

        # compute the metrics for the OD local
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]
        y_pred_local = y_pred[entries_regular_local.index]
        y_labels_local = class_labels[entries_regular_local.index]


        # compute the metrics for the RE global
        entries_regular_global= entries[entries[statistics['je_class_field']] != 2]
        y_pred_global_re = y_pred_re[entries_regular_global.index]

        # compute the metrics for the RE local
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]
        y_pred_local_re = y_pred_re[entries_regular_local.index]



        # Cálculo das métricas com mesma estrutura de erro
        try:
            # Threshold adaptativo igual ao usado em AUC

            #metrics for outlier detection
            results['od_accuracy_all'] = accuracy_score(y_true=class_labels, y_pred=y_pred)
            results['od_precision_all'] = precision_score(y_true=class_labels,y_pred=y_pred)
            results['od_recall_all'] = recall_score(y_true=class_labels,y_pred=y_pred)
            results['od_f1_score_all'] = f1_score(y_true=class_labels, y_pred=y_pred)

            results['od_accuracy_global'] = accuracy_score(y_true=y_labels_global, y_pred=y_pred_global)
            results['od_precision_global'] = precision_score(y_true=y_labels_global, y_pred=y_pred_global)
            results['od_recall_global'] = recall_score(y_true=y_labels_global, y_pred=y_pred_global)
            results['od_f1_score_global'] = f1_score(y_true=y_labels_global, y_pred=y_pred_global)

            results['od_accuracy_local'] = accuracy_score(y_true=y_labels_local, y_pred=y_pred_local)
            results['od_precision_local'] = precision_score(y_true=y_labels_local, y_pred=y_pred_local)
            results['od_recall_local'] = recall_score(y_true=y_labels_local, y_pred=y_pred_local)
            results['od_f1_score_local'] = f1_score(y_true=y_labels_local, y_pred=y_pred_local)

            #metrics for reconstruction error
            results['re_accuracy_all'] = accuracy_score(y_true=class_labels, y_pred=y_pred_re)
            results['re_precision_all'] = precision_score(y_true=class_labels, y_pred=y_pred_re)
            results['re_recall_all'] = recall_score(y_true=class_labels, y_pred=y_pred_re)
            results['re_f1_score_all'] = f1_score(y_true=class_labels, y_pred=y_pred_re)

            results['re_accuracy_global'] = accuracy_score(y_true=y_labels_global, y_pred=y_pred_global_re)
            results['re_precision_global'] = precision_score(y_true=y_labels_global, y_pred=y_pred_global_re)
            results['re_recall_global'] = recall_score(y_true=y_labels_global, y_pred=y_pred_global_re)
            results['re_f1_score_global'] = f1_score(y_true=y_labels_global, y_pred=y_pred_global_re)

            results['re_accuracy_local'] = accuracy_score(y_true=y_labels_local, y_pred=y_pred_local_re)
            results['re_precision_local'] = precision_score(y_true=y_labels_local, y_pred=y_pred_local_re)
            results['re_recall_local'] = recall_score(y_true=y_labels_local, y_pred=y_pred_local_re)
            results['re_f1_score_local'] = f1_score(y_true=y_labels_local, y_pred=y_pred_local_re)

        except Exception as e:

            print(f"Error calculating classification metrics: {str(e)}")

        return results
