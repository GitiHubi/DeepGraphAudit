# import python libraries
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import sklearn libraries
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import average_precision_score, roc_auc_score

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
        rec_error_scaled_global = scaler.fit_transform(entries_regular_global['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_global = entries_regular_global[statistics['je_class_field']].copy(deep=True).to_numpy()

        # compute overall average precision score
        results['error_pr_auc_global'] = average_precision_score(y_true=class_labels_global, y_score=rec_error_scaled_global)

        # determine regular journal entries and global anomalies
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]

        # min-max normalize reconstruction error
        rec_error_scaled_local = scaler.fit_transform(entries_regular_local['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_local = entries_regular_local[statistics['je_class_field']].copy(deep=True).to_numpy()

        # reset class label
        class_labels_local[class_labels_local == 2] = 1

        # compute overall average precision score
        results['error_pr_auc_local'] = average_precision_score(y_true=class_labels_local, y_score=rec_error_scaled_local)

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
        rec_error_scaled_global = scaler.fit_transform(entries_regular_global['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_global = entries_regular_global[statistics['je_class_field']].copy(deep=True).to_numpy()

        # compute overall average precision score
        results['error_roc_auc_global'] = roc_auc_score(y_true=class_labels_global, y_score=rec_error_scaled_global)

        # determine regular journal entries and global anomalies
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]

        # min-max normalize reconstruction error
        rec_error_scaled_local = scaler.fit_transform(entries_regular_local['Y_REC_ERROR'].to_numpy().reshape(-1, 1)).flatten()

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
        rec_error_scaled_global = scaler.fit_transform(entries_regular_global['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_global = entries_regular_global[statistics['je_class_field']].copy(deep=True).to_numpy()

        # compute overall average precision score
        results['density_pr_auc_global'] = average_precision_score(y_true=class_labels_global, y_score=rec_error_scaled_global)

        # determine regular journal entries and global anomalies
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]

        # min-max normalize reconstruction error
        rec_error_scaled_local = scaler.fit_transform(entries_regular_local['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_local = entries_regular_local[statistics['je_class_field']].copy(deep=True).to_numpy()

        # reset class label
        class_labels_local[class_labels_local == 2] = 1

        # compute overall average precision score
        results['density_pr_auc_local'] = average_precision_score(y_true=class_labels_local, y_score=rec_error_scaled_local)

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
        rec_error_scaled_global = scaler.fit_transform(entries_regular_global['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_global = entries_regular_global[statistics['je_class_field']].copy(deep=True).to_numpy()

        # compute overall average precision score
        results['density_roc_auc_global'] = roc_auc_score(y_true=class_labels_global, y_score=rec_error_scaled_global)

        # determine regular journal entries and global anomalies
        entries_regular_local = entries[entries[statistics['je_class_field']] != 1]

        # min-max normalize reconstruction error
        rec_error_scaled_local = scaler.fit_transform(entries_regular_local['Y_ANOMALY_SCORE'].to_numpy().reshape(-1, 1)).flatten()

        # convert class labels to numpy
        class_labels_local = entries_regular_local[statistics['je_class_field']].copy(deep=True).to_numpy()

        # reset class label
        class_labels_local[class_labels_local == 2] = 1

        # compute overall average precision score
        results['density_roc_auc_local'] = roc_auc_score(y_true=class_labels_local, y_score=rec_error_scaled_local)

        # return overall, local and global pr-auc scores
        return results