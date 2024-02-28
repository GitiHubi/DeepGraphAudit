# import python libraries
import os

# limit the number of threads
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import class libraries
import numpy as np
import pandas as pd
import random as rd

# import scikit learn algorithms
#from sklearn.svm import OneClassSVM
#from sklearn.neighbors import LocalOutlierFactor
#from sklearn.ensemble import IsolationForest
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.hbos import HBOS
from pyod.models.cblof import CBLOF
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM

# import scikit learn utilities
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer

# import hdbscan library
import hdbscan

# class for generation of artificial anomalies
class AnomalyHandler(object):

    # init anomaly handler
    def __init__(self):

        pass

    # create global graph anomalies
    def generate_global_graph_anomalies(self, statistics, entries, top=20, no_anomalies=10, seed=1111):

        # init stochastic random samplers
        np.random.seed(seed + 1)  # +1 to differentiate random global and local anomalies
        rd.seed(seed + 1) # +1 to differentiate random global and local anomalies

        # aggregate journal entry feature information per entry
        entries_aggregated = self.aggregate_entries_per_belnr(statistics, fields=[statistics['je_identifier_field']] + statistics['je_header_features'], entries=entries)

        # determine journal entry feature information count
        entries_aggregated_count = entries_aggregated.groupby(statistics['je_header_features'] + statistics['je_segment_features_categorical']).count().reset_index()

        # filter top-n most occurring journal entry feature combinations
        entries_aggregated_top_feature_combinations = entries_aggregated_count.sort_values(by=statistics['je_identifier_field'], ascending=False).iloc[0:top]

        # filter top-n most occurring journal entry feature combinations
        entries_aggregated_top_feature_combinations = entries_aggregated_top_feature_combinations[statistics['je_header_features'] + statistics['je_segment_features_categorical']]

        # init data frame of created global anomalies
        global_anomalies = pd.DataFrame(columns=entries.columns)

        # randomly sample top-n most occurring journal entry feature combinations
        anomaly_samples_ids = np.random.choice(list(range(0, top)), size=no_anomalies, replace=False)

        # iterate over number of to be created anomalies
        for anomaly_sample_id in anomaly_samples_ids:

            # determine random top-n occurring journal entry feature combination
            single_random_top_feature_combination = dict(entries_aggregated_top_feature_combinations.iloc[anomaly_sample_id])

            # determine aggregated entries of random top-n journal entry feature combination
            entries_aggregated_single_random_top_feature_combination = entries_aggregated[entries_aggregated[single_random_top_feature_combination.keys()].isin(single_random_top_feature_combination.values()).all(axis=1)]

            # select entry ids of random top-n journal entry feature combination
            entries_aggregated_single_random_top_feature_combination_ids = entries_aggregated_single_random_top_feature_combination[statistics['je_identifier_field']]

            # select random entry id of random top-n journal entry feature combination
            entries_aggregated_single_random_top_feature_combination_id = entries_aggregated_single_random_top_feature_combination_ids.iloc[np.random.choice(list(range(0, len(entries_aggregated_single_random_top_feature_combination_ids))))]

            # determine single random entry of random top-n journal entry feature combination
            single_random_top_feature_combination_entry = entries[entries[statistics['je_identifier_field']] == entries_aggregated_single_random_top_feature_combination_id]

            # update single random entry with random gl account numbers
            single_random_top_feature_combination_entry[statistics['je_gl_account_field']] = [rd.randint(90000, 99999) for ele in single_random_top_feature_combination_entry[statistics['je_gl_account_field']]]

            # update single random entry with new journal entry identifier column
            single_random_top_feature_combination_entry[statistics['je_identifier_field']] = ['AG' + str(ele) for ele in single_random_top_feature_combination_entry[statistics['je_identifier_field']]]

            # collect created global anomaly
            global_anomalies = global_anomalies.append(single_random_top_feature_combination_entry, ignore_index=True)

        # add global anomaly class label
        global_anomalies[statistics['je_class_field']] = 1
        global_anomalies[statistics['je_class_name_field']] = 'global'

        # return created global anomalies
        return global_anomalies

    # create local graph anomalies
    def generate_local_graph_anomalies(self, statistics, entries, top=20, no_anomalies=10, seed=1111):

        # init stochastic random samplers
        np.random.seed(seed + 2)  # +2 to differentiate random global and local anomalies
        rd.seed(seed + 2)  # +2 to differentiate random global and local anomalies

        # aggregate journal entry feature information per entry
        entries_aggregated = self.aggregate_entries_per_belnr(statistics, fields=[statistics['je_identifier_field']] + statistics['je_header_features'], entries=entries)

        # determine journal entry feature information count
        entries_aggregated_count = entries_aggregated.groupby(statistics['je_header_features'] + statistics['je_segment_features_categorical']).count().reset_index()

        # filter top-n most occurring journal entry feature combinations
        entries_aggregated_top_feature_combinations = entries_aggregated_count.sort_values(by=statistics['je_identifier_field'], ascending=False).iloc[0:top]

        # filter top-n most occurring journal entry feature combinations
        entries_aggregated_top_feature_combinations = entries_aggregated_top_feature_combinations[statistics['je_header_features'] + statistics['je_segment_features_categorical']]

        # init data frame of created global anomalies
        local_anomalies = pd.DataFrame(columns=entries.columns)

        # determine unique gl accounts evident in the original dataset
        entries_gl_accounts = entries[statistics['je_gl_account_field']].unique()

        # randomly sample top-n most occurring journal entry feature combinations
        anomaly_samples_ids = np.random.choice(list(range(0, top)), size=no_anomalies, replace=False)

        # iterate over number of to be created anomalies
        for anomaly_sample_id in anomaly_samples_ids:

            # determine random top-n occurring journal entry feature combination
            single_random_top_feature_combination = dict(entries_aggregated_top_feature_combinations.iloc[anomaly_sample_id])

            # determine aggregated entries of random top-n journal entry feature combination
            entries_aggregated_single_random_top_feature_combination = entries_aggregated[entries_aggregated[single_random_top_feature_combination.keys()].isin(single_random_top_feature_combination.values()).all(axis=1)]

            # select entry ids of random top-n journal entry feature combination
            entries_aggregated_single_random_top_feature_combination_ids = entries_aggregated_single_random_top_feature_combination[statistics['je_identifier_field']]

            # select random entry id of random top-n journal entry feature combination
            entries_aggregated_single_random_top_feature_combination_id = entries_aggregated_single_random_top_feature_combination_ids.iloc[np.random.choice(list(range(0, len(entries_aggregated_single_random_top_feature_combination_ids))))]

            # determine single random entry of random top-n journal entry feature combination
            single_random_top_feature_combination_entry = entries[entries[statistics['je_identifier_field']] == entries_aggregated_single_random_top_feature_combination_id]

            # update single random entry with random gl account numbers evident in the original dataset
            single_random_top_feature_combination_entry[statistics['je_gl_account_field']] = [entries_gl_accounts[rd.randint(0, len(entries_gl_accounts)-1)] for ele in single_random_top_feature_combination_entry[statistics['je_gl_account_field']]]

            # update single random entry with new journal entry identifier column
            single_random_top_feature_combination_entry[statistics['je_identifier_field']] = ['AL' + str(ele) for ele in single_random_top_feature_combination_entry[statistics['je_identifier_field']]]

            # collect created global anomaly
            local_anomalies = local_anomalies.append(single_random_top_feature_combination_entry, ignore_index=True)

        # add global anomaly class label
        local_anomalies[statistics['je_class_field']] = 2
        local_anomalies[statistics['je_class_name_field']] = 'local'

        # return created local anomalies
        return local_anomalies

    # aggregate entries on belnr, and hkont level
    def aggregate_entries_per_belnr(self, statistics, fields, entries):

        # aggregate journal entry line items
        aggregated_entries = entries[fields].groupby(fields).count()

        # reset the aggregation index
        aggregated_entries = aggregated_entries.reset_index()

        # iterate over categorical line item attributes
        for segment_feature in statistics['je_segment_features_categorical']:

            # convert to categorical
            entries[segment_feature] = entries[segment_feature].astype(str)

            # case: je line item attribute
            if segment_feature == 'Y_JE_ITEM_IDENTIFIER':

                # aggregate line item attributes
                aggregated_entries[segment_feature] = entries.groupby(fields)[segment_feature].max().values

            # case: other attribute
            else:

                # aggregate line item attributes
                aggregated_entries[segment_feature] = entries.groupby(fields)[segment_feature].apply(' :: '.join).values

                # trim aggregated line item attributes
                aggregated_entries[segment_feature] = [ele[0:50] for ele in aggregated_entries[segment_feature]]

        # return aggregated entries
        return aggregated_entries

    # run hdbscan parameter grid search
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

    # run latent space anomaly detection
    def run_anomaly_detection(self, parameter, results, entries):
        #QH revised 2/28/2024
        # init number of clusters
        results['no_detected_clusters'] = 0

        # init number of global and local anomalies
        results['no_detected_global_anomalies'] = 0
        results['no_detected_local_anomalies'] = 0

        # init anomaly prediction
        entries['Y_ANOMALY_CLASS'] = -1

        # init anomaly score
        entries['Y_ANOMALY_SCORE'] = 0.0

        # init lof parameter
        parameter['best_n_neighbors'] = -1
        parameter['best_leaf_size'] = -1

        # init hdbscan parameter
        parameter['best_min_cluster_size'] = -1
        parameter['best_min_samples'] = -1
        parameter['best_metric'] = -1
        parameter['best_cluster_selection_method'] = -1

        # case: one-class svm anomaly detection
        if parameter['algo'] == 'ocsvm':

            # init the one-class anomaly detection model
            ocsvm_model = OCSVM()
            ocsvm_model.fit(entries[['z1', 'z2']])

            # determine anomaly detection prediction
            entries['Y_ANOMALY_CLASS'] = ocsvm_model.labels_

            # determine anomaly detection score
            entries['Y_ANOMALY_SCORE'] = ocsvm_model.decision_scores_

        # case: local outlier factor anomaly detection
        elif parameter['algo'] == 'lof':

            # update best experiment parameter
            #parameter['best_n_neighbors'] = parameter['n_neighbors']
            #parameter['best_leaf_size'] = parameter['leaf_size']

            # init the local outlier factor model
            lof_model = LOF()
            lof_model.fit(entries[['z1', 'z2']])

            # determine anomaly detection prediction
            entries['Y_ANOMALY_CLASS'] = lof_model.labels_

            # determine anomaly detection score
            entries['Y_ANOMALY_SCORE'] = lof_model.decision_scores_

        # case: isolation forest anomaly detection
        elif parameter['algo'] == 'iforest':

            # init the local outlier factor model
            iforest_model = IForest()
            iforest_model.fit(entries[['z1', 'z2']])

            # determine anomaly detection prediction
            entries['Y_ANOMALY_CLASS'] = iforest_model.labels_

            # determine anomaly detection score
            entries['Y_ANOMALY_SCORE'] = iforest_model.decision_scores_

        elif parameter['algo'] == 'knn':

            # init the local outlier factor model
            knn_model = KNN(n_neighbors=parameter['n_neighbors_knn'], leaf_size=parameter['leaf_size_knn'])
            knn_model.fit(entries[['z1', 'z2']])
            #pred = knn_model.predict(entries[['z1', 'z2']])

            # determine anomaly detection prediction
            entries['Y_ANOMALY_CLASS'] = knn_model.labels_

            # determine anomaly detection score
            entries['Y_ANOMALY_SCORE'] = knn_model.decision_scores_

        elif parameter['algo'] == 'hbos':

            # init the local outlier factor model
            hbos_model = HBOS()
            hbos_model.fit(entries[['z1', 'z2']])
            #pred = hbos_model.predict(entries[['z1', 'z2']])

            # determine anomaly detection prediction
            entries['Y_ANOMALY_CLASS'] = hbos_model.labels_

            # determine anomaly detection score
            entries['Y_ANOMALY_SCORE'] = hbos_model.decision_scores_

        elif parameter['algo'] == 'cblof':

            # init the local outlier factor model
            cblof_model = CBLOF()
            cblof_model.fit(entries[['z1', 'z2']])
            #pred = cblof_model.predict(entries[['z1', 'z2']])

            # determine anomaly detection prediction
            entries['Y_ANOMALY_CLASS'] = cblof_model.labels_

            # determine anomaly detection score
            entries['Y_ANOMALY_SCORE'] = cblof_model.decision_scores_



        # case: isolation hdbscan anomaly detection
        elif parameter['algo'] == 'hdbscan':

            # grid search best hdbscan parameters
            best_parameters = self.grid_search_hdbscan_parameter(parameter=parameter, aggregated_entries=entries)

            # update experiment parameter
            parameter['best_min_cluster_size'] = best_parameters['min_cluster_size']
            parameter['best_min_samples'] = best_parameters['min_samples']
            parameter['best_metric'] = best_parameters['metric']
            parameter['best_cluster_selection_method'] = best_parameters['cluster_selection_method']

            # init the hdbscan model with grid searched parameters
            hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=parameter['min_cluster_size'], min_samples=parameter['min_samples'], metric=parameter['metric'], cluster_selection_method=parameter['cluster_selection_method'])

            # determine hdbscan clustering prediction
            predictions = hdbscan_model.fit_predict(entries[['z1', 'z2']])

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
            entries['Y_ANOMALY_CLASS'] = predictions

            # collect local outlier factor anomaly scores resultsß
            entries['Y_ANOMALY_SCORE'] = scores

            # collect local outlier factor anomaly prediction results
            entries['Y_ANOMALY_LABEL'] = 'Regular'
            entries['Y_ANOMALY_LABEL'] = np.where(entries['Y_ANOMALY_CLASS'] == -1, 'Global Anomaly', entries['Y_ANOMALY_LABEL'])
            entries['Y_ANOMALY_LABEL'] = np.where(entries['Y_ANOMALY_CLASS'] == -2, 'Local Anomaly', entries['Y_ANOMALY_LABEL'])

            # determine global anomalies
            global_anomalies = entries[entries['Y_ANOMALY_CLASS'] == -1]

            # determine local anomalies
            local_anomalies = entries[entries['Y_ANOMALY_CLASS'] == -2]

            # determine number of clusters
            results['no_detected_clusters'] = int(len(entries['Y_ANOMALY_CLASS'].unique())-2)

            # determine number of global and local anomalies
            results['no_detected_global_anomalies'] = int(global_anomalies.shape[0])
            results['no_detected_local_anomalies'] = int(local_anomalies.shape[0])

        # return anomaly detection results
        return results, entries
