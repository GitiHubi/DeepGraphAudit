# import python libraries
import os

# limit the number of threads
import torch

os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4" # export NUMEXPR_NUM_THREADS=6
print("NUMBER OF THREADS ARE LIMITED NOW ...")

# import libraries
import datetime as dt
import numpy as np
import pandas as pd
import itertools as it

# import networkx
import networkx as nx

# import sklearn
from sklearn.preprocessing import LabelEncoder

# import project libraries
from AnomalyHandler import AnomalyHandler
from VisualisationHandler import VisualisationHandler

# class DataHandler
class DataHandler(object):

    def __init__(self):

        # init anomaly handler
        self.aha = AnomalyHandler.AnomalyHandler()

        # init visualization handler
        self.vha = VisualisationHandler.VisualisationHandler()

    def get_ernstyoung_data_range(self, parameter):

        # case: sample evaluation enabled
        if parameter['sample_eval'] is True and parameter['dataset'].split('/')[-1] == 'ernstyoung':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '01_ernstyoung', 'Analytics_mindset_case_studies_PSU_Hotel_data.csv'), sep=',', encoding='utf-8').iloc[0:parameter['sample_size'], :]

        # case: sample evaluation not enbaled
        elif parameter['sample_eval'] is False and parameter['dataset'].split('/')[-1] == 'ernstyoung':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '01_ernstyoung', 'Analytics_mindset_case_studies_PSU_Hotel_data.csv'), sep=',', encoding='utf-8')

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: transactional EY data of shape {} rows and {} columns successfully loaded.'.format(now, str(original_entries.shape[0]), str(original_entries.shape[1])))

        ### Step 0: Define Data Attributes and Statistcs ############################################################

        # init client transactions statistics
        entries_statistics = {}

        # determine the categorical attributes
        cat_attr = [
            'JEIdentifier'  # The unique identifier of the journal entry.
            , 'AccountType'  # The type of the general ledger account.
            , 'AccountClass'  # The class of the general ledger account.
            , 'GLAccountNumber'  # The number of the general ledger account.
            , 'GLAccountName'  # The name of the general ledger account.
            , 'PreparerID'  # The id of the entry preparer.
            , 'Source'  # The source of the entry.
        ]

        # determine the numerical attributes
        num_attr = ['Amount'  # The amount of the journal entry (USD$).
                    ]

        # collect categorical and numerical attributes
        entries_statistics['cat_attr'] = cat_attr
        entries_statistics['num_attr'] = num_attr

        ### Step 1: Pre-process journal entries attribute values #####################################################

        # pre-process the categorical journal entry attributes
        processed_categorical_entries, created_cat_attributes = self.preprocess_ernstyoung_categorical_attributes(entries=original_entries, categorical_attributes=cat_attr)

        ### Step 2: One-hot encode journal entry attribute values ####################################################

        # encode the pre-processed categorical transaction attributes
        encoded_categorical_entries, entries_statistics = self.one_hot_encode_ernstyoung_categorical_attributes(entries=processed_categorical_entries, categorical_attributes=created_cat_attributes, encoded_attributes=['Y_GL_ACCOUNT_NUMBER'], entries_statistics=entries_statistics)

        ### Step 3: Aggregate journal entry attribute values #########################################################

        # aggregate the pre-processed categorical transaction attributes
        aggregated_categorical_entries = self.aggregate_ernstyoung_categorical_attributes(entries=encoded_categorical_entries, header_attributes=['Y_JE_IDENTIFIER', 'Y_PREPARER_ID', 'Y_SOURCE'], line_attributes=['Y_JE_ITEM_IDENTIFIER', 'Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME'])

        ### Step 4: Prepare journal entry attribute training values #########################################################

        # prepare the pre-processed categorical transaction attributes
        prepared_categorical_entries = aggregated_categorical_entries.set_index(aggregated_categorical_entries['Y_JE_IDENTIFIER']).drop(['Y_JE_IDENTIFIER', 'Y_PREPARER_ID', 'Y_SOURCE'], axis=1)

        # return original and encoded transactions
        return processed_categorical_entries, aggregated_categorical_entries, prepared_categorical_entries, entries_statistics

    # load the EY training data
    def load_ey_data(self, parameter):

        # case: sample evaluation enabled
        if parameter['sample_eval'] is True and parameter['dataset'].split('/')[-1] == 'ey':

            # read the transactional data
            original_dataset = pd.read_csv(os.path.join(parameter['data_dir'], '01_ernstyoung', 'Analytics_mindset_case_studies_PSU_Hotel_data_v1.csv'), sep=',', encoding='utf-8').iloc[0:parameter['sample_size'], :]

        # case: sample evaluation not enbaled
        elif parameter['sample_eval'] is False and parameter['dataset'].split('/')[-1] == 'ey':

            # read the transactional data
            original_dataset = pd.read_csv(os.path.join(parameter['data_dir'], '01_ernstyoung', 'Analytics_mindset_case_studies_PSU_Hotel_data_v1.csv'), sep=',', encoding='utf-8')

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: {} transactional data of shape {} rows, {} columns, and belnr {}, successfully loaded.'.format(now, str(parameter['dataset']).upper(), str(original_dataset.shape[0]), str(original_dataset.shape[1]), str(len(original_dataset['JEIdentifier'].unique()))))

        # return the transactional data
        return original_dataset

    # load the SAP training data
    def load_sap_data(self, parameter):

        # case: sample evaluation enabled
        if parameter['sample_eval'] is True and parameter['dataset'].split('/')[-1] == 'sap':

            # read the transactional data
            original_dataset = pd.read_csv(os.path.join(parameter['data_dir'], '03_sap', '2023-01-27_SF_extract.csv'), sep=',', encoding='utf-8').iloc[0:parameter['sample_size'], :]

        # case: sample evaluation not enbaled
        elif parameter['sample_eval'] is False and parameter['dataset'].split('/')[-1] == 'sap':

            # read the transactional data
            original_dataset = pd.read_csv(os.path.join(parameter['data_dir'], '03_sap', '2023-01-27_SF_extract.csv'), sep=',', encoding='utf-8')

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: {} transactional data of shape {} rows, {} columns, and belnr {}, successfully loaded.'.format(now, str(parameter['dataset']), str(original_dataset.shape[0]), str(original_dataset.shape[1]), str(len(original_dataset['DocumentNr'].unique()))))

        # return the transactional data
        return original_dataset

    # pre-process the EY training data
    def preprocess_ey_data(self, statistics, dataset):

        # processed journal entry attributes
        preprocessed_detailed_entries = pd.DataFrame({statistics['je_identifier_field']: dataset['JEIdentifier'].astype(str)})

        # pre-process the categorical journal entry attributes
        categorical_attributes = statistics['je_header_attributes'] + statistics['je_segment_attributes_categorical']
        categorical_features = statistics['je_header_features'] + statistics['je_segment_features_categorical']
        preprocessed_detailed_cat_entries = self.preprocess_ey_categorical_attributes(entries=dataset, categorical_attributes=categorical_attributes, categorical_features=categorical_features)

        # pre-process the numerical journal entry attributes
        numerical_attributes = statistics['je_segment_attributes_numerical']
        numerical_features = statistics['je_segment_features_numerical']
        preprocessed_detailed_num_entries = self.preprocess_ey_numerical_attributes(entries=dataset, numerical_attributes=numerical_attributes, numerical_features=numerical_features)

        # combine pre-processed categorical and numerical attributes
        preprocessed_detailed_entries = pd.concat([preprocessed_detailed_entries, preprocessed_detailed_cat_entries, preprocessed_detailed_num_entries], axis=1)

        # return pre-processed detailed entries
        return preprocessed_detailed_entries

    # pre-process the SAP training data
    def preprocess_sap_data(self, statistics, dataset):

        # processed journal entry attributes
        preprocessed_detailed_entries = pd.DataFrame({statistics['je_identifier_field']: dataset['DocumentNr'].astype(str)})

        # pre-process the categorical journal entry attributes
        categorical_attributes = statistics['je_header_attributes'] + statistics['je_segment_attributes_categorical']
        categorical_features = statistics['je_header_features'] + statistics['je_segment_features_categorical']
        preprocessed_detailed_cat_entries = self.preprocess_sap_categorical_attributes(entries=dataset, categorical_attributes=categorical_attributes, categorical_features=categorical_features)

        # pre-process the numerical journal entry attributes
        numerical_attributes = statistics['je_segment_attributes_numerical']
        numerical_features = statistics['je_segment_features_numerical']
        preprocessed_detailed_num_entries = self.preprocess_sap_numerical_attributes(entries=dataset, numerical_attributes=numerical_attributes, numerical_features=numerical_features)

        # combine pre-processed categorical and numerical attributes
        preprocessed_detailed_entries = pd.concat([preprocessed_detailed_entries, preprocessed_detailed_cat_entries, preprocessed_detailed_num_entries], axis=1)

        # return pre-processed detailed entries
        return preprocessed_detailed_entries

    # filter the SAP training data
    def filter_sap_data(self, parameter, dataset):

        # aggregate journal entry line items
        aggregated_original_entries = dataset[['DocumentNr', 'LineItem']].groupby(['DocumentNr']).count()

        # reset the aggregation index
        aggregated_original_entries = aggregated_original_entries.reset_index()

        # determine manual (non automated) journal entry postings
        selected_belnr = aggregated_original_entries[(aggregated_original_entries['LineItem'] >= parameter['min_line_items']) & (aggregated_original_entries['LineItem'] <= parameter['max_line_items'])]['DocumentNr']

        # filter for manual (non automated) journal entry postings
        filtered_entries = dataset[dataset['DocumentNr'].isin(selected_belnr)]

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: transactional {} data of shape {} rows, {} columns, and belnr {}, successfully loaded.'.format(now, str(parameter['dataset']).upper(), str(filtered_entries.shape[0]), str(filtered_entries.shape[1]), str(len(filtered_entries['DocumentNr'].unique()))))

        # return filtered journal entries
        return filtered_entries

    # add anomalies to pre-processed data
    def add_data_anomalies(self, parameter, statistics, dataset):

        # init pre-processed, global and local anomaly entries
        anomalies_detailed_entries = dataset.copy(deep=True)

        # add data class label
        anomalies_detailed_entries[statistics['je_class_field']] = 0
        anomalies_detailed_entries[statistics['je_class_name_field']] = 'regular'

        # create global graph anomalies
        global_anomalies = self.aha.generate_global_graph_anomalies(statistics=statistics, entries=dataset, no_anomalies=parameter['no_global_anomalies'], seed=parameter['seed'])

        # create local graph anomalies
        local_anomalies = self.aha.generate_local_graph_anomalies(statistics=statistics, entries=dataset, no_anomalies=parameter['no_local_anomalies'], seed=parameter['seed'])

        # combine pre-processed, global and local anomaly entries
        anomalies_detailed_entries = pd.concat([anomalies_detailed_entries, global_anomalies, local_anomalies], axis=0)

        # determine pre-processed global and local anomaly entry class labels
        anomalies_detailed_entries_labels = anomalies_detailed_entries[[statistics['je_identifier_field']] + [statistics['je_gl_account_field']] + [statistics['je_class_field']] + [statistics['je_class_name_field']]]

        # remove pre-processed, global and local anomaly entry class label information
        anomalies_detailed_entries = anomalies_detailed_entries[dataset.columns]

        # reset indexes upon anomaly entry injection
        anomalies_detailed_entries.reset_index(inplace=True, drop=True)
        anomalies_detailed_entries_labels.reset_index(inplace=True, drop=True)

        # return anomalies dataset
        return anomalies_detailed_entries, anomalies_detailed_entries_labels

    # aggregate pre-processed data
    def aggregate_data(self, statistics, data, labels):

        # aggregate journal entries on an account posting line item level (belnr, hkont, and shkzg) -> used to create the adjacency matrices
        fields = [statistics['je_identifier_field']] + [statistics['je_gl_account_field']] + [statistics['je_line_item_field']] + [statistics['je_debit_credit_field']]
        belnr_hkont_shkzg_entries = self.aggregate_entries_per_belnr_hkont_shkzg(statistics=statistics, fields=fields, entries=data)

        # aggregate journal entries on an account posting level (belnr and hkont)  -> used to create the feature matrices
        fields = [statistics['je_identifier_field']] + [statistics['je_gl_account_field']]
        belnr_hkont_entries = self.aggregate_entries_per_belnr_hkont(statistics=statistics, fields=fields, entries=data, labels=labels)

        # aggregate journal entries on belnr level -> used to visualize the results
        fields = [statistics['je_identifier_field']]
        belnr_entries = self.aggregate_entries_per_belnr(statistics=statistics, fields=fields, entries=data, labels=labels)

        # determine journal entry data statistics
        statistics['no_journal_entries'] = len(belnr_entries[statistics['je_identifier_field']].unique())
        statistics['no_posting_accounts'] = len(belnr_hkont_entries[statistics['je_gl_account_field']].unique())
        statistics['no_posting_features'] = len(statistics['je_features'])

        # return distinct aggregations
        return belnr_hkont_shkzg_entries, belnr_hkont_entries, belnr_entries, statistics

    # encode pre-processed data
    def encode_data_attributes(self, parameter, statistics, data):

        # encode journal entry header features
        feat_header, statistics = self.encode_je_header_features(statistics=statistics, entries=data, type=parameter['encoder_type'])

        # encoder journal entry segment features
        feat_segment, statistics = self.encode_je_segment_features(statistics=statistics, entries=data, type=parameter['encoder_type'])

        # combine journal entry header and segment features
        features = pd.concat([feat_header, feat_segment], axis=1)

        # drop duplicate identifier column
        features = features.loc[:,~features.columns.duplicated()]

        # return encoded features
        return data, features, statistics

    def create_feat_vectors(self, statistics, data):

        # copy feature vectors
        feat_vectors = data.copy(deep=True)

        # filter for seleted journal entry features
        feat_vectors = feat_vectors[statistics['je_features']]

        # convert to numpy array
        feat_vectors = feat_vectors.to_numpy().astype('float64')

        # return feature vectors
        return feat_vectors

    def get_gnn_data_range_sap(self, parameter, statistics):

        # case: sample evaluation enabled
        if parameter['sample_eval'] is True and parameter['dataset'].split('/')[-1] == 'sap':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '03_sap', '2023-01-27_SF_extract.csv'), sep=',', encoding='utf-8').iloc[0:parameter['sample_size'], :]

        # case: sample evaluation not enbaled
        elif parameter['sample_eval'] is False and parameter['dataset'].split('/')[-1] == 'sap':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '03_sap', '2023-01-27_SF_extract.csv'), sep=',', encoding='utf-8')

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: {} transactional data of shape {} rows, {} columns, and belnr {}, successfully loaded.'.format(now, str(parameter['dataset']), str(original_entries.shape[0]), str(original_entries.shape[1]), str(len(original_entries['DocumentNr'].unique()))))

        ### Step 1: Filter large-scale automated postings ####################################################

        # aggregate journal entry line items
        aggregated_original_entries = original_entries[['DocumentNr', 'LineItem']].groupby(['DocumentNr']).count()

        # reset the aggregation index
        aggregated_original_entries = aggregated_original_entries.reset_index()

        # determine manual (non automated) journal entry postings
        selected_belnr = aggregated_original_entries[(aggregated_original_entries['LineItem'] >= parameter['min_line_items']) & (aggregated_original_entries['LineItem'] <= parameter['max_line_items'])]['DocumentNr']

        # filter for manual (non automated) journal entry postings
        original_entries = original_entries[original_entries['DocumentNr'].isin(selected_belnr)]

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: transactional {} data of shape {} rows, {} columns, and belnr {}, successfully loaded.'.format(now, str(parameter['dataset']).upper(), str(original_entries.shape[0]), str(original_entries.shape[1]), str(len(original_entries['DocumentNr'].unique()))))

        ### Step 2: Pre-process journal entries attribute values ############################################

        # processed journal entry attributes
        processed_detailed_entries = pd.DataFrame({statistics['je_identifier_field']: original_entries['DocumentNr'].astype(str)})

        # pre-process the categorical journal entry attributes
        categorical_attributes = statistics['je_header_attributes'] + statistics['je_segment_attributes_categorical']
        categorical_features = statistics['je_header_features'] + statistics['je_segment_features_categorical']
        processed_detailed_cat_entries = self.preprocess_sap_categorical_attributes(entries=original_entries, categorical_attributes=categorical_attributes, categorical_features=categorical_features)

        # pre-process the numerical journal entry attributes
        numerical_attributes = statistics['je_segment_attributes_numerical']
        numerical_features = statistics['je_segment_features_numerical']
        processed_detailed_num_entries = self.preprocess_sap_numerical_attributes(entries=original_entries, numerical_attributes=numerical_attributes, numerical_features=numerical_features)

        # combine pre-processed categorical and numerical attributes
        processed_detailed_entries = pd.concat([processed_detailed_entries, processed_detailed_cat_entries, processed_detailed_num_entries], axis=1)

        ### Step 3: Create and add global/local anomalies ############################################################

        # init pre-processed, global and local anomaly entries
        processed_detailed_entries_anomalies = processed_detailed_entries.copy(deep=True)

        # add data class label
        processed_detailed_entries_anomalies[statistics['je_class_field']] = 0
        processed_detailed_entries_anomalies[statistics['je_class_name_field']] = 'regular'

        # create global graph anomalies
        global_anomalies = self.aha.generate_global_graph_anomalies(statistics=statistics, entries=processed_detailed_entries, no_anomalies=parameter['no_global_anomalies'], seed=parameter['seed'])

        # create local graph anomalies
        local_anomalies = self.aha.generate_local_graph_anomalies(statistics=statistics, entries=processed_detailed_entries, no_anomalies=parameter['no_local_anomalies'], seed=parameter['seed'])

        # combine pre-processed, global and local anomaly entries
        processed_detailed_entries_anomalies = pd.concat([processed_detailed_entries_anomalies, global_anomalies, local_anomalies], axis=0)

        # determine pre-processed global and local anomaly entry class labels
        processed_detailed_entries_labels = processed_detailed_entries_anomalies[[statistics['je_identifier_field']] + [statistics['je_gl_account_field']] + [statistics['je_class_field']] + [statistics['je_class_name_field']]]

        # remove pre-processed, global and local anomaly entry class label information
        processed_detailed_entries = processed_detailed_entries_anomalies[processed_detailed_entries.columns]

        ### Step 4: Encode general ledger account attribute ##################################################

        # encode general ledger account attribute
        processed_detailed_entries[statistics['je_gl_account_field']] = pd.Categorical(processed_detailed_entries[statistics['je_gl_account_field']]).codes

        ### Step 5: Aggregate journal entry attribute values #################################################

        # aggregate journal entries on a belnr, hkont, and shkzg level -> used to create the adjacency matrices
        fields = [statistics['je_identifier_field']] + [statistics['je_gl_account_field']] + [statistics['je_line_item_field']] + [statistics['je_debit_credit_field']]
        belnr_hkont_shkzg_entries = self.aggregate_entries_per_belnr_hkont_shkzg(statistics=statistics, fields=fields, entries=processed_detailed_entries)

        # aggregate journal entries on belnr and hkont level -> used to create the feature matrices
        fields = [statistics['je_identifier_field']] + [statistics['je_gl_account_field']]
        belnr_hkont_entries = self.aggregate_entries_per_belnr_hkont(statistics=statistics, fields=fields, entries=processed_detailed_entries)

        # aggregate journal entries on belnr level -> used to visualize the results
        fields = [statistics['je_identifier_field']]
        belnr_entries = self.aggregate_entries_per_belnr(statistics=statistics, fields=fields, entries=processed_detailed_entries, labels=processed_detailed_entries_labels)

        ### Step 6: Encode journal entry attribute values ####################################################

        # determine unique posted journal entries
        statistics['no_journal_entries'] = len(belnr_entries[statistics['je_identifier_field']].unique())

        # determine unique posted accounts
        statistics['no_posting_accounts'] = len(belnr_hkont_entries[statistics['je_gl_account_field']].unique())

        # determine unique posted features
        statistics['no_posting_features'] = len(statistics['je_features'])

        # encode the pre-processed categorical transaction attributes
        belnr_hkont_entries_encoded, statistics, belnr_hkont_entries = self.encode_je_attributes(parameter=parameter, statistics=statistics, entries=belnr_hkont_entries)

        ### Step 7: Create feature and adjacency matrices ####################################################

        # create the graph learning adjacency and feature matrices
        posting_ids, adj_matrices, feat_matrices, statistics = self.create_adj_feat_matrices(parameter=parameter, statistics=statistics, adjacencies=belnr_hkont_shkzg_entries, features=belnr_hkont_entries, encoded_features=belnr_hkont_entries_encoded)

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: {} transactional data, {} adjacency matrices and {} feature matrices created.'.format(now, str(parameter['dataset']).upper(), str(len(adj_matrices)), str(len(feat_matrices))))

        ### Step 8: Visualise journal entry graphs ###########################################################

        # case: graph visualization enabled
        if parameter['visualize']:

            # visualize journal entries graphs
            # self.visualize_single_entries_graph(parameter=parameter, statistics=statistics, adjacencies=belnr_hkont_shkzg_entries, features=belnr_hkont_entries, encoded_features=belnr_hkont_entries_encoded)

            # visualize journal entries graphs
            self.visualize_entire_entries_graph(parameter=parameter, statistics=statistics, adjacencies=belnr_hkont_shkzg_entries, features=belnr_hkont_entries, encoded_features=belnr_hkont_entries_encoded)

        # return original and encoded transactions
        return posting_ids, adj_matrices, feat_matrices, belnr_hkont_entries, belnr_entries, statistics

    def get_gnn_data_range_serpro(self, parameter):

        # case: sample evaluation enabled
        if parameter['sample_eval'] is True and parameter['dataset'].split('/')[-1] == 'serpro':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '02_serpro', '2023-01-25_Serpro_Extraction_ERP_1stQ_2ndQ.csv'), sep=',', encoding='utf-8').iloc[0:parameter['sample_size'], :]

        # case: sample evaluation not enbaled
        elif parameter['sample_eval'] is False and parameter['dataset'].split('/')[-1] == 'serpro':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '02_serpro', '2023-01-25_Serpro_Extraction_ERP_1stQ_2ndQ.csv'), sep=',', encoding='utf-8')

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: transactional Serpro data of shape {} rows and {} columns successfully loaded.'.format(now, str(original_entries.shape[0]), str(original_entries.shape[1])))

        ### Step 0: Define Data Attributes and Statistcs ############################################################

        # init client transactions statistics
        entries_statistics = {}

        # determine the categorical attributes
        cat_attr = [
            'Doc'  # The unique identifier of the journal entry.
            , 'Cd_Account'  # The number of the general ledger account.
            , 'Dsc_Account'  # The name of the general ledger account.
            , 'PreparerID'  # The id of the entry preparer.
            , 'Source'  # The source of the entry.
        ]

        # determine the numerical attributes
        num_attr = ['Amount'  # The amount of the journal entry (USD$).
                    ]

        # collect categorical and numerical attributes
        entries_statistics['cat_attr'] = cat_attr
        entries_statistics['num_attr'] = num_attr

        ### Step 1: Pre-process journal entries attribute values #####################################################

        # pre-process the categorical journal entry attributes
        processed_categorical_entries, created_cat_attributes = self.preprocess_ernstyoung_categorical_attributes(entries=original_entries, categorical_attributes=cat_attr)

        ### Step 2: encode journal entry attribute values ####################################################

        # encode the pre-processed categorical transaction attributes
        posting_ids, adj_matrices, feat_matrices, entries_statistics = self.encode_ernstyoung_gnn_categorical_attributes(entries=processed_categorical_entries, categorical_attributes=created_cat_attributes, encoded_attributes=['Y_GL_ACCOUNT_NUMBER'], entries_statistics=entries_statistics)

        ### Step 3: Aggregate journal entry attribute values #########################################################

        # aggregate the pre-processed categorical transaction attributes
        aggregated_entries = self.aggregate_ernstyoung_gnn_categorical_attributes(entries=processed_categorical_entries, header_attributes=['Y_JE_IDENTIFIER'], line_attributes=['Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME', 'Y_DEBIT_CREDIT'])

        # return original and encoded transactions
        return posting_ids, adj_matrices, feat_matrices, aggregated_entries, entries_statistics

    # pre-process categorical attributes of the EY dataset
    def preprocess_ey_categorical_attributes(self, entries, categorical_attributes, categorical_features):

        # extract the categorical attributes
        cat_entries = entries[categorical_attributes].copy()

        # create JE line item identifier
        cat_entries['Y_JE_ITEM_IDENTIFIER'] = cat_entries.groupby(['JEIdentifier']).cumcount()

        # convert categorical attributes to string
        cat_entries['Y_GL_ACCOUNT_TYPE'] = cat_entries['AccountType'].astype(str)
        cat_entries['Y_GL_ACCOUNT_CLASS'] = cat_entries['AccountClass'].astype(str)
        cat_entries['Y_GL_ACCOUNT_NUMBER'] = cat_entries['GLAccountNumber'].astype(str)
        cat_entries['Y_GL_ACCOUNT_NAME'] = cat_entries['GLAccountName'].astype(str)
        cat_entries['Y_SOURCE'] = cat_entries['Source'].astype(str)

        # convert and clean JE creator field
        cat_entries['Y_PREPARER_ID'] = cat_entries['PreparerID'].astype(str)
        cat_entries['Y_PREPARER_ID'] = [ele.replace(u'\xa0', u' ') for ele in cat_entries['Y_PREPARER_ID']]

        # create JE debit and credit identifier
        cat_entries['Y_DEBIT_CREDIT'] = 0
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where((entries['Debit'].notna()) & (entries['Credit'].isnull()), 'Credit')
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where((entries['Credit'].notna()) & (entries['Debit'].isnull()), 'Debit')

        # iterate over distinct categorical attributes
        for feature in categorical_features:

            # determine distinct attribute values per categorical attribute
            feature_values = cat_entries[feature].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: EY categorical attribute: {}, no. of. distinct values: {}.'.format(now, str(feature), str(feature_values)))

        # return pre-processed numerical transaction attributes
        return cat_entries[categorical_features]

    # pre-process numerical attributes of the EY dataset
    def preprocess_ey_numerical_attributes(self, entries, numerical_attributes, numerical_features):

        # extract the numerical attributes
        num_entries = entries[numerical_attributes].copy()

        # convert categorical attributes to string
        num_entries['Y_DMBTR'] = [float(ele.replace(',','')) for ele in num_entries['Amount']]

        # iterate over distinct categorical attributes
        for attribute in numerical_features:

            # determine distinct attribute values per categorical attribute
            attr_values = num_entries[attribute].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: EY numerical attribute: {}, no. of. distinct values: {}.'.format(now, str(attribute), str(attr_values)))

        # return pre-processed numerical transaction attributes
        return num_entries[numerical_features]

    # pre-process categorical attributes of the SAP dataset
    def preprocess_sap_categorical_attributes(self, entries, categorical_attributes, categorical_features):

        # extract the categorical attributes
        cat_entries = entries[categorical_attributes].copy()

        # create JE line item identifier
        cat_entries['Y_JE_ITEM_IDENTIFIER'] = cat_entries.groupby(['DocumentNr']).cumcount()

        # convert categorical attributes to string
        cat_entries['Y_BLART'] = cat_entries['DocType'].astype(str)
        cat_entries['Y_BLART_TEXT'] = cat_entries['DocTypeDescr'].astype(str)
        cat_entries['Y_HKONT'] = cat_entries['GL Accountnr'].astype(str)
        cat_entries['Y_HKONT_TEXT'] = cat_entries['GL AccountDescr'].astype(str)
        cat_entries['Y_TCODE'] = cat_entries['TransactionDescription'].astype(str)
        cat_entries['Y_BSCHL'] = cat_entries['PostingKey'].astype(str)
        cat_entries['Y_PRCTR'] = cat_entries['ProfitCenter'].astype(str)
        cat_entries['Y_KOSTL'] = cat_entries['CostCenter'].astype(str)

        # convert JE creator field
        cat_entries['Y_USNAM'] = cat_entries['UserName Post'].astype(str)

        # create JE debit and credit identifier
        cat_entries['Y_DEBIT_CREDIT'] = 0
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where(entries['D/C'] == 'H', 'Credit')
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where(entries['D/C'] == 'S', 'Debit')

        # iterate over distinct categorical attributes
        for attribute in categorical_features:

            # determine distinct attribute values per categorical attribute
            attr_values = cat_entries[attribute].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: SAP categorical attribute: {}, no. of. distinct values: {}.'.format(now, str(attribute), str(attr_values)))

        # return pre-processed numerical transaction attributes
        return cat_entries[categorical_features]

    # pre-process numerical attributes of the SAP dataset
    def preprocess_sap_numerical_attributes(self, entries, numerical_attributes, numerical_features):

        # extract the numerical attributes
        num_entries = entries[numerical_attributes].copy()

        # convert categorical attributes to string
        num_entries['Y_DMBTR'] = [float(ele) for ele in num_entries['AmountinUSD']]

        # iterate over distinct categorical attributes
        for attribute in numerical_features:

            # determine distinct attribute values per categorical attribute
            attr_values = num_entries[attribute].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: SAP numerical attribute: {}, no. of. distinct values: {}.'.format(now, str(attribute), str(attr_values)))

        # return pre-processed numerical transaction attributes
        return num_entries[numerical_features]

    # one-hot encode categorical attributes of the EY dataset
    def one_hot_encode_ernstyoung_categorical_attributes(self, entries, categorical_attributes, encoded_attributes, entries_statistics):

        # init the encoded categorical entries
        encoded_cat_entries = pd.DataFrame(entries[categorical_attributes])

        # iterate over categorical attributes
        for i in range(0, len(encoded_attributes)):

            # determine one-hot encoding of current attribute
            encoded_attribute = pd.get_dummies(entries[encoded_attributes[i]])

            # determine number of one-hot encoded dimensions
            entries_statistics[encoded_attributes[i] + '_size'] = encoded_attribute.shape[1]

            # collect and concat one-hot encoding of current attribute
            encoded_cat_entries = pd.concat([encoded_cat_entries, encoded_attribute], axis=1)

        # return one-hot encoded entries and entry statistics
        return encoded_cat_entries, entries_statistics

    # encode categorical attributes of the EY dataset
    def encode_ernstyoung_gnn_categorical_attributes(self, entries, categorical_attributes, encoded_attributes, entries_statistics):

        # init the encoded categorical entries
        encoded_cat_entries = pd.DataFrame(entries[categorical_attributes])

        # one-hot encode categorical attributes
        one_hot_encoded_cat_entries = pd.get_dummies(encoded_cat_entries['Y_PREPARER_ID'])

        # add journal entry identifier to one-hot encode categorical attributes
        one_hot_encoded_cat_entries['Y_JE_IDENTIFIER'] = encoded_cat_entries['Y_JE_IDENTIFIER']

        # remove duplicate one-hot encoded categorical attributes
        one_hot_encoded_cat_entries = one_hot_encoded_cat_entries.drop_duplicates()

        # determine unique posting ids
        posting_ids = encoded_cat_entries.groupby(['Y_JE_IDENTIFIER']).count().index

        # collect number of graph nodes
        entries_statistics['no_journal_entries'] = len(posting_ids)

        # determine unique posted accounts
        posting_accounts = encoded_cat_entries[encoded_attributes[0]].unique()

        # collect number of graph nodes
        entries_statistics['no_accounts'] = len(posting_accounts)

        # encode posting general ledger
        encoded_cat_entries['Y_GL_ACCOUNT_CODE'] = pd.Categorical(encoded_cat_entries[encoded_attributes[0]]).codes

        # init adjacency and feature matrices
        adj_matrices = []
        feat_matrices = []

        # iterate over distinct posting ids
        for i, posting_id in enumerate(posting_ids):

            # create journal entry adjacency matrix of the EY dataset
            adj_matrix = self.create_ernstyoung_adjacency_matrix(no_posting_accounts=len(posting_accounts), encoded_cat_entries=encoded_cat_entries, posting_id=posting_id)

            # collect adjacency matrix
            adj_matrices.append(adj_matrix)

            # create journal entry feature matrix of the EY dataset
            feat_matrix = self.create_ernstyoung_feature_matrix(no_posting_accounts=len(posting_accounts), no_account_features=one_hot_encoded_cat_entries.shape[1], account_features=one_hot_encoded_cat_entries, posting_id=posting_id)

            # collect feature matrix
            feat_matrices.append(feat_matrix)

            # case: log adjacency matrix creation process
            if i % 1000 == 0:

                # log configuration processing
                now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
                print('[INFO {}] DataHandler :: EY adjacency and feature matrix: {} of: {} matrices created.'.format(now, str(i), str(len(posting_ids))))

        # return adjacency
        return posting_ids, adj_matrices, feat_matrices, entries_statistics

    # encode categorical attributes of the EY dataset
    def encode_je_attributes(self, parameter, statistics, entries):

        # encode journal entry header features
        feat_header, statistics = self.encode_je_header_features(statistics, entries, type=parameter['encoder_type'])

        # encoder journal entry segment features
        feat_segment, statistics = self.encode_je_segment_features(statistics, entries, type=parameter['encoder_type'])

        # combine journal entry header and segment features
        features = pd.concat([feat_header, feat_segment], axis=1)

        # drop duplicate identifier column
        features = features.loc[:,~features.columns.duplicated()]

        # return encoded features
        return features, statistics, entries

    def visualize_single_entries_graph(self, parameter, statistics, adjacencies, features, encoded_features):

        # determine unique posting ids
        posting_ids = adjacencies.groupby([statistics['je_identifier_field']]).count().index

        # iterate over distinct posting ids
        for i, posting_id in enumerate(posting_ids):

            # init the networkx posting graph
            entry_graph = nx.Graph()

            # determine current posting line items
            posting_adjacencies = adjacencies[adjacencies[statistics['je_identifier_field']] == posting_id]

            # determine all possible account pairs
            account_pairs = list(it.combinations(posting_adjacencies[statistics['je_gl_account_field']], 2))

            # determine current posting features
            posting_features = features[features[statistics['je_identifier_field']] == posting_id]

            # iterate over distinct account pairs
            for pair in account_pairs:

                # determine pairs debit and credit accounts
                pair_a_debit_credit = pair[0]
                pair_b_debit_credit = pair[1]

                # determine pairs debit and credit account features
                pair_a_debit_credit_name = posting_features[posting_features[statistics['je_gl_account_field']] == str(pair_a_debit_credit)]['Y_GL_ACCOUNT_NAME'] # ['Y_HKONT_TEXT']
                pair_b_debit_credit_name = posting_features[posting_features[statistics['je_gl_account_field']] == str(pair_b_debit_credit)]['Y_GL_ACCOUNT_NAME'] # ['Y_HKONT_TEXT']

                # add graph debit and credit nodes
                entry_graph.add_node(pair_a_debit_credit, account=pair_a_debit_credit_name)
                entry_graph.add_node(pair_b_debit_credit, account=pair_b_debit_credit_name)

                # add graph debit and credit edge
                entry_graph.add_edge(pair_a_debit_credit, pair_b_debit_credit)

            # save plot to plotting directory
            filename = '{}_single_graph_{}.png'.format(str(parameter['exp_timestamp']), str(posting_id).zfill(5))
            self.vha.plot_single_journal_entry_graph(parameter, entry_graph=entry_graph, filename=filename)

    def visualize_entire_entries_graph(self, parameter, statistics, adjacencies, features, encoded_features):

        # determine unique posting ids
        posting_ids = adjacencies.groupby([statistics['je_identifier_field']]).count().index

        # init the networkx posting graph
        entire_entries_graph = nx.Graph()

        # iterate over distinct posting ids
        for i, posting_id in enumerate(posting_ids):

            # determine current posting line items
            posting_adjacencies = adjacencies[adjacencies[statistics['je_identifier_field']] == posting_id]

            # determine all possible account pairs
            account_pairs = list(it.combinations(posting_adjacencies[statistics['je_gl_account_field']], 2))

            # determine current posting features
            posting_features = features[features[statistics['je_identifier_field']] == posting_id]

            # iterate over distinct account pairs
            for pair in account_pairs:

                # determine pairs debit and credit accounts
                pair_a_debit_credit = pair[0]
                pair_b_debit_credit = pair[1]

                # determine pairs debit and credit account features
                pair_a_debit_credit_name = posting_features[posting_features[statistics['je_gl_account_field']] == str(pair_a_debit_credit)]['Y_HKONT_TEXT']
                pair_b_debit_credit_name = posting_features[posting_features[statistics['je_gl_account_field']] == str(pair_b_debit_credit)]['Y_HKONT_TEXT']

                # add graph debit and credit edge
                entire_entries_graph.add_node(pair_a_debit_credit, account=pair_a_debit_credit_name)
                entire_entries_graph.add_node(pair_b_debit_credit, account=pair_b_debit_credit_name)

                # add graph debit and credit edge
                entire_entries_graph.add_edge(pair_a_debit_credit, pair_b_debit_credit)

        # determine plotting positions
        pos_nodes = nx.spring_layout(entire_entries_graph)

        # init the networkx posting graph
        partial_entries_graph = nx.Graph()

        # iterate over distinct posting ids
        for i, posting_id in enumerate(posting_ids):

            # determine current posting line items
            posting_adjacencies = adjacencies[adjacencies[statistics['je_identifier_field']] == posting_id]

            # determine all possible account pairs
            account_pairs = list(it.combinations(posting_adjacencies[statistics['je_gl_account_field']], 2))

            # determine current posting features
            posting_features = features[features[statistics['je_identifier_field']] == posting_id]

            # iterate over distinct account pairs
            for pair in account_pairs:

                # determine pairs debit and credit accounts
                pair_a_debit_credit = pair[0]
                pair_b_debit_credit = pair[1]

                # determine pairs debit and credit account features
                pair_a_debit_credit_name = posting_features[posting_features[statistics['je_gl_account_field']] == str(pair_a_debit_credit)]['Y_HKONT_TEXT']
                pair_b_debit_credit_name = posting_features[posting_features[statistics['je_gl_account_field']] == str(pair_b_debit_credit)]['Y_HKONT_TEXT']

                # add graph debit and credit edge
                partial_entries_graph.add_node(pair_a_debit_credit, account=pair_a_debit_credit_name)
                partial_entries_graph.add_node(pair_b_debit_credit, account=pair_b_debit_credit_name)

                # add graph debit and credit edge
                partial_entries_graph.add_edge(pair_a_debit_credit, pair_b_debit_credit)

            # save plot to plotting directory
            filename = '{}_entire_graph_{}.png'.format(str(parameter['exp_timestamp']), str(posting_id).zfill(5))
            self.vha.plot_entire_journal_entry_graph(parameter, entry_graph=partial_entries_graph, pos_nodes=pos_nodes, filename=filename)

    # prepare the feature and adjacency matrices
    def create_adj_feat_matrices(self, parameter, statistics, adjacencies, features, encoded_features):

        # determine unique posting ids
        posting_ids = adjacencies.groupby([statistics['je_identifier_field']]).count().index

        if parameter['mode'] in ['static']:

            ### prepare adjacency matrix and feature vector filling

            # init adjacency and feature matrices
            adj_matrices = np.zeros([len(posting_ids), statistics['no_posting_accounts'], statistics['no_posting_accounts']])
            feat_matrices = np.zeros([len(posting_ids), statistics['no_posting_accounts'], statistics['no_posting_features']])

            # iterate over distinct posting ids
            for i, posting_id in enumerate(posting_ids):

                # init posting adjacency matrix
                adj_matrix = np.zeros([statistics['no_posting_accounts'], statistics['no_posting_accounts']])

                # create journal entry adjacency matrix of current journal entry
                adj_matrix = self.fill_adjacency_matrix(je_identifier_field=statistics['je_identifier_field'], je_gl_account_code_field=statistics['je_gl_account_field'], je_debit_credit_field=statistics['je_debit_credit_field'], adj_matrix=adj_matrix, entries=adjacencies, posting_id=posting_id)

                # add identity matrix to account adjacency matrix
                adj_matrix += np.identity(statistics['no_posting_accounts'])

                # collect adjacency matrix -> nodes x nodes
                adj_matrices[i, :, :] = adj_matrix

                # init posting feature matrix
                feat_matrix = np.zeros([statistics['no_posting_accounts'], statistics['no_posting_features']])

                # create journal entry feature matrix of the SAP dataset
                feat_matrix = self.fill_feature_matrix(je_identifier_field=statistics['je_identifier_field'], je_gl_account_code_field=statistics['je_gl_account_field'], je_features_field=statistics['je_features'], feat_matrix=feat_matrix, entries=features, features=encoded_features, posting_id=posting_id)

                # collect feature matrix -> nodes x features
                feat_matrices[i, :, :] = feat_matrix

                # case: log adjacency matrix creation process
                if i % 1000 == 0:

                    # log configuration processing
                    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
                    print('[INFO {}] DataHandler :: {} adjacency and feature matrix: {} of: {} matrices created.'.format(now, str(statistics['dataset']).upper(), str(i), str(len(posting_ids))))

        elif parameter['mode'] in ['dynamic', 'baseline']:

            # init adjacency and feature matrices
            adj_matrices = []
            feat_matrices = []

            # iterate over distinct posting ids
            for i, posting_id in enumerate(posting_ids):

                # determine current posting line items
                posting_line_items = adjacencies[adjacencies[statistics['je_identifier_field']] == posting_id]

                # determine number of unique accounts
                no_posting_accounts = len(posting_line_items[statistics['je_gl_account_field']].unique())

                # init posting adjacency matrix
                adj_matrix = np.zeros([no_posting_accounts, no_posting_accounts])

                # create journal entry adjacency matrix of current journal entry
                adj_matrix = self.fill_adjacency_matrix_dynamic(je_identifier_field=statistics['je_identifier_field'], je_gl_account_code_field=statistics['je_gl_account_field'], je_debit_credit_field=statistics['je_debit_credit_field'], adj_matrix=adj_matrix, entries=adjacencies, posting_id=posting_id)

                # add identity matrix to account adjacency matrix ->
                adj_matrix += np.identity(no_posting_accounts)

                # collect adjacency matrix -> line-item nodes x line-item nodes
                adj_matrices.append(adj_matrix)

                # determine current posting line items
                posting_line_items = features[features[statistics['je_identifier_field']] == posting_id]

                # determine number of unique accounts
                no_posting_accounts = len(posting_line_items[statistics['je_gl_account_field']].unique())

                # init posting feature matrix
                feat_matrix = np.zeros([no_posting_accounts, statistics['no_posting_features']])

                # create journal entry feature matrix of the SAP dataset
                feat_matrix = self.fill_feature_matrix_dynamic(je_identifier_field=statistics['je_identifier_field'], je_gl_account_code_field=statistics['je_gl_account_field'], je_features_field=statistics['je_features'], feat_matrix=feat_matrix, entries=features, features=encoded_features, posting_id=posting_id)

                # collect feature matrix -> line-item nodes x features
                feat_matrices.append(feat_matrix)

            # case: log adjacency matrix creation process
                if i % 1000 == 0:

                    # log configuration processing
                    now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
                    print('[INFO {}] DataHandler :: {} adjacency and feature matrix: {} of: {} matrices created.'.format(now, str(statistics['dataset']).upper(), str(i), str(len(posting_ids))))

        # return adjacency and feature matrices
        return posting_ids, adj_matrices, feat_matrices, statistics

    # create single journal entry adjacency matrix of a given dataset
    def fill_adjacency_matrix(self, je_identifier_field, je_gl_account_code_field, je_debit_credit_field, adj_matrix, entries, posting_id):

        # determine current posting line items
        posting_line_items = entries[entries[je_identifier_field] == posting_id]

        # determine all possible account pairs
        account_pairs = list(it.combinations(posting_line_items[je_gl_account_code_field], 2))

        # iterate over distinct account pairs
        for pair in account_pairs:

            # determine pairs debit and credit structure
            pair_a_debit_credit = list(posting_line_items[posting_line_items[je_gl_account_code_field] == pair[0]][je_debit_credit_field])[0]
            pair_b_debit_credit = list(posting_line_items[posting_line_items[je_gl_account_code_field] == pair[1]][je_debit_credit_field])[0]

            # case: first account credit, second account debit
            if (pair_a_debit_credit == 'Credit') & (pair_b_debit_credit == 'Debit'):

                # fill adjacency matrix: credit -> debit
                adj_matrix[pair[0]][pair[1]] = 1

            # case: first account debit, second account credit
            elif (pair_a_debit_credit == 'Debit') & (pair_b_debit_credit == 'Credit'):

                # fill adjacency matrix: debit -> credit
                adj_matrix[pair[1]][pair[0]] = 1

            # case: first account similar to second
            else:

                # fill adjacency matrix
                adj_matrix[pair[1]][pair[0]] = 1
                adj_matrix[pair[0]][pair[1]] = 1

        # return adjacency matrix
        return adj_matrix

        # create single journal entry adjacency matrix of a given dataset
    def fill_adjacency_matrix_dynamic(self, je_identifier_field, je_gl_account_code_field, je_debit_credit_field, adj_matrix, entries, posting_id):

        # determine current posting line items
        posting_line_items = entries[entries[je_identifier_field] == posting_id]

        # encode posting line items accounts
        posting_line_items['je_gl_account_encoding'] = pd.Categorical(posting_line_items[je_gl_account_code_field]).codes

        # determine all possible account pairs
        account_pairs = list(it.combinations(posting_line_items['je_gl_account_encoding'], 2))

        # iterate over distinct account pairs
        for pair in account_pairs:

            # determine pairs debit and credit structure
            pair_a_debit_credit = list(posting_line_items[posting_line_items['je_gl_account_encoding'] == pair[0]][je_debit_credit_field])[0]
            pair_b_debit_credit = list(posting_line_items[posting_line_items['je_gl_account_encoding'] == pair[1]][je_debit_credit_field])[0]

            # case: first account credit, second account debit
            if (pair_a_debit_credit == 'Credit') & (pair_b_debit_credit == 'Debit'):

                # fill adjacency matrix: credit -> debit
                adj_matrix[pair[0]][pair[1]] = 1

            # case: first account debit, second account credit
            elif (pair_a_debit_credit == 'Debit') & (pair_b_debit_credit == 'Credit'):

                # fill adjacency matrix: debit -> credit
                adj_matrix[pair[1]][pair[0]] = 1

            # case: first account similar to second
            else:

                # fill adjacency matrix
                adj_matrix[pair[1]][pair[0]] = 1
                adj_matrix[pair[0]][pair[1]] = 1

        # return adjacency matrix
        return adj_matrix

    # create single journal entry adjacency matrix of the EY dataset
    def fill_feature_matrix(self, je_identifier_field, je_gl_account_code_field, je_features_field, feat_matrix, entries, features, posting_id):

        # determine current posting line items
        posting_line_items = entries[entries[je_identifier_field] == posting_id]

        # encode posting line items accounts
        posting_line_items['je_gl_account_encoding'] = pd.Categorical(posting_line_items[je_gl_account_code_field]).codes

        # determine current posting line items accounts
        posting_accounts = [int(account) for account in posting_line_items['je_gl_account_encoding'].values]

        # determine current posting features
        posting_features = features[features[je_identifier_field] == posting_id]

        # encode posting features accounts
        posting_features['je_gl_account_encoding'] = pd.Categorical(posting_line_items[je_gl_account_code_field]).codes

        # iterate over posting accounts
        for account in posting_accounts:

            # determine current posting account features
            posting_account_features = posting_features[posting_features['je_gl_account_encoding'] == account]

            # case: one-time account usage
            if posting_account_features.shape[0] == 1:

                # determine account features and convert to float
                features = [float(feat) for feat in posting_account_features[je_features_field].to_numpy()[0]]

                # fill posting features
                feat_matrix[account, :] = features

            # case: multiple-time account usage
            else:

                print('Hello World! - shouldn`t happen since we did aggregate before ...')
                print('... but can happen due to incomplete postins -> check data quality.')

        # return feature matrix
        return feat_matrix

        # create single journal entry adjacency matrix of the EY dataset
    def fill_feature_matrix_dynamic(self, je_identifier_field, je_gl_account_code_field, je_features_field, feat_matrix, entries, features, posting_id):

        # determine current posting line items
        posting_line_items = entries[entries[je_identifier_field] == posting_id]

        # encode posting line items accounts
        posting_line_items['je_gl_account_encoding'] = pd.Categorical(posting_line_items[je_gl_account_code_field]).codes

        # determine current posting line items accounts
        posting_accounts = [int(account) for account in posting_line_items['je_gl_account_encoding'].values]

        # determine current posting features
        posting_features = features[features[je_identifier_field] == posting_id]

        # encode posting features accounts
        posting_features['je_gl_account_encoding'] = pd.Categorical(posting_line_items[je_gl_account_code_field]).codes

        # iterate over posting accounts
        for account in posting_accounts:

            # determine current posting account features
            posting_account_features = posting_features[posting_features['je_gl_account_encoding'] == account]

            # case: one-time account usage
            if posting_account_features.shape[0] == 1:

                # determine account features and convert to float
                features = [float(feat) for feat in posting_account_features[je_features_field].to_numpy()[0]]

                # fill posting features
                feat_matrix[account, :] = features

            # case: multiple-time account usage
            else:

                print('Hello World! - shouldn`t happen since we did aggregate before ...')
                print('... but can happen due to incomplete postings -> check data quality.')

        # return feature matrix
        return feat_matrix

    # aggregate journal entries on a posting level
    def aggregate_entries_per_posting(self, statistics, entries):

        # aggregate header item attributes
        aggregated_entries = entries[[statistics['je_identifier_field'], statistics['je_line_item_field']]].groupby([statistics['je_identifier_field']]).count()

        # rename number of line items columns
        aggregated_entries.rename(columns={statistics['je_line_item_field']: 'Y_BUZEI'}, inplace=True)

        # create journal entry identifier column
        aggregated_entries[statistics['je_identifier_field']] = aggregated_entries.index

        # iterate over header attributes
        for header_attribute in statistics['je_header_attributes']:

            # aggregate header attributes
            aggregated_entries[header_attribute] = entries.groupby([statistics['je_identifier_field']])[header_attribute].first()

        # iterate over categorical line item attributes
        for line_attribute in statistics['je_segment_attributes_categorical']:

            # aggregate line item attributes
            aggregated_entries[line_attribute] = entries.groupby([statistics['je_identifier_field']])[line_attribute].apply(' :: '.join)

            # trim aggregated line item attributes
            aggregated_entries[line_attribute] = [ele[0:50] for ele in aggregated_entries[line_attribute]]

        # iterate over numerical line item attributes
        for line_attribute in statistics['je_segment_attributes_numerical']:

            # aggregate line item attributes
            summed_entries = entries.groupby([statistics['je_identifier_field'], 'Y_DEBIT_CREDIT'])[line_attribute].sum().reset_index()

            # trim aggregated line item attributes
            aggregated_entries[line_attribute] = summed_entries[summed_entries['Y_DEBIT_CREDIT'] == 'Debit'][line_attribute].values

        # return aggregated entries
        return aggregated_entries

    # aggregate entries on belnr, hkont, and shkzg level -> create adjacency matrices
    def aggregate_entries_per_belnr_hkont_shkzg(self, statistics, fields, entries):

        # aggregate journal entry line items
        aggregated_entries = entries[fields].groupby(fields).count()

        # reset the aggregation index
        aggregated_entries = aggregated_entries.reset_index()

        # reformat number of line items columns
        aggregated_entries[statistics['je_line_item_field']] = aggregated_entries[statistics['je_line_item_field']].astype(int)

        # rename number of line items columns
        aggregated_entries = aggregated_entries.rename(columns={statistics['je_line_item_field']: 'Y_BUZEI'}, inplace=False)

        # return aggregated entries
        return aggregated_entries

    # aggregate entries on belnr, and hkont level -> create feature matrices
    def aggregate_entries_per_belnr_hkont(self, statistics, fields, entries, labels):

        # aggregate journal entry line items
        aggregated_entries = entries[fields].groupby(fields).count()

        # reset the aggregation index
        aggregated_entries = aggregated_entries.reset_index()

        # rename number of line items columns
        aggregated_entries = aggregated_entries.rename(columns={statistics['je_line_item_field']: 'Y_BUZEI'}, inplace=False)

        # aggregate journal entry labels
        aggregated_labels = labels.groupby([statistics['je_identifier_field']] + [statistics['je_gl_account_field']] + [statistics['je_class_field']] + [statistics['je_class_name_field']]).count()

        # reset the aggregation index
        aggregated_labels = aggregated_labels.reset_index()

        # iterate over header attributes
        for header_feature in statistics['je_header_features']:

            # aggregate header attributes
            aggregated_entries[header_feature] = entries.groupby(fields)[header_feature].first().values

        # iterate over categorical line item attributes
        for segment_feature in statistics['je_segment_features_categorical']:

            # case: encoded gl account feature
            if segment_feature == statistics['je_gl_account_field']:

                # convert to categorical
                entries[segment_feature] = entries[segment_feature].astype(int)

                # aggregate line item attributes
                aggregated_entries[segment_feature] = entries.groupby(fields)[segment_feature].first().values

            # case; non-encoded gl account feature
            else:

                # convert to categorical
                entries[segment_feature] = entries[segment_feature].astype(str)

                # aggregate line item attributes
                aggregated_entries[segment_feature] = entries.groupby(fields)[segment_feature].apply(' :: '.join).values

                # trim aggregated line item attributes
                aggregated_entries[segment_feature] = [ele[0:50] for ele in aggregated_entries[segment_feature]]

        # iterate over numerical line item attributes
        for segment_feature in statistics['je_segment_features_numerical']:

            # convert to numerical
            entries[segment_feature] = entries[segment_feature].astype(float)

            # aggregate line item attributes
            summed_entries = entries.groupby(fields)[segment_feature].sum().reset_index()

            # trim aggregated line item attributes
            aggregated_entries[segment_feature] = summed_entries[segment_feature].values

        # add label information
        aggregated_entries[statistics['je_class_field']] = aggregated_labels[statistics['je_class_field']]
        aggregated_entries[statistics['je_class_name_field']] = aggregated_labels[statistics['je_class_name_field']]

        # return aggregated entries
        return aggregated_entries

    # aggregate entries on belnr, and hkont level -> create feature matrices
    def aggregate_entries_per_belnr(self, statistics, fields, entries, labels):

        # aggregate journal entry line items
        aggregated_entries = entries[fields + [statistics['je_line_item_field']]].groupby(fields).count()

        # reset the aggregation index
        aggregated_entries = aggregated_entries.reset_index()

        # rename number of line items columns
        aggregated_entries = aggregated_entries.rename(columns={statistics['je_line_item_field']: 'Y_BUZEI'}, inplace=False)

        # aggregate and collect journal entry accounts
        aggregated_entries['Y_NO_ACCOUNTS'] = entries[[statistics['je_identifier_field']] + [statistics['je_gl_account_field']]].groupby([statistics['je_identifier_field']]).nunique().values

        # aggregate journal entry labels
        aggregated_labels = labels.groupby([statistics['je_identifier_field']] + [statistics['je_class_field']] + [statistics['je_class_name_field']]).count()

        # reset the aggregation index
        aggregated_labels = aggregated_labels.reset_index()

        # iterate over header attributes
        for header_feature in statistics['je_header_features']:

            # aggregate header attributes
            aggregated_entries[header_feature] = entries.groupby(fields)[header_feature].first().values

        # iterate over categorical line item attributes
        for segment_feature in statistics['je_segment_features_categorical']:

            # convert to categorical
            entries[segment_feature] = entries[segment_feature].astype(str)

            # case: je line item attribute
            if segment_feature == statistics['je_identifier_field']:

                # aggregate line item attributes
                aggregated_entries[segment_feature] = entries.groupby(fields)[segment_feature].max().values

            # case: other attribute
            else:

                # aggregate line item attributes
                aggregated_entries[segment_feature] = entries.groupby(fields)[segment_feature].apply(' :: '.join).values

                # trim aggregated line item attributes
                aggregated_entries[segment_feature] = [ele[0:50] for ele in aggregated_entries[segment_feature]]

        # iterate over numerical line item attributes
        for segment_feature in statistics['je_segment_features_numerical']:

            # convert to numerical
            entries[segment_feature] = entries[segment_feature].astype(float)

            # aggregate line item attributes
            summed_entries = entries[entries[segment_feature] > 0.0].groupby(fields)[segment_feature].sum().reset_index()

            # trim aggregated line item attributes
            aggregated_entries[segment_feature] = summed_entries[segment_feature].values

        # add label information
        aggregated_entries[statistics['je_class_field']] = aggregated_labels[statistics['je_class_field']]
        aggregated_entries[statistics['je_class_name_field']] = aggregated_labels[statistics['je_class_name_field']]

        # return aggregated entries
        return aggregated_entries

    # encode the journal entry header features
    def encode_je_header_features(self, statistics, entries, type='embed'):

        # init encoded entries keys
        encoded_entries_keys = [statistics['je_identifier_field'], statistics['je_gl_account_field']]

        # init encoded entries
        encoded_entries = entries[encoded_entries_keys]

        # iterate over journal entry features
        for i, attribute in enumerate(statistics['je_header_features']):

            # case: one-hot encoding enabled
            if type == 'onehot':

                # determine one-hot encoding of current attribute
                encoded_feature = pd.get_dummies(entries[attribute])

                # collect encoded attribute dimensions
                statistics['{}_dim'.format(attribute)] = encoded_feature.shape[1]

            # case: embedding encoding enabled
            elif type == 'embed':

                # init attribute value encoder
                attribute_value_encoder = LabelEncoder()

                # convert attributes to string representation
                entries[attribute] = entries[attribute].astype(str)

                # clean whitspaces and fillna values
                entries[attribute] = entries[attribute].str.strip().fillna("-")

                # encode the categorical string representations
                encoded_feature = attribute_value_encoder.fit_transform(entries[attribute].values)

                # convert to pandas dataframe
                encoded_feature = pd.DataFrame(encoded_feature, columns=[attribute])

                # collect encoded attribute dimensions
                statistics['{}_dim'.format(attribute)] = len(attribute_value_encoder.classes_)

            # collect encoding of current attribute
            encoded_entries = pd.concat([encoded_entries, encoded_feature], axis=1)

        # return the encoded entries
        return encoded_entries, statistics

    # encoded the journal entry segment featurs
    def encode_je_segment_features(self, statistics, entries, type='embed'):

        # init encoded entries keys
        encoded_entries_keys = [statistics['je_identifier_field'], statistics['je_gl_account_field']]

        # init encoded entries
        encoded_entries = entries[encoded_entries_keys]

        # iterate over all features
        for i, feature in enumerate(statistics['je_segment_features']):

            # case: categorical feature
            if feature in statistics['je_segment_features_categorical']:

                # case: one-hot encoding enabled
                if type == 'onehot':

                    # determine one-hot encoding of current attribute
                    encoded_feature = pd.get_dummies(entries[feature])

                    # collect encoded attribute dimensions
                    statistics['{}_dim'.format(feature)] = encoded_feature.shape[1]

                    # case: embedding encoding enabled
                elif type == 'embed':

                    # init attribute value encoder
                    feature_value_encoder = LabelEncoder()

                    # convert attributes to string representation
                    entries[feature] = entries[feature].astype(str)

                    # clean whitspaces and fillna values
                    entries[feature] = entries[feature].str.strip().fillna("-")

                    # encode the categorical string representations
                    encoded_feature = feature_value_encoder.fit_transform(entries[feature].values)

                    # convert to pandas dataframe
                    encoded_feature = pd.DataFrame(encoded_feature, columns=[feature])

                    # collect encoded attribute dimensions
                    statistics['{}_dim'.format(feature)] = len(feature_value_encoder.classes_)

            # case numerical feature
            elif feature in statistics['je_segment_features_numerical']:

                # remove the amount sign information
                encoded_feature = np.abs(entries[feature])

                # log-transform the amount information
                encoded_feature = (encoded_feature + 1e-4).apply(np.log)

                # determine min and max of the transformed amount information
                numerical_min = encoded_feature.min()
                numerical_max = encoded_feature.max()

                # normalize the transaction amount information
                encoded_feature = (encoded_feature - numerical_min) / (numerical_max - numerical_min)

            # collect encoding of current attribute
            encoded_entries = pd.concat([encoded_entries, encoded_feature], axis=1)

        # return the encoded entries
        return encoded_entries, statistics

    # encode the gnn feature matrix
    def encode_gnn_features2(self, statistics, entries):

        # init encoded entries matrix
        encoded_entries = pd.DataFrame(0, index=np.arange(entries.shape[0]), columns=statistics['je_feature_fields'])

        # init number of encoded tokens
        statistics['token_no'] = 0

        # iterate over journal entry features
        for i, attribute in enumerate(statistics['je_feature_fields']):

            # init the label encoder
            label_encoder = LabelEncoder()

            # convert attributes to string representation
            entries[attribute] = entries[attribute].astype(str)

            # clean whitespaces and fillna values
            entries[attribute] = entries[attribute].str.strip().fillna("-")

            # encode the categorical string representations
            encoded_entries[attribute] = label_encoder.fit_transform(entries[attribute].values)

            # increase number of encoded tokens
            statistics['token_no'] += len(label_encoder.classes_)

        # add journal entry identifier to encoded journal entry attributes
        encoded_entries.insert(loc=0, column=statistics['je_identifier_field'], value=entries[statistics['je_identifier_field']])

        # remove duplicate one-hot encoded categorical attributes
        encoded_entries = encoded_entries.drop_duplicates()

        # return the encoded entries
        return encoded_entries, statistics

    # define customized collate function
    def collate_batch(self, batch):

        # init adj matrix and feat matrix lists
        adj_matrices = []
        feat_matrices = []

        # iterate batch adj and feat matrix
        for (adj_matrix, feat_matrix) in batch:

            # convert to pytorch tensors
            adj_matrix_tensor = torch.tensor(adj_matrix, dtype=torch.float64)
            feat_matrix_tensor = torch.tensor(feat_matrix, dtype=torch.float64)

            # append to batch list
            adj_matrices.append(adj_matrix_tensor)
            feat_matrices.append(feat_matrix_tensor)

        # return adj and feature matrices
        return adj_matrices, feat_matrices