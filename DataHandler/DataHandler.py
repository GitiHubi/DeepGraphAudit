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
import datetime as dt
import numpy as np
import pandas as pd
import itertools as it

# import sklearn
from sklearn.preprocessing import LabelEncoder

# class DataHandler
class DataHandler(object):

    def __init__(self):

        pass

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

    def get_gnn_data_range_ey(self, parameter, statistics):

        # case: sample evaluation enabled
        if parameter['sample_eval'] is True and parameter['dataset'].split('/')[-1] == 'ey':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '01_ernstyoung', 'Analytics_mindset_case_studies_PSU_Hotel_data_v1.csv'), sep=',', encoding='utf-8').iloc[0:parameter['sample_size'], :]

        # case: sample evaluation not enbaled
        elif parameter['sample_eval'] is False and parameter['dataset'].split('/')[-1] == 'ey':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '01_ernstyoung', 'Analytics_mindset_case_studies_PSU_Hotel_data_v1.csv'), sep=',', encoding='utf-8')

        # log configuration processing
        now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
        print('[INFO {}] DataHandler :: transactional EY data of shape {} rows and {} columns successfully loaded.'.format(now, str(original_entries.shape[0]), str(original_entries.shape[1])))

        ### Step 0: Define Data Attributes and Statistcs ############################################################

        # collect dataset statistics
        statistics['dataset'] = parameter['dataset']

        # set dataset parameter
        statistics['je_identifier_field'] = 'Y_JE_IDENTIFIER'
        statistics['je_line_item_field'] = 'Y_JE_ITEM_IDENTIFIER'
        statistics['je_gl_account_field'] = 'Y_GL_ACCOUNT_NUMBER'
        statistics['je_debit_credit_field'] = 'Y_DEBIT_CREDIT'

        statistics['je_header_attributes'] = ['Y_PREPARER_ID', 'Y_SOURCE']
        statistics['je_segment_attributes_categorical'] = ['Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME']
        statistics['je_segment_attributes_numerical'] = ['Y_DMBTR']

        statistics['je_header_features'] = ['Y_PREPARER_ID', 'Y_SOURCE']
        statistics['je_segment_features_numerical'] = ['Y_DMBTR']

        statistics['hover_attributes'] = ['Y_JE_IDENTIFIER', 'Y_BUZEI', 'Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME', 'Y_PREPARER_ID', 'Y_SOURCE', 'Y_DMBTR']

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
        statistics['cat_attr'] = cat_attr
        statistics['num_attr'] = num_attr

        ### Step 1: Pre-process journal entries attribute values #####################################################

        # processed journal entry attributes
        processed_detailed_entries = pd.DataFrame({statistics['je_identifier_field'] : original_entries['JEIdentifier'].astype(str)})

        # pre-process the categorical journal entry attributes
        processed_detailed_cat_entries, _ = self.preprocess_ey_categorical_attributes(entries=original_entries, categorical_attributes=cat_attr)

        # pre-process the numerical journal entry attributes
        processed_detailed_num_entries, _ = self.preprocess_ey_numerical_attributes(entries=original_entries, numerical_attributes=num_attr)

        # combine pre-processed categorical and numerical attributes
        processed_detailed_entries = pd.concat([processed_detailed_entries, processed_detailed_cat_entries, processed_detailed_num_entries], axis=1)

        ### Step 2: Aggregate journal entry attribute values #########################################################

        # aggregate the pre-processed categorical transaction attributes
        aggregated_entries = self.aggregate_gnn_attributes(statistics=statistics, entries=processed_detailed_entries)

        ### Step 3: filter graph exhibiting journal entries ####################################################

        # remove non graph exhibiting journal entries - single line item postings and batch process postings
        selected_aggregated_entries = aggregated_entries[(aggregated_entries['Y_BUZEI'] >= parameter['min_line_items']) & (aggregated_entries['Y_BUZEI'] <= parameter['max_line_items'])]

        # remove non graph exhibiting journal entries - single line item postings and batch process postings
        selected_detailed_entries = processed_detailed_entries[processed_detailed_entries[statistics['je_identifier_field']].isin(selected_aggregated_entries[statistics['je_identifier_field']])]

        ### Step 4: encode journal entry attribute values ####################################################

        # encode the pre-processed categorical transaction attributes
        posting_ids, adj_matrices, feat_matrices, statistics = self.encode_gnn_attributes(statistics=statistics, entries=selected_detailed_entries)

        # return original and encoded transactions
        return posting_ids, adj_matrices, feat_matrices, aggregated_entries, selected_aggregated_entries, statistics

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
        print('[INFO {}] DataHandler :: transactional {} data of shape {} rows and {} columns successfully loaded.'.format(now, str(parameter['dataset']), str(original_entries.shape[0]), str(original_entries.shape[1])))

        ### Step 0: Define Data Attributes and Statistcs ############################################################

        # collect dataset statistics
        statistics['dataset'] = parameter['dataset']

        # set dataset parameter
        statistics['je_identifier_field'] = 'Y_BELNR'
        statistics['je_line_item_field'] = 'Y_BUZEI'
        statistics['je_gl_account_field'] = 'Y_HKONT'
        statistics['je_debit_credit_field'] = 'Y_DEBIT_CREDIT'

        statistics['je_header_attributes'] = ['Y_BLART', 'Y_BLART_TEXT', 'Y_USNAM', 'Y_TCODE']
        statistics['je_segment_attributes_categorical'] = ['Y_HKONT', 'Y_HKONT_TEXT']
        statistics['je_segment_attributes_numerical'] = ['Y_DMBTR']

        statistics['je_header_features'] = ['Y_BLART', 'Y_BLART_TEXT', 'Y_USNAM', 'Y_TCODE']
        statistics['je_segment_features_numerical'] = ['Y_DMBTR']

        statistics['hover_attributes'] = ['Y_JE_IDENTIFIER', 'Y_BUZEI', 'Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME', 'Y_PREPARER_ID', 'Y_SOURCE', 'Y_DMBTR']

        # determine the categorical attributes
        cat_attr = [
            'DocumentNr'  # The unique identifier of the journal entry.
            , 'DocType'  # The document type of the journal entry.7
            , 'DocTypeDescr'  # The document type description of the journal entry.
            , 'GL Accountnr'  # The number of the general ledger account.
            , 'GL AccountDescr'  # The name of the general ledger account.
            , 'UserName Post'  # The id of the entry preparer.
            , 'TransactionDescription'  # The transaction code of the journal entry.
        ]

        # determine the numerical attributes
        num_attr = ['AmountinUSD'  # The amount of the journal entry (USD$).
                    ]

        # collect categorical and numerical attributes
        statistics['cat_attr'] = cat_attr
        statistics['num_attr'] = num_attr

        ### Step 1: Pre-process journal entries attribute values #####################################################

        # processed journal entry attributes
        processed_detailed_entries = pd.DataFrame({statistics['je_identifier_field'] : original_entries['DocumentNr'].astype(str)})

        # pre-process the categorical journal entry attributes
        processed_detailed_cat_entries, _ = self.preprocess_sap_categorical_attributes(entries=original_entries, categorical_attributes=cat_attr)

        # pre-process the numerical journal entry attributes
        processed_detailed_num_entries, _ = self.preprocess_sap_numerical_attributes(entries=original_entries, numerical_attributes=num_attr)

        # combine pre-processed categorical and numerical attributes
        processed_detailed_entries = pd.concat([processed_detailed_entries, processed_detailed_cat_entries, processed_detailed_num_entries], axis=1)

        ### Step 2: Aggregate journal entry attribute values #########################################################

        # aggregate the pre-processed categorical transaction attributes
        aggregated_entries = self.aggregate_gnn_attributes(statistics=statistics, entries=processed_detailed_entries)

        ### Step 3: filter graph exhibiting journal entries ####################################################

        # remove non graph exhibiting journal entries - single line item postings and batch process postings
        selected_aggregated_entries = aggregated_entries[(aggregated_entries['Y_BUZEI'] >= parameter['min_line_items']) & (aggregated_entries['Y_BUZEI'] <= parameter['max_line_items'])]

        # remove non graph exhibiting journal entries - single line item postings and batch process postings
        selected_detailed_entries = processed_detailed_entries[processed_detailed_entries[statistics['je_identifier_field']].isin(selected_aggregated_entries[statistics['je_identifier_field']])]

        ### Step 4: encode journal entry attribute values ####################################################

        # encode the pre-processed categorical transaction attributes
        posting_ids, adj_matrices, feat_matrices, statistics = self.encode_gnn_attributes(statistics=statistics, entries=selected_detailed_entries)

        # return original and encoded transactions
        return posting_ids, adj_matrices, feat_matrices, aggregated_entries, selected_aggregated_entries, statistics

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
    def preprocess_ey_categorical_attributes(self, entries, categorical_attributes):

        # extract the categorical attributes
        cat_entries = entries[categorical_attributes].copy()

        # convert categorical attributes to string
        cat_entries['Y_ACCOUNT_TYPE'] = cat_entries['AccountType'].astype(str)
        cat_entries['Y_ACCOUNT_CLASS'] = cat_entries['AccountClass'].astype(str)
        cat_entries['Y_GL_ACCOUNT_NUMBER'] = cat_entries['GLAccountNumber'].astype(str)
        cat_entries['Y_GL_ACCOUNT_NAME'] = cat_entries['GLAccountName'].astype(str)
        cat_entries['Y_PREPARER_ID'] = cat_entries['PreparerID'].astype(str)
        cat_entries['Y_SOURCE'] = cat_entries['Source'].astype(str)

        # clean JE creator field
        cat_entries['Y_PREPARER_ID'] = [ele.replace(u'\xa0', u' ') for ele in cat_entries['Y_PREPARER_ID']]

        # create JE line item identifier
        cat_entries['Y_JE_ITEM_IDENTIFIER'] = cat_entries.groupby(['JEIdentifier']).cumcount()

        # create JE debit and credit identifier
        cat_entries['Y_DEBIT_CREDIT'] = 0
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where((entries['Debit'].notna()) & (entries['Credit'].isnull()), 'Credit')
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where((entries['Credit'].notna()) & (entries['Debit'].isnull()), 'Debit')

        # set the pre-processed categorical attributes
        created_cat_attributes = ['Y_JE_ITEM_IDENTIFIER', 'Y_PREPARER_ID', 'Y_SOURCE', 'Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME', 'Y_DEBIT_CREDIT']

        # iterate over distinct categorical attributes
        for attribute in created_cat_attributes:

            # determine distinct attribute values per categorical attribute
            attr_values = cat_entries[attribute].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: EY categorical attribute: {}, no. of. distinct values: {}.'.format(now, str(attribute), str(attr_values)))

        # return pre-processed numerical transaction attributes
        return cat_entries[created_cat_attributes], created_cat_attributes

    # pre-process numerical attributes of the EY dataset
    def preprocess_ey_numerical_attributes(self, entries, numerical_attributes):

        # extract the numerical attributes
        num_entries = entries[numerical_attributes].copy()

        # convert categorical attributes to string
        num_entries['Y_DMBTR'] = [float(ele.replace(',','')) for ele in num_entries['Amount']]

        # set the pre-processed categorical attributes
        created_num_attributes = ['Y_DMBTR']

        # iterate over distinct categorical attributes
        for attribute in created_num_attributes:

            # determine distinct attribute values per categorical attribute
            attr_values = num_entries[attribute].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: EY numerical attribute: {}, no. of. distinct values: {}.'.format(now, str(attribute), str(attr_values)))

        # return pre-processed numerical transaction attributes
        return num_entries[created_num_attributes], created_num_attributes

    # pre-process categorical attributes of the SAP dataset
    def preprocess_sap_categorical_attributes(self, entries, categorical_attributes):

        # extract the categorical attributes
        cat_entries = entries[categorical_attributes].copy()

        # convert categorical attributes to string
        cat_entries['Y_BLART'] = cat_entries['DocType'].astype(str)
        cat_entries['Y_BLART_TEXT'] = cat_entries['DocTypeDescr'].astype(str)
        cat_entries['Y_HKONT'] = cat_entries['GL Accountnr'].astype(str)
        cat_entries['Y_HKONT_TEXT'] = cat_entries['GL AccountDescr'].astype(str)
        cat_entries['Y_USNAM'] = cat_entries['UserName Post'].astype(str)
        cat_entries['Y_TCODE'] = cat_entries['TransactionDescription'].astype(str)

        # create JE line item identifier
        cat_entries['Y_BUZEI'] = cat_entries.groupby(['DocumentNr']).cumcount()

        # create JE debit and credit identifier
        cat_entries['Y_DEBIT_CREDIT'] = 0
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where(entries['D/C'] == 'H', 'Credit')
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where(entries['D/C'] == 'S', 'Debit')

        # set the pre-processed categorical attributes
        created_cat_attributes = ['Y_BUZEI', 'Y_BLART', 'Y_BLART_TEXT', 'Y_HKONT', 'Y_HKONT_TEXT', 'Y_USNAM', 'Y_TCODE', 'Y_DEBIT_CREDIT']

        # iterate over distinct categorical attributes
        for attribute in created_cat_attributes:

            # determine distinct attribute values per categorical attribute
            attr_values = cat_entries[attribute].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: SAP categorical attribute: {}, no. of. distinct values: {}.'.format(now, str(attribute), str(attr_values)))

        # return pre-processed numerical transaction attributes
        return cat_entries[created_cat_attributes], created_cat_attributes

    # pre-process numerical attributes of the SAP dataset
    def preprocess_sap_numerical_attributes(self, entries, numerical_attributes):

        # extract the numerical attributes
        num_entries = entries[numerical_attributes].copy()

        # convert categorical attributes to string
        num_entries['Y_DMBTR'] = [float(ele) for ele in num_entries['AmountinUSD']]

        # set the pre-processed categorical attributes
        created_num_attributes = ['Y_DMBTR']

        # iterate over distinct categorical attributes
        for attribute in created_num_attributes:

            # determine distinct attribute values per categorical attribute
            attr_values = num_entries[attribute].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: SAP numerical attribute: {}, no. of. distinct values: {}.'.format(now, str(attribute), str(attr_values)))

        # return pre-processed numerical transaction attributes
        return num_entries[created_num_attributes], created_num_attributes

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
    def encode_gnn_attributes(self, statistics, entries):

        # determine unique posting ids
        posting_ids = entries.groupby([statistics['je_identifier_field']]).count().index

        ### prepare adjacency matrix fields

        # determine unique posted accounts
        no_posting_accounts = len(entries[statistics['je_gl_account_field']].unique())

        # encode posting general ledger
        entries['{}_CODE'.format(str(statistics['je_gl_account_field']))] = pd.Categorical(entries[statistics['je_gl_account_field']]).codes

        ### prepare feature vector fields

        # encode journal entry header features
        feat_header, statistics = self.encode_je_header_features(statistics, entries)

        # encoder journal entry segment features
        feat_segment, statistics = self.encode_je_segment_features(statistics, entries)

        # combine journal entry header and segment features
        features = pd.concat([feat_header, feat_segment], axis=1)

        # drop duplicate identifier column
        features = features.loc[:,~features.columns.duplicated()]

        # init adjacency and feature matrices
        adj_matrices = np.zeros([len(posting_ids), no_posting_accounts, no_posting_accounts])
        feat_matrices = np.zeros([len(posting_ids), no_posting_accounts, features.shape[1] - 1])

        # iterate over distinct posting ids
        for i, posting_id in enumerate(posting_ids):

            # init posting adjacency matrix
            adj_matrix = np.zeros([no_posting_accounts, no_posting_accounts])

            # create journal entry adjacency matrix of current journal entry
            adj_matrix = self.fill_adjacency_matrix(je_identifier_field=statistics['je_identifier_field'], je_gl_account_code_field='{}_CODE'.format(str(statistics['je_gl_account_field'])), je_debit_credit_field=statistics['je_debit_credit_field'], adj_matrix=adj_matrix, entries=entries, posting_id=posting_id)

            # add identity matrix to account adjacency matrix
            adj_matrix += np.identity(no_posting_accounts)

            # collect adjacency matrix -> nodes x nodes
            adj_matrices[i, :, :] = adj_matrix

            # init posting feature matrix
            feat_matrix = np.zeros([no_posting_accounts, features.shape[1] - 1])

            # create journal entry feature matrix of the SAP dataset
            feat_matrix = self.fill_feature_matrix(je_identifier_field=statistics['je_identifier_field'], je_gl_account_code_field='{}_CODE'.format(str(statistics['je_gl_account_field'])), feat_matrix=feat_matrix, entries=entries, features=features, posting_id=posting_id)

            # collect feature matrix -> nodes x features
            feat_matrices[i, :, :] = feat_matrix

            # case: log adjacency matrix creation process
            if i % 1000 == 0:

                # log configuration processing
                now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
                print('[INFO {}] DataHandler :: {} adjacency and feature matrix: {} of: {} matrices created.'.format(now, str(statistics['dataset']).upper(), str(i), str(len(posting_ids))))

        # return adjacency
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

    # create single journal entry adjacency matrix of the EY dataset
    def fill_feature_matrix(self, je_identifier_field, je_gl_account_code_field, feat_matrix, entries, features, posting_id):

        # determine current posting line items
        posting_line_items = entries[entries[je_identifier_field] == posting_id]

        # determine current posting line items accounts
        posting_accounts = posting_line_items[je_gl_account_code_field].values

        # determine current posting features
        posting_features = features[features[je_identifier_field] == posting_id]

        # iterate over posting accounts
        for account in posting_accounts:

            # fill posting features
            feat_matrix[account, :] = posting_features.to_numpy()[0][1:]

        # return feature matrix
        return feat_matrix

    # aggregate encoded categorical attributes of the EY dataset
    def aggregate_gnn_attributes(self, statistics, entries):

        # aggregate header item attributes
        aggregated_entries = entries[[statistics['je_identifier_field'], statistics['je_line_item_field']]].groupby([statistics['je_identifier_field']]).count()

        # todo: determine number of accounts per journal entry posting
        # entries[[je_identifier_field, 'Y_GL_ACCOUNT_NUMBER']].groupby([je_identifier_field]).count()

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
            aggregated_entries[line_attribute] = [ele[0:60] for ele in aggregated_entries[line_attribute]]

        # iterate over numerical line item attributes
        for line_attribute in statistics['je_segment_attributes_numerical']:

            # aggregate line item attributes
            summed_entries = entries.groupby([statistics['je_identifier_field'], 'Y_DEBIT_CREDIT'])[line_attribute].sum().reset_index()

            # trim aggregated line item attributes
            aggregated_entries[line_attribute] = summed_entries[summed_entries['Y_DEBIT_CREDIT'] == 'Debit'][line_attribute].values

        # return aggregated entries
        return aggregated_entries

    # encode the journal entry header features
    def encode_je_header_features(self, statistics, entries):

        # iterate over journal entry features
        for i, attribute in enumerate(statistics['je_header_features']):

            # determine one-hot encoding of current attribute
            encoded_attribute = pd.get_dummies(entries[attribute])

            # case: initial attribute
            if i == 0:

                # collect one-hot encoding of current attribute
                encoded_entries = pd.DataFrame(encoded_attribute, columns=encoded_attribute.columns)

            # case: non-initial attribute
            else:

                # collect one-hot encoding of current attribute
                encoded_entries = pd.concat([encoded_entries, encoded_attribute], axis=1)

        # add journal entry identifier to encoded journal entry attributes
        encoded_entries.insert(loc=0, column=statistics['je_identifier_field'], value=entries[statistics['je_identifier_field']])

        # remove duplicate one-hot encoded categorical attributes
        encoded_entries = encoded_entries.drop_duplicates()

        # return the encoded entries
        return encoded_entries, statistics

    # encoded the journal entry segment featurs
    def encode_je_segment_features(self, statistics, entries):

        # iterate over journal entry features
        for i, attribute in enumerate(statistics['je_segment_features_numerical']):

            # remove the amount sign information
            encoded_attribute = np.abs(entries[attribute])

            # log-transform the amount information
            encoded_attribute = (encoded_attribute + 1e-4).apply(np.log)

            # determine min and max of the transformed amount information
            numerical_min = encoded_attribute.min()
            numerical_max = encoded_attribute.max()

            # normalize the transaction amount information
            encoded_attribute = (encoded_attribute - numerical_min) / (numerical_max - numerical_min)

            # case: initial attribute
            if i == 0:

                # collect one-hot encoding of current attribute
                encoded_entries = pd.DataFrame(encoded_attribute, columns=[attribute])

            # case: non-initial attribute
            else:

                # collect one-hot encoding of current attribute
                encoded_entries = pd.concat([encoded_entries, encoded_attribute], axis=1)

        # add journal entry identifier to encoded journal entry attributes
        encoded_entries.insert(loc=0, column=statistics['je_identifier_field'], value=entries[statistics['je_identifier_field']])

        # remove duplicate one-hot encoded categorical attributes
        encoded_entries = encoded_entries.drop_duplicates()

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