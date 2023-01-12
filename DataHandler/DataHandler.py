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

    def get_ernstyoung_gnn_data_range(self, parameter):

        # case: sample evaluation enabled
        if parameter['sample_eval'] is True and parameter['dataset'].split('/')[-1] == 'ernstyoung':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '01_ernstyoung', 'Analytics_mindset_case_studies_PSU_Hotel_data_v1.csv'), sep=',', encoding='utf-8').iloc[0:parameter['sample_size'], :]

        # case: sample evaluation not enbaled
        elif parameter['sample_eval'] is False and parameter['dataset'].split('/')[-1] == 'ernstyoung':

            # read the transactional data
            original_entries = pd.read_csv(os.path.join(parameter['data_dir'], '01_ernstyoung', 'Analytics_mindset_case_studies_PSU_Hotel_data_v1.csv'), sep=',', encoding='utf-8')

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

        ### Step 2: encode journal entry attribute values ####################################################

        # encode the pre-processed categorical transaction attributes
        posting_ids, adj_matrices, feat_matrices, entries_statistics = self.encode_ernstyoung_gnn_categorical_attributes(entries=processed_categorical_entries, categorical_attributes=created_cat_attributes, encoded_attributes=['Y_GL_ACCOUNT_NUMBER'], entries_statistics=entries_statistics)

        ### Step 3: Aggregate journal entry attribute values #########################################################

        # aggregate the pre-processed categorical transaction attributes
        aggregated_entries = self.aggregate_ernstyoung_gnn_categorical_attributes(entries=processed_categorical_entries, header_attributes=['Y_JE_IDENTIFIER'], line_attributes=['Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME', 'Y_DEBIT_CREDIT'])

        # return original and encoded transactions
        return posting_ids, adj_matrices, feat_matrices, aggregated_entries, entries_statistics

    # pre-process categorical attributes of the EY dataset
    def preprocess_ernstyoung_categorical_attributes(self, entries, categorical_attributes):

        # extract the categorical attributes
        cat_entries = entries[categorical_attributes].copy()

        # convert categorical attributes to string
        cat_entries['Y_JE_IDENTIFIER'] = cat_entries['JEIdentifier'].astype(str)
        cat_entries['Y_ACCOUNT_TYPE'] = cat_entries['AccountType'].astype(str)
        cat_entries['Y_ACCOUNT_CLASS'] = cat_entries['AccountClass'].astype(str)
        cat_entries['Y_GL_ACCOUNT_NUMBER'] = cat_entries['GLAccountNumber'].astype(str)
        cat_entries['Y_GL_ACCOUNT_NAME'] = cat_entries['GLAccountName'].astype(str)
        cat_entries['Y_PREPARER_ID'] = cat_entries['PreparerID'].astype(str)
        cat_entries['Y_SOURCE'] = cat_entries['Source'].astype(str)

        # create JE line item identifier
        cat_entries['Y_JE_ITEM_IDENTIFIER'] = cat_entries.groupby(['Y_JE_IDENTIFIER']).cumcount()

        # create JE debit and credit identifier
        cat_entries['Y_DEBIT_CREDIT'] = 0
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where((entries['Debit'].notna()) & (entries['Credit'].isnull()), 'Credit')
        cat_entries['Y_DEBIT_CREDIT'] = cat_entries['Y_DEBIT_CREDIT'].where((entries['Credit'].notna()) & (entries['Debit'].isnull()), 'Debit')

        # set the pre-processed categorical attributes
        created_cat_attributes = ['Y_JE_IDENTIFIER', 'Y_JE_ITEM_IDENTIFIER', 'Y_PREPARER_ID', 'Y_SOURCE', 'Y_ACCOUNT_TYPE', 'Y_ACCOUNT_CLASS', 'Y_GL_ACCOUNT_NUMBER', 'Y_GL_ACCOUNT_NAME', 'Y_DEBIT_CREDIT']

        # iterate over distinct categorical attributes
        for attribute in created_cat_attributes:

            # determine distinct attribute values per categorical attribute
            attr_values = cat_entries[attribute].value_counts().shape[0]

            # log configuration processing
            now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
            print('[INFO {}] DataHandler :: EY categorical attribute: {}, no. of. distinct values: {}.'.format(now, str(attribute), str(attr_values)))

        # return pre-processed numerical transaction attributes
        return cat_entries[created_cat_attributes], created_cat_attributes

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

        # determine unique posting ids
        # posting_ids = encoded_cat_entries['Y_JE_IDENTIFIER'].unique()
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

            # init posting adjacency matrix
            adj_matrix = np.zeros([len(posting_accounts), len(posting_accounts)])

            # determine current posting line items
            posting_line_items = encoded_cat_entries[encoded_cat_entries['Y_JE_IDENTIFIER'] == posting_id]

            # determine all possible account pairs
            account_pairs = list(it.combinations(posting_line_items['Y_GL_ACCOUNT_CODE'], 2))

            # iterate over distinct account pairs
            for pair in account_pairs:

                # determine pairs debit and credit structure
                pair_a_debit_credit = list(posting_line_items[posting_line_items['Y_GL_ACCOUNT_CODE'] == pair[0]]['Y_DEBIT_CREDIT'])[0]
                pair_b_debit_credit = list(posting_line_items[posting_line_items['Y_GL_ACCOUNT_CODE'] == pair[1]]['Y_DEBIT_CREDIT'])[0]

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

                # add identity matrix
                adj_matrix += np.identity(len(posting_accounts))

                # fill feature matrix
                feat_matrix = np.identity(len(posting_accounts))

            # collect adjacency matrix
            adj_matrices.append(adj_matrix)

            # collect adjacency matrix
            feat_matrices.append(feat_matrix)

            # case: log adjacency matrix creation process
            if i % 1000 == 0:

                # log configuration processing
                now = dt.datetime.utcnow().strftime('%Y.%m.%d-%H:%M:%S')
                print('[INFO {}] DataHandler :: EY adjacency matrix: {} of: {} matrices created.'.format(now, str(i), str(len(posting_ids))))

        # return adjacency
        return posting_ids, adj_matrices, feat_matrices, entries_statistics

    # aggregate encoded categorical attributes of the EY dataset
    def aggregate_ernstyoung_categorical_attributes(self, entries, header_attributes, line_attributes):

        # remove line item attributes
        aggregated_cat_entries = entries.drop(line_attributes, axis=1)

        # aggregate header item attributes
        aggregated_cat_entries = aggregated_cat_entries.groupby(header_attributes).sum()

        # reset multi-index to retrieve columns
        aggregated_cat_entries = aggregated_cat_entries.reset_index()

        # return aggregated entries
        return aggregated_cat_entries

    # aggregate encoded categorical attributes of the EY dataset
    def aggregate_ernstyoung_gnn_categorical_attributes(self, entries, header_attributes, line_attributes):

        # aggregate header item attributes
        aggregated_cat_entries = entries[['Y_JE_IDENTIFIER', 'Y_JE_ITEM_IDENTIFIER']].groupby(['Y_JE_IDENTIFIER']).count()

        # rename number of line items columns
        aggregated_cat_entries.rename(columns={'Y_JE_ITEM_IDENTIFIER': 'Y_JE_LINE_ITEMS'}, inplace=True)

        # aggregate general ledger account type
        aggregated_cat_entries['Y_PREPARER_ID'] = entries.groupby(['Y_JE_IDENTIFIER'])['Y_PREPARER_ID'].apply(' :: '.join)

        # aggregate general ledger account type
        aggregated_cat_entries['Y_SOURCE'] = entries.groupby(['Y_JE_IDENTIFIER'])['Y_SOURCE'].apply(' :: '.join)

        # aggregate general ledger account class
        aggregated_cat_entries['Y_ACCOUNT_CLASS'] = entries.groupby(['Y_JE_IDENTIFIER'])['Y_ACCOUNT_CLASS'].apply(' :: '.join)

        # aggregate general ledger account name
        aggregated_cat_entries['Y_GL_ACCOUNT_NAME'] = entries.groupby(['Y_JE_IDENTIFIER'])['Y_GL_ACCOUNT_NAME'].apply(' :: '.join)

        # aggregate general ledger account type
        aggregated_cat_entries['Y_ACCOUNT_TYPE'] = entries.groupby(['Y_JE_IDENTIFIER'])['Y_ACCOUNT_TYPE'].apply(' :: '.join)

        # return aggregated entries
        return aggregated_cat_entries

