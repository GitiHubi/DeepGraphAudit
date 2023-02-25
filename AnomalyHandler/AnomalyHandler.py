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
from scipy import stats
import pandas as pd
import random as rd

# class for generation of artificial anomalies
class AnomalyHandler(object):

    # init anomaly handler
    def __init__(self):

        pass

    # create global ey graph anomalies
    def generate_global_graph_anomalies_ey(self, statistics, entries, top=20, n=10, seed=1111):

        # aggregate journal entry feature information per entry
        entries_aggregated = self.aggregate_entries_per_belnr(statistics, fields=[statistics['je_identifier_field']] + statistics['je_header_features'], entries=entries)

        # determine journal entry feature information count
        entries_aggregated_count = entries_aggregated.groupby(statistics['je_header_features'] + statistics['je_segment_features_categorical']).count().reset_index()

        # filter top-n most occurring journal entry feature combinations
        entries_aggregated_top_feature_combinations = entries_aggregated_count.sort_values(by=statistics['je_identifier_field'], ascending=False).iloc[0:top]

        # filter top-n most occurring journal entry feature combinations
        entries_aggregated_top_feature_combinations = entries_aggregated_top_feature_combinations[statistics['je_header_features'] + statistics['je_segment_features_categorical']]

        # determine random top-n occurring journal entry feature combination
        single_random_top_feature_combination = dict(entries_aggregated_top_feature_combinations.iloc[np.random.choice(list(range(0, top)))])

        # determine aggregated entries of random top-n journal entry feature combination
        entries_aggregated_single_random_top_feature_combination = entries_aggregated[entries_aggregated[single_random_top_feature_combination.keys()].isin(single_random_top_feature_combination.values()).all(axis=1)]

        # select entry ids of random top-n journal entry feature combination
        entries_aggregated_single_random_top_feature_combination_ids = entries_aggregated_single_random_top_feature_combination[statistics['je_identifier_field']]

        # select random entry id of random top-n journal entry feature combination
        entries_aggregated_single_random_top_feature_combination_id = entries_aggregated_single_random_top_feature_combination_ids.iloc[np.random.choice(list(range(0, len(entries_aggregated_single_random_top_feature_combination_ids))))]

        # determine single random entry of random top-n journal entry feature combination
        single_random_top_feature_combination_entry = entries[entries[statistics['je_identifier_field']] == entries_aggregated_single_random_top_feature_combination_id]

        # create random gl accounts
        single_random_top_feature_combination_entry[statistics['je_gl_account_field']] = [rd.randint(9000, 9999) for ele in single_random_top_feature_combination_entry[statistics['je_gl_account_field']]]


        def get_sub_obj():
            return str(np.random.randint(low=800, high=999))

        def get_document_no():
            return str('PHIL') + str(np.random.randint(low=0, high=99999999)).zfill(8)

        def get_dept():
            return np.random.randint(low=75, high=99)

        def get_char():
            return np.random.choice([1, 8, 9])

        def get_fm():
            return np.random.choice([13, 14, 15, 16])

        def get_vendor_name():
            return self.fake.company() + ' ' + self.fake.company_suffix()

        def get_doc_ref_no_prefix():
            return ''.join([self.fake.random_uppercase_letter() for _ in range(4)])

        def get_contract_number():
            return ''.join([self.fake.random_uppercase_letter() for _ in range(2)]) + str(np.random.randint(low=1, high=9999)).zfill(4)

        def get_amount():
            return np.random.choice([np.random.randint(low=-100000, high=-5000) - np.random.random(), np.random.randint(low=1000000, high=99999999) + np.random.random()])

        # init random seed
        np.random.seed(self.seed)

        # init global anomalies
        global_anomalies = []

        # iterate over number of samples
        for i in range(n):

            # sample single global anomaly
            sample = {'sub_obj': get_sub_obj(),
                      'document_no': get_document_no(),
                      'dept': get_dept(),
                      'char_': get_char(),
                      'fm': get_fm(),
                      'vendor_name': get_vendor_name(),
                      'doc_ref_no_prefix': get_doc_ref_no_prefix(),
                      'contract_number': get_contract_number(),
                      'transaction_amount': get_amount(),
                      'TYPE': 'global',
                      'CLASS': 1}

            # collect sampled random global anomaly
            global_anomalies.append(sample)

        # convert all global anomalies to pandas data frame
        global_anomalies = pd.DataFrame(global_anomalies)

        # check if global anomalies are not in original dataset
        for i, global_anomaly in global_anomalies.iterrows():

            # check for potential duplicates
            df_tmp = self.dataset[(self.dataset['document_no'] == global_anomaly['document_no'])
                                  & (self.dataset['sub_obj'] == global_anomaly['sub_obj'])
                                  & (self.dataset['fm'] == global_anomaly['fm'])
                                  & (self.dataset['dept'] == global_anomaly['dept'])
                                  & (self.dataset['doc_ref_no_prefix'] == global_anomaly['doc_ref_no_prefix'])]

            # case: no similar transactions found in original dataset
            if not df_tmp.empty:

                # raise exception
                raise Exception('global outliers need to be executed with different seed')

        # return created global anomalies
        return global_anomalies

    # generate local anomalies based on philadelphia dataset
    def generate_local_graph_anomalies_ey(self, n=10, top_frequent_values=10):

        def get_amount(amount_mean, amount_std):
            return abs(np.random.normal(amount_mean, amount_std, 1))[0]

        # determine amount mean and variance
        amount_mean = stats.mode(self.dataset['transaction_amount'])[0]
        amount_std = np.std(self.dataset['transaction_amount']) / 400.0

        # init random seed
        np.random.seed(self.seed)

        # init local anomalies
        local_anomalies = []

        # init local anomaly count
        i = 0

        # iterate over number of local anomalies
        while len(local_anomalies) < n:

            # init single local anomaly
            sample = {}

            # iterate over categorical attribute
            for cat_attr in self.categorical_attributes:

                # select top most frequent values of this attribute
                freq_values = self.dataset[cat_attr].value_counts().nlargest(top_frequent_values).index.tolist()

                # sample from the most frequent values
                sample[cat_attr] = np.random.choice(freq_values)

            # add amount manually
            sample['transaction_amount'] = get_amount(amount_mean=amount_mean, amount_std=amount_std)
            sample['TYPE'] = 'local'
            sample['CLASS'] = 1

            # case: generated local anomaly is not in regular data
            if self.dataset[(self.dataset['document_no'] == sample['document_no'])
                            & (self.dataset['sub_obj'] == sample['sub_obj'])
                            & (self.dataset['fm'] == sample['fm'])
                            & (self.dataset['dept'] == sample['dept'])
                            & (self.dataset['doc_ref_no_prefix'] == sample['doc_ref_no_prefix'])
                            & (self.dataset['vendor_name'] == sample['vendor_name'])].empty:

                # append local anomaly
                local_anomalies.append(sample)

            # case: generated local anomaly is in regular data
            else:

                # increase total count of to be generated anomalies
                i += 1

                # print a warning if there are too many generated anomalies being rejected
                if (i % 1000 == 0) and (i != 0):

                    # print anomaly generation result
                    print('{} generated local anomalies were rejected -> consider another seed or increase top frequent values parameter'.format(i))

        # convert to pandas data frame
        local_anomalies = pd.DataFrame(local_anomalies)

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