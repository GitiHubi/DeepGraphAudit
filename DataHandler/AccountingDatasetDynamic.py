# import pytorch libraries
from torch.utils import data

# define AEN accounting dataset
class AccountingDatasetDynamic(data.Dataset):

    # define the class constructor
    def __init__(self, feat_vectors):

        # set datasets
        self.feat_vectors = feat_vectors

    # define the length method
    def __len__(self):

        # returns the number of samples
        return len(self.feat_vectors)

    # define the get item method
    def __getitem__(self, index):

        # determine mini-batches
        feat_vectors_batch = self.feat_vectors[index, :]

        # return feature vector batch
        return feat_vectors_batch