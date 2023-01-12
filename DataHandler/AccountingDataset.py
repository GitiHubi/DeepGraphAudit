# import pytorch libraries
from torch.utils import data

# define accounting dataset
class AccountingDataset(data.Dataset):

    # define the class constructor
    def __init__(self, dataset):

        # set datasets
        self.input = dataset
        self.target = dataset

    # define the length method
    def __len__(self):

        # returns the number of samples
        return len(self.input)

    # define the get item method
    def __getitem__(self, index):

        # determine mini-batches
        input_batch = self.input[index, :]
        target_batch = self.target[index, :]

        # return sequences and target
        return input_batch, target_batch