import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np

class ImageWithAttentionDataset(Dataset):
    '''
    A wrapper class to create a dataset of the scan and the attention mask.
    '''
    def __init__(self, dataframe, scan_type, transform=None):
        '''
        The constructor for the class.

        Parameters:
        - dataframe(pd.DataFrame): the input dataframe.
        - scan_type(String): the scan type.
        - transform(Bool): whether to perform transform or not.
        '''
        self.dataframe = dataframe
        self.transform = transform
        self.scan_type = scan_type

    def __len__(self):
        '''
        Returns the size of the dataframe.
        '''
        return len(self.dataframe)

    def __getitem__(self, idx):
        '''
        Obtains an item at the given index.

        Parameters:
        - idx(Int): the index of the dataframe.

        Returns:
        - Dict: a sample scan with its image, attention, and label data.
        '''
        row = self.dataframe.iloc[idx]
        t1c_img = nib.load(row[self.scan_type + '_path']).get_fdata()
        t1c_img = np.expand_dims(t1c_img, axis=0)
        attn_mask = nib.load(row[self.scan_type + '_processed_scan_path']).get_fdata()
        attn_mask = np.expand_dims(attn_mask, axis=0)
        seg = nib.load(row['label_path']).get_fdata()
        seg = np.expand_dims(seg, axis=0)
        t1c_img = (t1c_img - np.min(t1c_img)) / (np.max(t1c_img) - np.min(t1c_img) + 1e-5)
        attn_mask = (attn_mask - np.min(attn_mask)) / (np.max(attn_mask) - np.min(attn_mask) + 1e-5)
        sample = {
            'image': torch.tensor(t1c_img, dtype=torch.float32),
            'attention': torch.tensor(attn_mask, dtype=torch.float32),
            'label': torch.tensor(seg, dtype=torch.long)
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
