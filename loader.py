import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from ordinal import build_training, build_test

def get_train_val_loaders():

    split_ratio = 5/6

    targets_frame, features_frame = build_training()

    train_dataset = RatingsDataset(
        features_frame=features_frame, targets_frame=targets_frame
    )
    val_dataset =  RatingsDataset(
        features_frame=features_frame, targets_frame=targets_frame
    )

    idxs = torch.randperm(len(train_dataset)).tolist()
    idx = int(split_ratio * len(idxs))

    train_dataset = torch.utils.data.Subset(train_dataset, idxs[0:idx])
    val_dataset = torch.utils.data.Subset(val_dataset, idxs[idx:])


    train_loader = DataLoader(
        train_dataset, batch_size = 2**5, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size = 2**5, shuffle=False
    )
    return train_loader, val_loader

def get_test_loader():
    targets_frame, features_frame = build_test()
    test_dataset = RatingsDataset(
        features_frame=features_frame, targets_frame=targets_frame
    )
    test_loader = DataLoader(
        test_dataset , batch_size = 2**5, shuffle=False
    )
    return test_loader

class RatingsDataset(Dataset):

    def __init__(self, features_frame, targets_frame):
        self.features_frame = features_frame
        self.targets_frame = targets_frame

    def __len__(self):
        return len(self.targets_frame)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        features = self.features_frame.iloc[idx,:]
        features = np.array(features)
        features = torch.tensor(features, dtype=torch.float32)
        target = self.targets_frame[idx]
        target = torch.tensor(target, dtype=torch.long)
        return features, target


