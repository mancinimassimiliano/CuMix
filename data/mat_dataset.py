# Code taken from https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning
# just added the domain tensor for compliance.

# import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
import torch.utils.data as data
import os

def map_label(label, classes, shift=0):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i + shift
    return mapped_label

def get_attributes(labels, attributes):
    mapped_attributes = torch.FloatTensor(labels.size(0), attributes.size(1))
    for i,l in enumerate(labels):
        mapped_attributes[i] = attributes[l]
    return mapped_attributes

class MatDataset(data.Dataset):
    def __init__(self, data_root, dataset, image_embedding='res101', class_embedding='att_splits', train=True, preprocess=True, standardization=False):
        self.dataset = dataset
        self.image_embedding = image_embedding
        self.class_embedding = class_embedding
        self.data_root = data_root
        self.train = train
        self.preprocessing = preprocess
        self.standardization = standardization
        self.read_matdataset()

    def read_matdataset(self):
        path_images = os.path.join(self.data_root, self.dataset, self.image_embedding + ".mat")
        path_embeddings = os.path.join(self.data_root, self.dataset, self.class_embedding + ".mat")
        matcontent = sio.loadmat(path_images)
        feature = matcontent['features'].T
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(path_embeddings)

        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.class_attributes = torch.from_numpy(matcontent['att'].T).float()

        if self.preprocessing:
                if self.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                train_feature = torch.from_numpy(_train_feature).float()
                mx = train_feature.max()
                train_feature.mul_(1 / mx)
                train_label = torch.from_numpy(label[trainval_loc]).long()
                test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                test_unseen_feature.mul_(1 / mx)
                test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
        else:
                train_feature = torch.from_numpy(feature[trainval_loc]).float()
                train_label = torch.from_numpy(label[trainval_loc]).long()
                test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()

        seenclasses = torch.from_numpy(np.unique(train_label.numpy()))
        unseenclasses = torch.from_numpy(np.unique(test_unseen_label.numpy()))

        if self.train:
            self.original_labels = train_label
            train_mapped_label = map_label(train_label, seenclasses)
            self.features = train_feature
            self.labels = train_mapped_label
            self.classes = seenclasses.size(0)
        else:
            self.original_labels = test_unseen_label
            test_unseen_mapped_label = map_label(test_unseen_label, unseenclasses)
            self.features = test_unseen_feature
            self.labels = test_unseen_mapped_label
            self.classes = unseenclasses.size(0)

        self.attributes = get_attributes(self.original_labels, self.class_attributes)
        self.seen = seenclasses
        self.unseen = unseenclasses
        self.full_labels = torch.cat([seenclasses, unseenclasses], dim=0)
        self.full_attributes = self.class_attributes


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, attribute, target) where target is class_index of the target class
                    and attribute its corresponding embedding.
        """
        return self.features[index], self.attributes[index], 0, self.labels[index]

    def __len__(self):
        return self.features.size(0)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.dataset
        return fmt_str



