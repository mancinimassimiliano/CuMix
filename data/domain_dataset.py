import numpy as np
import torch
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import torch.utils.data as data
import os
from random import shuffle
import random
import torch.distributed as dist
import math

# Map label to ensure correct order on the classifier
def map_label(label, classes, shift=0):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i + shift
    return mapped_label

# Add attributes to labels
def get_attributes(labels, attributes):
    mapped_attributes = torch.FloatTensor(labels.size(0), attributes.size(1))
    for i, l in enumerate(labels):
        mapped_attributes[i] = attributes[l]
    return mapped_attributes

# Read text labels
def read_split_line(line):
    path, class_id = line.split(' ')
    class_name = path.split('/')[1]
    return path, class_name, int(class_id)

# Check overlap of splits, to ensure no unseen classes are on the seen
def check_integrity(split1, split2):
    for s in split1:
        if s in split2:
            print(s)
            return False
    return True

# DomainNet dataset
class DomainDataset(data.Dataset):
    def __init__(self, data_root, domains, attributes='w2v_domainnet.npy', train=True, validation=False, transformer=None):
        # Init dataset given the configs
        self.domains = domains
        self.n_doms = len(domains)
        self.class_embedding = attributes
        self.data_root = data_root
        self.train = train
        self.val = validation

        if self.val:
            seen_list = os.path.join(self.data_root, 'train_classes.npy')
            self.seen_list = list(np.load(seen_list))
            unseen_list = os.path.join(self.data_root, 'val_classes.npy')
            self.unseen_list = list(np.load(unseen_list))
        else:
            seen_list_train = os.path.join(self.data_root, 'train_classes.npy')
            self.seen_list = list(np.load(seen_list_train))
            seen_list_val = os.path.join(self.data_root, 'val_classes.npy')
            self.seen_list = self.seen_list + list(np.load(seen_list_val))
            unseen_list = os.path.join(self.data_root, 'test_classes.npy')
            self.unseen_list = list(np.load(unseen_list))

        # DomainNet specific checks
        assert len(self.unseen_list + self.seen_list) == 345 or len(
            self.unseen_list + self.seen_list) == 300, 'Not all the classes are there but ' + str(
            len(self.unseen_list + self.seen_list))
        assert check_integrity(self.unseen_list, self.seen_list), 'This is bad: seen and unseen classes are mixed!'

        self.full_classes = self.seen_list + self.unseen_list
        self.seen = torch.LongTensor([self.full_classes.index(k) for k in self.seen_list])
        self.unseen = torch.LongTensor([self.full_classes.index(k) for k in self.unseen_list])
        self.full_classes_idx = torch.cat([self.seen, self.unseen], dim=0)

        if self.train:
            self.valid_classes = self.seen_list
        else:
            self.valid_classes = self.unseen_list

        attributes_list = os.path.join(self.data_root, self.class_embedding)
        self.attributes_dict = np.load(attributes_list, allow_pickle=True, encoding='latin1').item()
        for k in list(self.attributes_dict.keys()) + []:
            self.attributes_dict[k] = torch.from_numpy(self.attributes_dict[k]).float()
        for i, k in enumerate(self.full_classes):
            self.attributes_dict[i] = self.attributes_dict[k]

        self.image_paths = []
        self.labels = []
        self.attributes = []
        self.domain_id = []

        self.loader = default_loader
        if transformer is None:
            self.transformer = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])
        else:
            self.transformer = transformer

        if isinstance(domains, list):
            for i, d in enumerate(domains):
                self.read_single_domain(d, id=i)
        else:
            self.read_single_domain(domains)

        self.labels = torch.LongTensor(self.labels)
        self.domain_id = torch.LongTensor(self.domain_id)
        self.attributes = torch.cat(self.attributes, dim=0) #torch.LongTensor(self.attributes)
        self.classes = len(self.valid_classes)
        self.full_attributes = self.attributes_dict

    # Read data from a single domain
    def read_single_domain(self, domain, id=0):
        if self.val or self.train:
            file_names = [domain + '_train.txt']
        else:
            # Note: if we are testing, we use all images of unseen classes contained in the domain,
            # no matter of the split. The images of the unseen classes are NOT present in the training phase.
            file_names = [domain + '_train.txt', domain + '_test.txt']

        for file_name in file_names:
            self.read_single_file(file_name, id)

    # Read all needed elements from file
    def read_single_file(self, filename, id):
        domain_images_list_path = os.path.join(self.data_root, filename)
        with open(domain_images_list_path, 'r') as files_list:
            for line in files_list:
                line = line.strip()
                local_path, class_name, _ = read_split_line(line)
                if class_name in self.valid_classes:
                    self.image_paths.append(os.path.join(self.data_root, local_path))
                    self.labels.append(self.valid_classes.index(class_name))
                    self.domain_id.append(id)
                    self.attributes.append(self.attributes_dict[class_name].unsqueeze(0))

    def get_domains(self):
        return self.domain_id, self.n_doms

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        features = self.loader(self.image_paths[index])
        features = self.transformer(features)
        return features, self.attributes[index], self.domain_id[index], self.labels[index]

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.dataset
        return fmt_str



# PACS dataset, same methods as the one above without seen vs unseen checks
class PACSDataset(data.Dataset):
    def __init__(self, data_root, domains, train=True, validation=False, transformer=None):
        self.domains = domains
        self.n_doms = len(domains)
        self.data_root = data_root
        self.train = train
        self.val = validation

        self.full_attributes = None
        self.seen = 7
        self.unseen = 7

        self.image_paths = []
        self.labels = []
        self.attributes = []
        self.domain_id = []

        self.loader = default_loader
        if transformer is None:
            self.transformer = transforms.Compose([transforms.Resize((224, 224)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                   ])
        else:
            self.transformer = transformer

        if isinstance(domains, list):
            for i, d in enumerate(domains):
                self.read_single_domain(d,i)
        else:
            self.read_single_domain(domains,0)

        self.labels = torch.LongTensor(self.labels)
        self.domain_id = torch.LongTensor(self.domain_id)
        self.attributes = torch.LongTensor(self.attributes)
        self.classes = 7

    def read_single_domain(self, domain, id):
        if self.train:
            file_names = [domain + '_train_kfold.txt']
        elif self.val:
            file_names = [domain + '_crossval_kfold.txt']
        else:
            file_names = [domain + '_test_kfold.txt']

        for file_name in file_names:
            self.read_single_file(file_name, id)

    def read_single_file(self, filename, id):
        domain_images_list_path = os.path.join(self.data_root, filename)
        with open(domain_images_list_path, 'r') as files_list:
            for line in files_list:
                line = line.strip()
                local_path, _, class_id = read_split_line(line)
                self.image_paths.append(os.path.join(self.data_root, 'kfold', local_path))
                self.labels.append(class_id-1)
                self.domain_id.append(id)
                self.attributes.append(0)

    def get_domains(self):
        return self.domain_id, self.n_doms

    def get_labels(self):
        return self.labels

    def __getitem__(self, index):
        features = self.loader(self.image_paths[index])
        features = self.transformer(features)
        return features, self.attributes[index], self.domain_id[index], self.labels[index]

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.dataset
        return fmt_str



# Balanced sampler, ensuring equal number of images per domain are present in the batch
class DistributedBalancedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, samples_per_domain, num_replicas=None, rank=None, shuffle=True, iters='min',
                 domains_per_batch=5):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.domain_ids, self.n_doms = self.dataset.get_domains()
        self.domain_ids = np.array(self.domain_ids)
        self.dict_domains = {}
        self.indeces = {}

        for i in range(self.n_doms):
            self.dict_domains[i] = []
            self.indeces[i] = 0

        self.dpb = domains_per_batch
        self.dbs = samples_per_domain
        self.bs = self.dpb * self.dbs

        for idx, d in enumerate(self.domain_ids):
            self.dict_domains[d].append(idx)

        min_dom = 10000000
        max_dom = 0

        for d in self.domain_ids:
            if len(self.dict_domains[d]) < min_dom:
                min_dom = len(self.dict_domains[d])
            if len(self.dict_domains[d]) > max_dom:
                max_dom = len(self.dict_domains[d])


        # When to conclude an iteration over the dataset
        if iters == 'min':
            self.iters = min_dom // self.dbs
        elif iters == 'max':
            self.iters = max_dom // self.dbs
        else:
            self.iters = int(iters)

        if shuffle:
            for idx in range(self.n_doms):
                random.shuffle(self.dict_domains[idx])

        self.num_samples = self.iters * self.dbs * self.n_doms // self.num_replicas
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

        self.samples = torch.LongTensor(self._get_samples())

    def __len__(self):
        return self.iters * self.bs

    # Sampling from one domain
    def _sampling(self, d_idx, n):
        if self.indeces[d_idx] + n >= len(self.dict_domains[d_idx]):
            self.dict_domains[d_idx] += self.dict_domains[d_idx]
        self.indeces[d_idx] = self.indeces[d_idx] + n
        return self.dict_domains[d_idx][self.indeces[d_idx] - n:self.indeces[d_idx]]

    # Order indeces to ensure balance
    def _get_samples(self):
        sIdx = []
        for i in range(self.iters // self.num_replicas):
            for j in range(self.n_doms):
                sIdx += self._sampling(j, self.dbs * self.num_replicas)
        return np.array(sIdx)

    def __iter__(self):
        if self.shuffle:
            indices = list(range(len(self.samples)))
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(self.samples[indices])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# Here we define a Sampler that has all the samples of each batch from the same domain,
# same as before but not distributed
class BalancedSampler(torch.utils.data.sampler.Sampler):

    def __init__(self, dataset, samples_per_domain, domains_per_batch=1, iters='min'):
        self.dataset = dataset
        self.domain_ids, self.n_doms = self.dataset.get_domains()
        self.domain_ids = np.array(self.domain_ids)
        self.dict_domains = {}
        self.indeces = {}

        for i in range(self.n_doms):
            self.dict_domains[i] = []
            self.indeces[i] = 0

        self.dpb = domains_per_batch
        self.dbs = samples_per_domain
        self.bs = self.dpb * self.dbs

        for idx, d in enumerate(self.domain_ids):
            self.dict_domains[d].append(idx)

        min_dom = 10000000
        max_dom = 0

        for d in self.domain_ids:
            if len(self.dict_domains[d]) < min_dom:
                min_dom = len(self.dict_domains[d])
            if len(self.dict_domains[d]) > max_dom:
                max_dom = len(self.dict_domains[d])

        if iters == 'min':
            self.iters = min_dom // self.dbs
        elif iters == 'max':
            self.iters = max_dom // self.dbs
        else:
            self.iters = int(iters)

        for idx in range(self.n_doms):
            shuffle(self.dict_domains[idx])
    def _sampling(self, d_idx, n):
        if self.indeces[d_idx] + n >= len(self.dict_domains[d_idx]):
            self.dict_domains[d_idx] += self.dict_domains[d_idx]
        self.indeces[d_idx] = self.indeces[d_idx] + n
        return self.dict_domains[d_idx][self.indeces[d_idx] - n:self.indeces[d_idx]]

    def _shuffle(self):
        sIdx = []
        for i in range(self.iters):
            for j in range(self.n_doms):
                sIdx += self._sampling(j, self.dbs)
        return np.array(sIdx)

    def __iter__(self):
        return iter(self._shuffle())

    def __len__(self):
        return self.iters * self.bs
