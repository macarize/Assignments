import os
import pickle
import warnings
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import random

warnings.filterwarnings("ignore")
BASEDIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_data_dir():
    return os.path.join(BASEDIR, 'data')


class TieredimageNetDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        super().__init__()
        self.root_dir = BASEDIR + '/data/tieredImageNet/'

        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

        if mode == 'matching_train':
            import numpy as np
            dataset_train = pickle.load(open(os.path.join(self.root_dir, 'mini-imagenet-cache-train.pkl'), 'rb'))
            dataset_val = pickle.load(open(os.path.join(self.root_dir, 'mini-imagenet-cache-val.pkl'), 'rb'))

            image_data_train = dataset_train['image_data']
            class_dict_train = dataset_train['class_dict']
            image_data_val = dataset_val['image_data']
            class_dict_val = dataset_val['class_dict']

            image_data = np.concatenate((image_data_train, image_data_val), axis=0)
            class_dict = class_dict_train.copy()
            class_dict.update(class_dict_val)
            dataset = {'image_data': image_data, 'class_dict': class_dict}
        else:
            dataset = pickle.load(open(os.path.join(self.root_dir, 'mini-imagenet-cache-' + mode + '.pkl'), 'rb'))

        self.x = dataset['image_data']

        self.y = torch.arange(len(self.x))
        for idx, (name, id) in enumerate(dataset['class_dict'].items()):
            if idx > 63:
                id[0] = id[0] + 38400
                id[-1] = id[-1] + 38400
            s = slice(id[0], id[-1] + 1)
            self.y[s] = idx

    def __getitem__(self, index):
        img = self.x[index]
        x = self.transform(image=img)['image']

        return x, self.y[index]

    def __len__(self):
        return len(self.x)


class MiniImageNetDataset(Dataset):
    def __init__(self, mode='train', transform=None):
        super().__init__()
        self.root_dir = BASEDIR + '/data/miniImageNet/Miniimagenet'

        if transform is None:
            self.transform = A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transform = transform

        if mode == 'matching_train':
            import numpy as np
            dataset_train = pickle.load(open(os.path.join(self.root_dir, 'mini-imagenet-cache-train.pkl'), 'rb'))
            dataset_val = pickle.load(open(os.path.join(self.root_dir, 'mini-imagenet-cache-val.pkl'), 'rb'))

            image_data_train = dataset_train['image_data']
            class_dict_train = dataset_train['class_dict']
            image_data_val = dataset_val['image_data']
            class_dict_val = dataset_val['class_dict']

            image_data = np.concatenate((image_data_train, image_data_val), axis=0)
            class_dict = class_dict_train.copy()
            class_dict.update(class_dict_val)
            dataset = {'image_data': image_data, 'class_dict': class_dict}
        else:
            dataset = pickle.load(open(os.path.join(self.root_dir, 'mini-imagenet-cache-' + mode + '.pkl'), 'rb'))

        self.x = dataset['image_data']

        self.y = torch.arange(len(self.x))
        for idx, (name, id) in enumerate(dataset['class_dict'].items()):
            if idx > 63:
                id[0] = id[0] + 38400
                id[-1] = id[-1] + 38400
            s = slice(id[0], id[-1] + 1)
            self.y[s] = idx

    def __getitem__(self, index):
        img = self.x[index]
        x = self.transform(image=img)['image']

        return x, self.y[index]

    def __len__(self):
        return len(self.x)


class tieredImageNetGenerator(object):
    """tieredImageNetGenerator
    Args:
        image_file (str): 'data/train_images.npz' or 'data/test_images.npz' or 'data/val_images.npz'
        label_file (str): 'data/train_labels.pkl' or 'data/test_labels.pkl' or 'data/val_labels.pkl'
        nb_classes (int): number of classes in an episode
        nb_samples_per_class (int): nuber of samples per class in an episode
        max_iter (int): max number of episode generation
        xp: numpy or cupy
    """

    def __init__(self, image_file, label_file, nb_classes=5, nb_samples_per_class=10,
                 max_iter=None, xp=np):
        super(tieredImageNetGenerator, self).__init__()
        self.image_file = image_file
        self.label_file = label_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self._load_data(self.image_file, self.label_file)

    def _load_data(self, image_file, label_file):
        with np.load(image_file, mmap_mode="r", encoding='latin1') as data:
            images = data["images"]
        with open(label_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            label_specific = data[b"label_specific"]
            label_specific_str = data[b"label_specific_str"]
        num_ex = label_specific.shape[0]
        ex_ids = np.arange(num_ex)
        num_label_cls_specific = len(label_specific_str)
        self.label_specific_idict = {}
        for cc in range(num_label_cls_specific):
            self.label_specific_idict[cc] = ex_ids[label_specific == cc]
        self.images = images

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.images)
    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels = self.sample(self.nb_classes, self.nb_samples_per_class)

            return ((self.num_iter - 1), (images, labels))
        else:
            self.num_iter = 0
            raise StopIteration()

    def sample(self, nb_classes, nb_samples_per_class):
        sampled_characters = random.sample(self.label_specific_idict.keys(), nb_classes)
        labels_and_images = []
        for (k, char) in enumerate(sampled_characters):
            _ind = random.sample(list(self.label_specific_idict[char]), nb_samples_per_class)
            labels_and_images.extend([(k, self.xp.array(self.images[i] / np.float32(255).flatten())) for i in _ind])
        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i + j * self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)

        return images, labels