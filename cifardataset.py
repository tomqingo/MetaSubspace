from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
import random
import pdb

class CIFAR100SUB(VisionDataset):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d']
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc']
    ]

    meta = {
        'filename': 'meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, classselect, train=True, trainlist=False, transform=None, target_transform=None,
                 download=False, sample=False, samplenum=5):

        super(CIFAR100SUB, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.root = root
        self.train = train  # training set or test set
        self.sample = sample
        #pdb.set_trace()

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.classselect = classselect

        data = []
        targets = []
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        label_names = pickle.load(open(os.path.join(self.root, self.base_folder, 'meta'), 'rb'), encoding='ASCII')
        self.label_names = label_names['fine_label_names']

        if trainlist:
            with open('./cifar-fs-splits/train.txt', 'r') as f:
                content = f.readlines()
        else:
            with open('./cifar-fs-splits/test.txt', 'r') as f:
                content = f.readlines()
        classes = [x.strip() for x in content]
        classesid = []
        for idx in range(len(classes)):
            classidper = self.label_names.index(classes[idx])
            classesid.append(classidper)
        
        classesid.sort()
        
        self.classidselect = [classesid[idx] for idx in classselect]

        #pdb.set_trace()


        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                data.append(entry['data'])
                if 'labels' in entry:
                    targets.extend(entry['labels'])
                else:
                    targets.extend(entry['fine_labels'])

        data = np.vstack(data).reshape(-1, 3, 32, 32)
        #data = data.transpose((0, 2, 3, 1))  # convert to HWC

        targets = np.array(targets)
        for index in range(len(self.classidselect)):
            targetindex = [i for i,x in enumerate(targets) if x==self.classidselect[index]]
            if self.sample:
                random.shuffle(targetindex)
                targetindex = targetindex[0:samplenum]
                targetselect = [index for i in range(samplenum)]
            else:
                targetselect = [index for i,x in enumerate(targets) if x==self.classidselect[index]]
            dataselect = data[targetindex]
            self.targets.extend(targetselect)
            self.data.append(dataselect)
        #pdb.set_trace()

        self.data = np.vstack(self.data).reshape(-1,3,32,32)
        self.data = self.data.transpose((0,2,3,1))
        self.targets = np.array(self.targets)
        sampleindex = list(range(self.data.shape[0]))
        random.shuffle(sampleindex)
        self.data = self.data[sampleindex]
        self.targets = self.targets[sampleindex]
        self.targets = self.targets.tolist()

        #self._load_meta(self.classidselect)

    def _load_meta(self, classselect):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        self.class_to_idx = {}
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        
        for i in range(len(classselect)):
            classid = classselect[i]
            _class = self.classes[classid]
            self.class_to_idx.update({_class:(i,classid)})

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
