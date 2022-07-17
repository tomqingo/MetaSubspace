# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 22:23:32 2019
@author: Chen
"""

from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
from torchvision.datasets.utils import check_integrity, download_url
import torchvision.transforms as transforms
    
import torch.utils.data as data
import pdb

from PIL import Image

                
class CIFAR100_FS(data.Dataset):
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
    data_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc']
    ]

    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }
    
    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz, download=False):
        self.root = os.path.expanduser(root)
        self.batchsz = batchsz
        self.k_shot = k_shot
        self.n_way = n_way
        self.k_query = k_query
        self.imgsz = imgsz
        
        if not self._check_exists():
            if download:
                self.download()
            else:
                raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')
            
        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' + 
                               'You can use download=True to download it.')

        self.data = []
        self.labels = []
        
        label_names = pickle.load(open(os.path.join(self.root, self.base_folder, 'meta'), 'rb'), encoding="ASCII")
        self.label_names = label_names['fine_label_names']
        for fentry in self.data_list:
            f = fentry[0]
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
                #pdb.set_trace()
            self.data.append(entry['data'])
            #print(entry['fine_labels'])
            #pdb.set_trace()
            if 'labels' in entry:
                self.labels += entry['labels']
            else:
                self.labels += entry['fine_labels']
            fo.close()
                
        self.data = np.concatenate(self.data)
        self.data = self.data.reshape((60000, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))
        ord_idx = np.argsort(self.labels)
        self.data = self.data[ord_idx]
        self.labels = np.array(self.labels, dtype=np.int64)[ord_idx]
        #pdb.set_trace()
        

        # divide by the classes [[0:600],[600:1200],..100 classes]
        
        x_train = []
        x_val = []
        x_test = []

        for datatype in ['train','val','test']:
            with open(os.path.join('cifar-fs-splits', datatype + '.txt'), 'r') as f:
                content = f.readlines()
            classes = [x.strip() for x in content]
            for img_class in classes:
                 class_idx = self.label_names.index(img_class)
                 if datatype == 'train':          
                     x_train.append(self.data[class_idx*600:(class_idx+1)*600])
                 elif datatype == 'val':
                     x_val.append(self.data[class_idx*600:(class_idx+1)*600])
                 else:
                     x_test.append(self.data[class_idx*600:(class_idx+1)*600])
        self.x_train = np.array(x_train)
        self.x_val = np.array(x_val)
        self.x_test = np.array(x_test)

        #labels_all = list(range(0,100))
        #random.shuffle(labels_all)
        #pdb.set_trace()
        #self.x_train, self.x_val, self.x_test = self.data[labels_all[0:64]], self.data[labels_all[64:80]], self.data[labels_all[80:]]

        self.batchsz = batchsz
        self.n_cls = self.data.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20
        self.resize = imgsz

        self.indexes = {"train": 0, "val": 0, "test": 0}
        self.datasets = {"train": self.x_train, "val": self.x_val, "test": self.x_test}  # original data cached

        print("DB: train", self.x_train.shape, "val", self.x_val.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "val": self.load_data_cache(self.datasets["val"]),
                               "test": self.load_data_cache(self.datasets["test"])}

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        #if mode == 'train':
        querysz = self.k_query * self.n_way
        #else:
        #    querysz = self.k_shot*self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                np.random.shuffle(selected_cls)

                for j, cur_class in enumerate(selected_cls):
                    #if mode == 'train':
                    selected_img = np.random.choice(data_pack[cur_class].shape[0], self.k_shot + self.k_query, False)
                    #else:
                    #    selected_img = np.random.choice(data_pack[cur_class].shape[0], self.k_shot*2, False)
                    np.random.shuffle(selected_img)
                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    #if mode == 'train':
                    y_qry.append([j for _ in range(self.k_query)])
                    #else:
                    #    y_qry.append([j for _ in range(self.k_shot)])

                # shuffle inside a batch
                perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize, self.resize, 3)[perm]
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
                #if mode == 'train':
                perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize, self.resize, 3)[perm]
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]
                #else:
                #    perm = np.random.permutation(self.n_way * self.k_shot)
                #    x_qry = np.array(x_qry).reshape(self.n_way * self.k_shot, self.resize, self.resize, 3)[perm]
                #    y_qry = np.array(y_qry).reshape(self.n_way * self.k_shot)[perm]                   

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)


            # [b, setsz, 3, 32, 32]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.resize, self.resize, 3)
            y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
            # [b, qrysz, 3, 32, 32]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.resize, self.resize, 3)
            y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])
            #pdb.set_trace()
        #pdb.set_trace()
        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])
        
        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
    
    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return img, target
    
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in self.data_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True
    
    def _check_exists(self):
        return os.path.join(self.root, self.base_folder)
    
    def download(self):
        import tarfile
        
        if self._check_integrity():
            print('Files already downloaded and verified.')
            return
        
        root = self.root
        download_url(self.url, root, self.filename, self.tgz_md5)
        
        cwd = os.getcwd()
        tar = tarfile.open(os.path.join(root, self.filename), 'r:gz')
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        if self.train and not self.val:
            tmp = 'train'
        elif self.train and self.val:
            tmp = 'val'
        else:
            tmp = 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

if __name__ == '__main__':

    import  time
    import  torch
    import  visdom

    # plt.ion()
    viz = visdom.Visdom(env='omniglot_view')

    db = CIFAR100_FS('/home/datasets/CIFAR100', batchsz=20, n_way=5, k_shot=5, k_query=15, imgsz=32)

    train_transform = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    for i in range(1000):
        x_spt, y_spt, x_qry, y_qry = db.next('train')

        #pdb.set_trace()
        # [b, setsz, h, w, c] => [b, setsz, c, w, h] => [b, setsz, 3c, w, h]
        y_spt = torch.from_numpy(y_spt)
        y_qry = torch.from_numpy(y_qry)

        x_spt = x_spt.reshape(-1, 32, 32, 3)
        x_qry = x_qry.reshape(-1, 32, 32, 3)
        # pdb.set_trace()
        x_spt_tensor = Image.fromarray(np.uint8(x_spt[0]))
        x_spt_tensor = train_transform(x_spt_tensor)
        #pdb.set_trace()

        for x_spt_id in range(1,x_spt.shape[0]):
            x_spt_per = Image.fromarray(np.uint8(x_spt[x_spt_id]))
            x_spt_per = train_transform(x_spt_per)
            x_spt_tensor = torch.cat([x_spt_tensor, x_spt_per], dim=0)

        x_spt_tensor = x_spt_tensor.reshape(20,25,3,32,32)

        x_qry_tensor = Image.fromarray(np.uint8(x_qry[0]))
        x_qry_tensor = train_transform(x_qry_tensor)

        for x_qry_id in range(1,x_qry.shape[0]):
            x_qry_per = Image.fromarray(np.uint8(x_qry[x_qry_id]))
            x_qry_per = test_transform(x_qry_per)
            x_qry_tensor = torch.cat([x_qry_tensor, x_qry_per], dim=0)

        x_qry_tensor = x_qry_tensor.reshape(20,75,3,32,32)

        batchsz, setsz, c, h, w = x_spt_tensor.shape

        print(x_qry_tensor.shape)

        #viz.images(x_spt[0], nrow=5, win='x_spt', opts=dict(title='x_spt'))
        #viz.images(x_qry[0], nrow=15, win='x_qry', opts=dict(title='x_qry'))
        #viz.text(str(y_spt[0]), win='y_spt', opts=dict(title='y_spt'))
        #viz.text(str(y_qry[0]), win='y_qry', opts=dict(title='y_qry'))
        


        #time.sleep(10)
    
