import os, sys
import os.path as osp
import lmdb
import pyarrow as pa
import torch.utils.data as data
import cv2
import numpy as np


root_folder = '/DataS/zhanghan_data/lmdb/'
folder_name = 'train/'
root_data = root_folder + folder_name


def get_pair_lst(root_dir, folder_dir):
    files_all = os.listdir(root_dir + folder_dir)
    files_opt = []
    files_sar = []
    # pair_list = []
    for f in files_all:
        if f[0:3] == 'opt':
            files_opt.append(f)
            fs = 'sar' + f[3:]
            files_sar.append(fs)
    pair_lst = [(files_opt[i], files_sar[i]) for i in range(len(files_opt))]
    return pair_lst


def imagepairs2lmdb(name_lmdb, write_frequency=5000):
    root_dir = '/DataS/zhanghan_data/OSdataset/256/'
    folder_imagepair = 'test/'

    print("Loading dataset from %s" % folder_imagepair)
    pair_lst = get_pair_lst(root_dir, folder_imagepair)

    len_dataset = len(pair_lst)

    img_name_slave = os.path.join(root_dir + folder_imagepair, pair_lst[0][0])
    patch_s = cv2.imread(img_name_slave, 0)

    data_size_per_img = patch_s.nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len_dataset * 2

    lmdb_path = osp.join(root_folder, "%s" % name_lmdb)
    isdir = os.path.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=data_size * 1.1, readonly=False,
                   meminit=False, map_async=True)

    txn = db.begin(write=True)

    idx = 0
    for names in pair_lst:
        img_name_master = os.path.join(root_dir+folder_imagepair, names[0])
        patch_e = cv2.imread(img_name_master, 0)
        img_name_slave = os.path.join(root_dir+folder_imagepair, names[1])
        patch_s = cv2.imread(img_name_slave, 0)

        img_pair = np.array([patch_e, patch_s])

        txn.put(u'{}'.format(idx).encode('ascii'), dumps_pyarrow(img_pair))
        if idx == 14319:
            print(idx)

        if idx % write_frequency == 0:
            print("[%d/%d]" % (idx, len_dataset))
            txn.commit()
            txn = db.begin(write=True)
        idx = idx + 1

    idx = idx - 1

        # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', dumps_pyarrow(keys))
        txn.put(b'__len__', dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    db.sync()
    db.close()
    return 0


class DatasetLMDB(data.Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.transform = transform

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        img_pair = loads_pyarrow(byteflow)

        return img_pair

    def __len__(self):
        return self.length


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


if __name__ == '__main__':

    name_lmdb = 'osdataset_test.lmdb'
    file_lmdb = root_folder + name_lmdb

    print(osp.exists(file_lmdb))
    if not osp.exists(file_lmdb):
        imagepairs2lmdb(name_lmdb, write_frequency=5000)
    print(len(DatasetLMDB(file_lmdb)))
    dset = DatasetLMDB(file_lmdb)
    import matplotlib.pyplot as plt
    for id in range(0,10):
        tmp = dset[id]
        plt.figure(1)
        plt.subplot(1, 2, 1)
        plt.imshow(tmp[0, :, :])
        plt.subplot(1, 2, 2)
        plt.imshow(tmp[1, :, :])
        plt.show()



