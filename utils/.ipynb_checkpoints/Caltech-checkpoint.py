from PIL import Image
import os
import os.path
import random
import torch.utils.data as data
import pandas as pd
# from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg


class Caltech101:
    """`Caltech 101 <http://www.vision.caltech.edu/Image_Datasets/Caltech101/>`_ Dataset.
    .. warning::
        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech101`` exists or will be saved to if download is set to True.
        target_type (string or list, optional): Type of target to use, ``category`` or
        ``annotation``. Can also be a list to output a tuple with all specified target types.
        ``category`` represents the target class, and ``annotation`` is a list of points
        from a hand-generated outline. Defaults to ``category``.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, target_type="category", download=False):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        if not isinstance(target_type, list):
            target_type = [target_type]
        self.target_type = [verify_str_arg(t, "target_type", ("category", "annotation"))
                            for t in target_type]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "101_ObjectCategories")))
        self.categories.remove("BACKGROUND_Google")  # this is not a real class

        # For some reason, the category names in "101_ObjectCategories" and
        # "Annotations" do not always match. This is a manual map between the
        # two. Defaults to using same name, since most names are fine.
        name_map = {"Faces": "Faces_2",
                    "Faces_easy": "Faces_3",
                    "Motorbikes": "Motorbikes_16",
                    "airplanes": "Airplanes_Side_2"}
        self.annotation_categories = list(map(lambda x: name_map[x] if x in name_map else x, self.categories))

        self.index = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "101_ObjectCategories", c)))
            self.index.extend(range(1, n + 1))
            self.y.extend(n * [i])



    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "101_ObjectCategories"))

    def __len__(self):
        return len(self.index)

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

#         download_and_extract_archive(
#             "http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz",
#             self.root,
#             filename="101_ObjectCategories.tar.gz",
#             md5="b224c7392d521a49829488ab0f1120d9")
#         download_and_extract_archive(
#             "http://www.vision.caltech.edu/Image_Datasets/Caltech101/Annotations.tar",
#             self.root,
#             filename="101_Annotations.tar",
#             md5="6f83eeb1f24d99cab4eb377263132c91")

#     def extra_repr(self):
#         return "Target type: {target_type}".format(**self.__dict__)


class Caltech256:
    """`Caltech 256 <http://www.vision.caltech.edu/Image_Datasets/Caltech256/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``caltech256`` exists or will be saved to if download is set to True.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    def __init__(self, root, train_file=None, val_file=None, test_file=None, download=False):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.num_train = 30
        self.num_val = 25
        self.num_test = 25
        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.categories = sorted(os.listdir(os.path.join(self.root, "256_ObjectCategories")))
        self.train_label = []
        self.train_images = []
        self.val_label = []
        self.val_images = []
        self.test_label = []
        self.test_images = []
        if train_file is not None and val_file is not None and test_file is not None:
            traindf = pd.read_csv(train_file)
            valdf = pd.read_csv(val_file)
            testdf = pd.read_csv(test_file)
            for index, row in traindf.iterrows():
                self.train_images.append(os.path.join(self.root,"256_ObjectCategories",row[0],row[1]))
                self.train_label.append(self.categories.index(row[0]))
            for index, row in valdf.iterrows():
                self.val_images.append(os.path.join(self.root,"256_ObjectCategories",row[0],row[1]))
                self.val_label.append(self.categories.index(row[0]))
            for index, row in testdf.iterrows():
                self.test_images.append(os.path.join(self.root,"256_ObjectCategories",row[0],row[1]))
                self.test_label.append(self.categories.index(row[0]))
        else:

            for (i, c) in enumerate(self.categories):
                n = len([name for name in os.listdir(os.path.join(self.root, "256_ObjectCategories", c)) if name.endswith('jpg')])
                index = list(range(1,n+1))
                random.shuffle(index)
                self.test_label.extend(self.num_test*[i])
                self.test_images.extend([os.path.join(self.root,
                             "256_ObjectCategories",c,
                             "{:03d}_{:04d}.jpg".format(i+1, k)) for k in index[:self.num_test]])
                self.val_label.extend(self.num_val * [i])
                self.val_images.extend([os.path.join(self.root,
                             "256_ObjectCategories",c,
                             "{:03d}_{:04d}.jpg".format(i+1, k)) for k in index[self.num_test:self.num_val+self.num_test]])
                self.train_label.extend(self.num_train * [i])
                self.train_images.extend([os.path.join(self.root,
                             "256_ObjectCategories",c,
                             "{:03d}_{:04d}.jpg".format(i+1, k)) for k in index[self.num_val+self.num_test : self.num_val+self.num_test+self.num_train ]])

    def _check_integrity(self):
        # can be more robust and check hash of files
        return os.path.exists(os.path.join(self.root, "256_ObjectCategories"))


    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return

#         download_and_extract_archive(
#             "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar",
#             self.root,
#             filename="256_ObjectCategories.tar",
#             md5="67b4f42ca05d46448c6bb8ecd2220f6d")

class ImageDataset(data.Dataset):


    def __init__(self, images, labels, transforms=None):

        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img = Image.open(self.images[index]).convert('RGB')
        target = self.labels[index]

        if self.transforms:
            img = self.transforms(img)


        return img, target
