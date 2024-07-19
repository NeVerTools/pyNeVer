import abc
from collections.abc import Callable

import numpy as np
import torch.utils.data as tdata
import torchvision as tv


class Dataset(abc.ABC):
    """
    An abstract class used to represent a Dataset. The concrete descendant must
    implement the methods __getitem__ and __len__.

    """

    @abc.abstractmethod
    def __getitem__(self, index: int):
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError


class TorchMNIST(Dataset, tv.datasets.MNIST):
    """
    A concrete class used to represent the MNIST Dataset. It leverages the torch dataset MNIST.

    Attributes
    ----------
    data_path : str
        Path to the folder in which the dataset will be saved.
    train : bool
        If True then the training set is loaded otherwise the test set is loaded.
    transform : Callable, Optional
        Transformation to apply to the data. We assume this is an object like the transforms presented in torchvision.
        The parameters of the callable (other than the object subject to the transformation) should be attributes of
        the object.
    target_transform : Callable, Optional
        Transformation to apply to the targets. We assume this is an object like the transforms presented in
        torchvision. The parameters of the callable (other than the object subject to the transformation) should be
        attributes of the object.
    download : bool
        True if the dataset must be downloaded, False otherwise.

    """

    def __init__(self, data_path: str, train: bool, transform: Callable | None = None,
                 target_transform: Callable | None = None, download: bool = True):
        Dataset.__init__(self)
        tv.datasets.MNIST.__init__(self, data_path, train, transform, target_transform, download)

    def __getitem__(self, index: int):
        return tv.datasets.MNIST.__getitem__(self, index)

    def __len__(self):
        return tv.datasets.MNIST.__len__(self)


class TorchFMNIST(Dataset, tv.datasets.FashionMNIST):
    """
    A concrete class used to represent the FMNIST Dataset. It leverages the torch dataset FMNIST.

    Attributes
    ----------
    data_path : str
        Path to the folder in which the dataset will be saved.
    train : bool
        If True then the training set is loaded otherwise the test set is loaded.
    transform : Callable, Optional
        Transformation to apply to the data. We assume this is an object like the transforms presented in torchvision.
        The parameters of the callable (other than the object subject to the transformation) should be attributes of
        the object.
    target_transform : Callable, Optional
        Transformation to apply to the targets. We assume this is an object like the transforms presented in
        torchvision. The parameters of the callable (other than the object subject to the transformation) should be
        attributes of the object.
    download : bool
        True if the dataset must be downloaded, False otherwise.

    """

    def __init__(self, data_path: str, train: bool, transform: Callable | None = None,
                 target_transform: Callable | None = None, download: bool = True):
        Dataset.__init__(self)
        tv.datasets.FashionMNIST.__init__(self, data_path, train, transform, target_transform, download)

    def __getitem__(self, index: int):
        return tv.datasets.FashionMNIST.__getitem__(self, index)

    def __len__(self):
        return tv.datasets.FashionMNIST.__len__(self)


class GenericFileDataset(Dataset, tdata.Dataset):
    """
    A concrete class used to represent a generic dataset memorized as a txt file. It loads the values using numpy
    loadtxt function. It assumes each line of the file is a separated datapoint.
    For each line we assume that the first n values are the input and the following are the target. The index of the
    first element of the target is identified by the target_index attribute.

    Attributes
    ----------
    filepath : str
        Path to the file containing the dataset.
        N.B.: the names of the dataset are supposed to be jame_pos_*.txt where * can be tested or train.
    target_index : int
        Index of the first element of the outputs.
    dtype : type, Optional
        Data type of the values of the data-points. Refer to numpy.loadtxt for more details.
    delimiter : str, Optional
        Delimiter between the different values of the data-points. Refer to numpy.loadtxt for more details.
    transform : Callable, Optional
        Transformation to apply to the data. We assume this is an object like the transforms presented in torchvision.
        The parameters of the callable (other than the object subject to the transformation) should be attributes of
        the object.
    target_transform : Callable, Optional
        Transformation to apply to the targets. We assume this is an object like the transforms presented in
        torchvision. The parameters of the callable (other than the object subject to the transformation) should be
        attributes of the object.

    """

    def __init__(self, filepath: str, target_index: int, dtype: type = float, delimiter: str = ",",
                 transform: Callable | None = None, target_transform: Callable | None = None):

        self.filepath = filepath
        self.target_index = target_index
        self.dtype = dtype
        self.delimiter = delimiter
        self.transform = transform
        self.target_transform = target_transform

        dataset = np.loadtxt(filepath, dtype=self.dtype, delimiter=self.delimiter)

        self.__data, self.__targets = (dataset[:, 0:self.target_index], dataset[:, self.target_index:])

    def __getitem__(self, index: int) -> tuple:

        data, target = self.__data[index], self.__targets[index]
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self):
        return len(self.__data)


class DynamicsJamesPos(GenericFileDataset, tdata.Dataset):
    """
    A concrete class used to represent the Dynamic James Dataset presented in the paper
    "Challenging SMT solvers to verify neural networks" by Pulina and Tacchella (2012).
    Automatic download is at present not supported, therefore the dataset must be downloaded manually.

    Attributes
    ----------
    data_path : str
        Path to the folder containing the training set and the test set.
        N.B.: the names of the dataset are supposed to be james_pos_*.txt where * can be tested or train.
    train : bool
        If True then the training set is loaded otherwise the test set is loaded.
    transform : Callable, Optional
        Transformation to apply to the data. We assume this is an object like the transforms presented in torchvision.
        The parameters of the callable (other than the object subject to the transformation) should be attributes of
        the object.
    target_transform : Callable, Optional
        Transformation to apply to the targets. We assume this is an object like the transforms presented in
        torchvision. The parameters of the callable (other than the object subject to the transformation) should be
        attributes of the object.

    """

    def __init__(self, data_path: str, train: bool, transform: Callable | None = None,
                 target_transform: Callable | None = None):

        tdata.Dataset.__init__(self)

        if train:
            dataset_path = data_path + "james_pos_train.txt"
        else:
            dataset_path = data_path + "james_pos_test.txt"

        GenericFileDataset.__init__(self, dataset_path, 8, transform=transform, target_transform=target_transform)
