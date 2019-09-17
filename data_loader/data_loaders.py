import os

from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader.DTU3DFaceDataset import DTU3DFaceDataset


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class DTU3DDataLoader(BaseDataLoader):
    """
    DTU-3D data loader
    """

    def __init__(self, data_dir, heatmap_size=256, image_size=256, image_channels="RGB", n_views=96, batch_size=8,
                 shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.csv_file_name = os.path.join(self.data_dir, 'face_dataset_full.txt')
        self.dataset = DTU3DFaceDataset(csv_file=self.csv_file_name, root_dir=data_dir,
                                        heatmap_size=heatmap_size, image_size=image_size,
                                        image_channels = image_channels, n_views=n_views)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
