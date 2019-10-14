import os

from base import BaseDataLoader
from data_loader.FaceDataset import FaceDataset


class FaceDataLoader(BaseDataLoader):
    """
    Face data loader
    """
    def __init__(self, data_dir, heatmap_size=256, image_size=256, image_channels="RGB", n_views=96, batch_size=8,
                 shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.csv_file_name = os.path.join(self.data_dir, 'dataset_train.txt')
        self.dataset = FaceDataset(csv_file=self.csv_file_name, root_dir=data_dir,
                                   heatmap_size=heatmap_size, image_size=image_size,
                                   image_channels=image_channels, n_views=n_views)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
