import imageio as imageio
from torch.utils.data import Dataset
# import pandas as pd
import os
import numpy as np
from skimage import transform


class DTU3DFaceDataset(Dataset):
    """
    DTU-3D face dataset
    Class inspired from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, heatmap_size=256, image_size=256, image_channels="RGB",
                 n_views=96, tfrm=None):
        """
        Args:
            csv_file (string): Path to the csv/txt file with file ids.
            root_dir (string): Root directory for data.
            tfrm (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.file_ids = []
        with open(csv_file) as f:
            for line in f:
                line = line.strip("/n")
                line = line.strip("\n")
                if len(line) > 0:
                    self.file_ids.append(line)
        # Use semicolon as delimiter since some file names have comma
        # self.file_ids_frame = pd.read_csv(csv_file, sep=';')
        # TODO: remove debug print
        print('Read ', len(self.file_ids), ' file ids')

        self.root_dir = root_dir
        self.transform = tfrm
        self.heatmap_size = heatmap_size
        self.image_size = image_size
        self.image_channels = image_channels
        self.n_views = n_views

        # Generate the ids of the augmented file names
        self.id_table = []
        for f_name in self.file_ids:
            clean_name, file_extension = os.path.splitext(f_name)
            for n in range(self.n_views):
                augment_name = clean_name + '_' + str(n)
                self.id_table.append(augment_name)
        print('Generated ', len(self.id_table), ' file ids including augmentations')

    def _make_gaussian(self, height, width, sigma=3, center=None):
        """
        Make a square gaussian kernel.
        size is the length of a side of the square
        sigma is full-width-half-maximum, which
        can be thought of as an effective radius.
        """
        x = np.arange(0, width, 1, float)
        y = np.arange(0, height, 1, float)[:, np.newaxis]
        if center is None:
            x0 = width // 2
            y0 = height // 2
        else:
            x0 = center[0]
            y0 = center[1]
        return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2)

    def _generate_heat_maps(self, height, width, lms, max_length):
        """
        Generate a full Heap Map for every landmark in an array
        Args:
            height           : Wanted Height for the heat Map
            width            : Wanted Width for the heat Map
            lms              : Array of landmarks
            max_length        : Length of the Bounding Box
        """
        num_lms = lms.shape[0]
        hm = np.zeros((height, width, num_lms), dtype=np.float32)
        for i in range(num_lms):
            if not (np.array_equal(lms[i], [-1, -1])):
                s = int(np.sqrt(max_length) * max_length * 10 / 4096) + 2
                hm[:, :, i] = self._make_gaussian(height, width, sigma=s, center=(lms[i, 0], lms[i, 1]))
            else:
                hm[:, :, i] = np.zeros((height, width))
        return hm

    def __len__(self):
        return len(self.id_table)

    def __getitem__(self, idx):
        # print('Returning item ', idx)
        file_name = self.id_table[idx]
        lm_name = os.path.join(self.root_dir, '2D LM', file_name + '.txt')
        try:
            input_file = open(lm_name, 'r')
        except IOError:
            print('Cannot open ', lm_name)
            return None, None
        landmarks = np.array([line.rstrip().split(' ') for line in input_file])
        landmarks = landmarks.astype(float)

        # Generate target heat maps
        hm_size = self.heatmap_size
        # TODO this should be put into a configuration file - when the whole training loop is converted to Python
        org_img_size = 1024
        scaled_lm = landmarks / org_img_size * hm_size
        heat_map = self._generate_heat_maps(hm_size, hm_size, scaled_lm, hm_size)

        # Expand the heatmap so there are one for each stack in the network
        # the copies of the heat map is used to make the target data compatible with the network output
        # where there is a heatmap generated after each stack
        # TODO: number of stacks should be in configuration file
        n_stacks = 2
        heat_map = np.expand_dims(heat_map, axis=0)
        heat_map = np.repeat(heat_map, n_stacks, axis=0)

        # Type of rendering: geom, depth, RGB, curvature, geom+depth
        rendering_type = self.image_channels

        # Size of input image in network
        img_size = self.image_size
        if rendering_type == 'geometry':
            image = np.zeros((img_size, img_size, 1), dtype=np.float32)
            image_file = os.path.join(self.root_dir, 'images', file_name + '_geometry.png')
            img_in = transform.resize(imageio.imread(image_file), (img_size, img_size), mode='constant')
            image[:, :, 0] = img_in[:, :, 0]  # depth image is stored as a 3-channel image
        elif rendering_type == 'depth':
            image = np.zeros((img_size, img_size, 1), dtype=np.float32)
            image_file = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
            img_in = transform.resize(imageio.imread(image_file), (img_size, img_size), mode='constant')
            image[:, :, 0] = img_in[:, :]  # depth image is a pure grey level image
        elif rendering_type == 'RGB':
            image = np.zeros((img_size, img_size, 3), dtype=np.float32)
            image_file = os.path.join(self.root_dir, 'images', file_name + '.png')
            img_in = transform.resize(imageio.imread(image_file), (img_size, img_size), mode='constant')
            image[:, :, :] = img_in[:, :, :]
        else:
            print('Rendering type ', rendering_type, ' not supported')
            image = None

        sample = {'image': image, 'heat_map_stack': heat_map}

        if self.transform:
            sample = self.transform(sample)

        return sample
