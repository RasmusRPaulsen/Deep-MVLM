import imageio as imageio
from torch.utils.data import Dataset
import os
import numpy as np
from skimage import transform


class FaceDataset(Dataset):
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
        self._check_image_files()

    def _check_if_valid_file(self, file_name):
        if not os.path.isfile(file_name):
            print(file_name, " is not a file!")
            return False
        elif os.stat(file_name).st_size < 10:
            print(file_name, " is not valid (length less than 10 bytes)")
            return False
        return True

    def _check_image_files(self):
        rendering_type = self.image_channels
        print('Checking if all files are there')
        new_id_table = []
        for file_name in self.id_table:
            if rendering_type == 'geometry':
                image_file = os.path.join(self.root_dir, 'images', file_name + '_geometry.png')
                if self._check_if_valid_file(image_file):
                    new_id_table.append(file_name)
            elif rendering_type == 'depth':
                image_file = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
                if self._check_if_valid_file(image_file):
                    new_id_table.append(file_name)
            elif rendering_type == 'RGB':
                image_file = os.path.join(self.root_dir, 'images', file_name + '.png')
                if self._check_if_valid_file(image_file):
                    new_id_table.append(file_name)
            elif rendering_type == 'RGB+depth':
                image_file = os.path.join(self.root_dir, 'images', file_name + '.png')
                image_file2 = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
                if self._check_if_valid_file(image_file) and self._check_if_valid_file(image_file2):
                    new_id_table.append(file_name)
            elif rendering_type == 'geometry+depth':
                image_file = os.path.join(self.root_dir, 'images', file_name + '_geometry.png')
                image_file2 = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
                if self._check_if_valid_file(image_file) and self._check_if_valid_file(image_file2):
                    new_id_table.append(file_name)

        print('Checking done')
        self.id_table = new_id_table
        print('Final ', len(self.id_table), ' file ids including augmentations')

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

    def _safe_read_and_scale_image(self, image_file, img_size):
        img_in = None
        org_size = 1024  # Default value
        if self._check_if_valid_file(image_file):
            try:
                img_t = imageio.imread(image_file)
                org_size = img_t.shape[0]
                if org_size == img_size:
                    img_in = img_t / 255  # The resize operation scale the pixel values to [0,1]. With no scale we do it
                else:
                    img_in = transform.resize(img_t, (img_size, img_size), mode='constant')
            except IOError as e:
                print("File ", image_file, " raises exception")
                print("I/O error({0}): {1}".format(e.errno, e.strerror))
            except ValueError:
                print("File ", image_file, " raises exception")
                print("ValueError")
        return img_in, org_size

    def __len__(self):
        return len(self.id_table)

    def __getitem__(self, idx):
        # print('Returning item ', idx)
        file_name = self.id_table[idx]

        # Type of rendering: geom, depth, RGB, curvature, geom+depth
        rendering_type = self.image_channels
        org_img_size = 1024  # just a default value

        # Size of input image in network
        img_size = self.image_size
        if rendering_type == 'geometry':
            image = np.zeros((img_size, img_size, 1), dtype=np.float32)
            image_file = os.path.join(self.root_dir, 'images', file_name + '_geometry.png')
            img_in, org_img_size = self._safe_read_and_scale_image(image_file, img_size)
            if img_in is not None:
                image[:, :, 0] = img_in[:, :, 0]  # geometry image is stored as a 3-channel image
        elif rendering_type == 'depth':
            image = np.zeros((img_size, img_size, 1), dtype=np.float32)
            image_file = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
            img_in, org_img_size = self._safe_read_and_scale_image(image_file, img_size)
            if img_in is not None:
                image[:, :, 0] = img_in[:, :]  # depth image is a pure grey level image
        elif rendering_type == 'RGB':
            image = np.zeros((img_size, img_size, 3), dtype=np.float32)
            image_file = os.path.join(self.root_dir, 'images', file_name + '.png')
            img_in, org_img_size = self._safe_read_and_scale_image(image_file, img_size)
            if img_in is not None:
                image[:, :, :] = img_in[:, :, :]
        elif rendering_type == 'RGB+depth':
            image = np.zeros((img_size, img_size, 4), dtype=np.float32)
            image_file = os.path.join(self.root_dir, 'images', file_name + '.png')
            img_in, org_img_size = self._safe_read_and_scale_image(image_file, img_size)
            if img_in is not None:
                image[:, :, 0:3] = img_in[:, :, :]
            image_file = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
            img_in, org_img_size = self._safe_read_and_scale_image(image_file, img_size)
            if img_in is not None:
                image[:, :, 3] = img_in[:, :]  # depth image is a pure grey level image
        elif rendering_type == 'geometry+depth':
            image = np.zeros((img_size, img_size, 2), dtype=np.float32)
            image_file = os.path.join(self.root_dir, 'images', file_name + '_geometry.png')
            img_in, org_img_size = self._safe_read_and_scale_image(image_file, img_size)
            if img_in is not None:
                image[:, :, 0] = img_in[:, :, 0]  # geometry image is stored as a 3-channel image
            image_file = os.path.join(self.root_dir, 'images', file_name + '_zbuffer.png')
            img_in, org_img_size = self._safe_read_and_scale_image(image_file, img_size)
            if img_in is not None:
                image[:, :, 1] = img_in[:, :]  # depth image is a pure grey level image
        else:
            print('Rendering type ', rendering_type, ' not supported')
            image = None

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
        scaled_lm = landmarks / org_img_size * hm_size
        heat_map = self._generate_heat_maps(hm_size, hm_size, scaled_lm, hm_size)

        # Expand the heatmap so there are one for each stack in the network
        # the copies of the heat map is used to make the target data compatible with the network output
        # where there is a heatmap generated after each stack
        # In this implementation we are limited to two stacks
        n_stacks = 2
        heat_map = np.expand_dims(heat_map, axis=0)
        heat_map = np.repeat(heat_map, n_stacks, axis=0)

        sample = {'image': image, 'heat_map_stack': heat_map}

        if self.transform:
            sample = self.transform(sample)

        return sample
