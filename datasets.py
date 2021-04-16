import os
import cv2
import copy
import json
import jpeg4py
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cameralib
import torch
import utils
import pickle5 as pickle
import glob
import torch.utils.data as data

from torchvision import datasets
from torchvision import transforms
from augment_colour import random_color


def data_loader(args, phase, data_info):
    dataset = Dataset(data_info, phase, args)

    shuffle = args.shuffle if phase == 'train' else False

    return data.DataLoader(dataset, args.batch_size, shuffle, num_workers = args.workers, pin_memory = True)


def h36m_split(split, phase, sample):
    folder = os.path.basename(os.path.dirname(sample['image']))

    return folder.split('.')[0] in split[phase]


class Dataset(data.Dataset):

    def __init__(self, data_info, phase, args):
        assert len(data_info.short_names) == args.num_joints

        self.data_name = args.data_name
        with open('/globalwork/liu/metadata.json') as file:
            metadata = json.load(file)

        self.root = metadata['root'][args.data_name]

        self.data_info = data_info
        self.samples = getattr(self, 'get_' + args.data_name + '_samples')(phase, globals()[args.data_name + '_split'])

        self.at_test = phase != 'train'
        self.side_in = args.side_in

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.nexponent = args.nexponent
        self.colour = args.colour and (not self.at_test)
        self.geometry = args.geometry and (not self.at_test)
        self.random_zoom = args.random_zoom

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = self.mean, std = self.dev)])


    def get_h36m_samples(self, phase, split_by):
        sample_file = os.path.join(self.root, 'samples.pkl')

        with open(sample_file, 'rb') as file:
            samples = pickle.load(file)

        with open(os.path.join(self.root, 'split.json')) as file:
            split = json.load(file)

        return [sample for sample in samples if split_by(split, phase, sample)]


    def get_input_image(self, image_path, camera, bbox, do_flip, random_zoom):
        '''
        Turn towards the center of bbox then crop a square-shaped image aligned with the height of the bbox
        Args:
            image_path: path to the image that matches the camera's current state
            camera: current state of the camera
            bbox: bbox of the person that matches the camera's current state
        '''
        center = bbox[:2] + bbox[2:] / 2
        
        width = np.array([bbox[2] / 2, 0])
        height = np.array([0, bbox[3] / 2])

        if bbox[2] < bbox[3]:
            vertical = True
            near_side = np.stack([center - width, center + width])
            far_side = np.stack([center - height, center + height])
        else:
            vertical = False
            far_side = np.stack([center - width, center + width])
            near_side = np.stack([center - height, center + height])

        new_cam = copy.deepcopy(camera)
        new_cam.turn_towards(center)
        new_cam.undistort()
        new_cam.square_pixels()

        far_side = new_cam.world_to_image(camera.image_to_world(far_side))

        far_dist = np.linalg.norm(far_side[0] - far_side[1])

        new_cam.zoom(self.side_in / far_dist)
        new_cam.center_principal_point((self.side_in, self.side_in))

        if self.geometry:
            new_cam.zoom(random_zoom)

        if do_flip:
            new_cam.horizontal_flip()

        image = plt.imread(image_path)
        image = cameralib.reproject_image(image, camera, new_cam, (self.side_in, self.side_in))

        return image, new_cam


    def parse_sample(self, sample):
        do_flip = (not self.at_test) and (np.random.rand() < 0.5)

        random_zoom = np.random.uniform(self.random_zoom, self.random_zoom ** (-1))

        color_image, new_color_cam = self.get_input_image(sample['image'], sample['camera'], sample['bbox'], do_flip, random_zoom)

        color_image = self.transform(random_color(color_image) if self.colour else color_image.copy())

        world_coords = sample['skeleton']
        camera_coords = new_color_cam.world_to_camera(world_coords)
        valid = sample['valid']

        if do_flip:
            camera_coords = camera_coords[self.data_info.mirror]
            valid = valid[self.data_info.mirror]

        if self.at_test:
            back_rotate = sample['camera'].R @ new_color_cam.R.T

            return color_image, camera_coords, valid, back_rotate
        else:
            return color_image, camera_coords, valid


    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])


    def __len__(self):
        return len(self.samples)


def show_mat(image_coord, ax, bbox = None):
    '''
    Shows skeleton(mat)

    Args:
        image_coord: (21, 2)
    '''
    from joint_settings import h36m_short_names as short_names
    from joint_settings import h36m_parent as parent

    mapper = dict(zip(short_names, range(len(short_names))))

    body_edges = [mapper[parent[name]] for name in short_names]
    body_edges = np.hstack(
        [
            np.arange(len(body_edges)).reshape(-1, 1),
            np.array(body_edges).reshape(-1, 1)
        ]
    )
    ax.plot(image_coord[:, 0], image_coord[:, 1], '.', color = 'yellow')

    for edge in body_edges:
        ax.plot(image_coord[edge, 0], image_coord[edge, 1], '--', color = 'b')

    if bbox is not None:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(rect)
