import os
import cv2
import copy
import jpeg4py
import numpy as np
import random
import data_groups
import cameralib
import torch
import utils
import torch.utils.data as data

from torchvision import datasets
from torchvision import transforms
from augment_colour import random_color
from augment_occluder import random_erase
from augment_occluder import random_occlu


def get_data_loader(args, phase):
    data_info, samples = getattr(data_groups, 'get_' + args.data_name + '_group')(phase, args)

    dataset = Lecture(data_info, samples, args) if phase == 'train' else Exam(samples, args)

    shuffle = args.shuffle if phase == 'train' else False

    data_loader = data.DataLoader(dataset, args.batch_size, shuffle, num_workers = args.workers, pin_memory = True)

    return data_loader, data_info


class Lecture(data.Dataset):

    def __init__(self, data_info, samples, args):

        self.data_info = data_info
        self.samples = samples

        self.side_in = args.side_in
        self.random_zoom = args.random_zoom
        self.joint_space = args.joint_space
        self.extra_channel = args.extra_channel

        self.geometry = args.geometry
        self.colour = args.colour
        self.eraser = args.eraser

        self.occluder = args.occluder
        self.occ_path = args.occ_path
        self.occ_count = torch.load(os.path.join(self.occ_path, 'count.pth'))['count']

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

    def parse_sample(self, sample):

        center = sample.bbox[:2] + sample.bbox[2:] / 2
        
        width = np.array([sample.bbox[2] / 2, 0])
        height = np.array([0, sample.bbox[3] / 2])

        if sample.bbox[2] < sample.bbox[3]:
            vertical = True
            near_side = np.stack([center - width, center + width])
            far_side = np.stack([center - height, center + height])
        else:
            vertical = False
            far_side = np.stack([center - width, center + width])
            near_side = np.stack([center - height, center + height])

        camera = copy.deepcopy(sample.camera)
        camera.turn_towards(center)
        camera.undistort()
        camera.square_pixels()

        far_side = camera.world_to_image(sample.camera.image_to_world(far_side))

        far_dist = np.linalg.norm(far_side[0] - far_side[1])

        camera.zoom(self.side_in / far_dist)
        camera.center_principal((self.side_in, self.side_in))

        if self.geometry:
            camera.zoom(np.random.uniform(self.random_zoom, self.random_zoom ** (-1)))
            camera.rotate(roll = np.random.uniform(- np.pi / 9, np.pi / 9))

        world_coords = sample.body_pose

        if np.random.rand() < 0.5:
            camera.horizontal_flip()
            camera_coords = camera.world_to_camera(world_coords)[self.data_info.mirror]
            sample.valid = sample.valid[self.data_info.mirror]
        else:
            camera_coords = camera.world_to_camera(world_coords)

        image_coords = camera.camera_to_image(camera_coords)

        image = jpeg4py.JPEG(sample.image_path).decode()

        image = cameralib.reproject_image(image, sample.camera, camera, (self.side_in, self.side_in))

        image = self.transform(self.augment(image))

        if self.extra_channel:
            channel = np.zeros((self.side_in, self.side_in, 1))

            near_side = camera.world_to_image(sample.camera.image_to_world(near_side))
            near_dist = np.linalg.norm(near_side[0] - near_side[1])

            near_in = int(np.round((self.side_in - near_dist) / 2.0))
            near_out = int(np.round((self.side_in + near_dist) / 2.0))

            if vertical:
                channel[:, near_in:near_out] = 1.0
            else:
                channel[near_in:near_out, :] = 1.0

            image = np.concatenate([image, channel.transpose(2, 0, 1)])

        if self.joint_space:
            return image, camera_coords, image_coords, np.uint8(sample.valid), camera.intrinsics
        else:
            return image, camera_coords, np.uint8(sample.valid)

    def augment(self, image):

        if self.occluder and np.random.uniform() < 0.2:
            image = random_occlu(image, self.occ_count, self.occ_path)

        if self.eraser and np.random.uniform() < 0.2:
            image = random_erase(image)

        if self.colour and np.random.uniform() < 0.9:
            image = random_color(image)

        return image

    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])

    def __len__(self):
        return len(self.samples)


class Exam(data.Dataset):

    def __init__(self, samples, args):

        self.side_in = args.side_in
        self.joint_space = args.joint_space
        self.extra_channel = args.extra_channel

        self.samples = samples

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

    def parse_sample(self, sample):
        center = sample.bbox[:2] + sample.bbox[2:] / 2
        
        width = np.array([sample.bbox[2] / 2, 0])
        height = np.array([0, sample.bbox[3] / 2])

        if sample.bbox[2] < sample.bbox[3]:
            vertical = True
            near_side = np.stack([center - width, center + width])
            far_side = np.stack([center - height, center + height])
        else:
            vertical = False
            far_side = np.stack([center - width, center + width])
            near_side = np.stack([center - height, center + height])

        camera = copy.deepcopy(sample.camera)
        camera.turn_towards(center)
        camera.undistort()
        camera.square_pixels()

        far_side = camera.world_to_image(sample.camera.image_to_world(far_side))

        far_dist = np.linalg.norm(far_side[0] - far_side[1])

        camera.zoom(self.side_in / far_dist)
        camera.center_principal((self.side_in, self.side_in))

        world_coords = sample.body_pose

        camera_coords = camera.world_to_camera(world_coords)
        image_coords = camera.camera_to_image(camera_coords)

        image = jpeg4py.JPEG(sample.image_path).decode()
        image = cameralib.reproject_image(image, sample.camera, camera, (self.side_in, self.side_in))
        image = self.transform(image.copy())

        if self.extra_channel:
            channel = np.zeros((self.side_in, self.side_in, 1))

            near_side = camera.world_to_image(sample.camera.image_to_world(near_side))
            near_dist = np.linalg.norm(near_side[0] - near_side[1])

            near_in = int(np.round((self.side_in - near_dist) / 2.0))
            near_out = int(np.round((self.side_in + near_dist) / 2.0))

            if vertical:
                channel[:, near_in:near_out] = 1.0
            else:
                channel[near_in:near_out, :] = 1.0

            image = np.concatenate([image, channel.transpose(2, 0, 1)])

        back_rotation = sample.camera.R @ camera.R.T

        if self.joint_space:
            return image, camera_coords, image_coords, back_rotation, np.uint8(sample.valid), camera.intrinsics
        else:
            return image, camera_coords, back_rotation, np.uint8(sample.valid)

    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])

    def __len__(self):
        return len(self.samples)
