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
from augment_colour import augment_color
from augment_occluder import random_erase
from augment_occluder import augment_object


def get_data_loader(args, phase):
    data_group = getattr(data_groups, 'get_' + args.data_name + '_group')(phase, args)

    dataset = Lecture(data_group, args) if phase == 'train' else Exam(data_group, args)

    shuffle = args.shuffle if phase == 'train' else False

    return data.DataLoader(
        dataset,
        batch_size = args.batch_size,
        shuffle = shuffle,
        num_workers = args.workers,
        pin_memory = True
    ), data_group.data_info


class Lecture(data.Dataset):

    def __init__(self, data_group, args):

        assert data_group.phase == 'train'

        self.side_in = args.side_in
        self.random_zoom = args.random_zoom
        self.joint_space = args.joint_space

        self.geometry = args.geometry
        self.colour = args.colour
        self.eraser = args.eraser

        self.occluder = args.occluder
        self.occ_path = args.occluder_path
        self.occ_count = torch.load(os.path.join(self.occ_path, 'count.pth'))['count']
        
        self.data_info = data_group.data_info
        self.samples = data_group.samples

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

    def parse_sample(self, sample):

        center = sample.bbox[:2] + sample.bbox[2:] / 2
        
        width = np.array([sample.bbox[2] / 2, 0])
        height = np.array([0, sample.bbox[3] / 2])

        if self.geometry:
            center += np.random.uniform(-0.05, 0.05, size = 2) * sample.bbox[2:]

        if sample.bbox[2] < sample.bbox[3]:
            box_verge = np.stack([center - height, center + height])
        else:
            box_verge = np.stack([center - width, center + width])

        camera = copy.deepcopy(sample.camera)
        camera.turn_towards(center)
        camera.undistort()
        camera.square_pixels()

        box_verge = sample.camera.image_to_world(box_verge)
        box_verge = camera.world_to_image(box_verge)
        side_crop = np.linalg.norm(box_verge[0] - box_verge[1])

        camera.zoom(self.side_in / side_crop)
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

        if self.joint_space:
            return image, camera_coords, image_coords, np.uint8(sample.valid), camera.intrinsics
        else:
            return image, camera_coords, np.uint8(sample.valid)

    def augment(self, image):

        if self.occluder and np.random.uniform() < 0.2:
            image = augment_object(image, self.occ_count, self.occ_path)

        if self.eraser and np.random.uniform() < 0.2:
            image = random_erase(image)

        if self.colour and np.random.uniform() < 0.5:
            image = augment_color(image)

        return image

    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])

    def __len__(self):
        return len(self.samples)


class Exam(data.Dataset):

    def __init__(self, data_group, args):
        
        assert data_group.phase != 'train'

        self.side_in = args.side_in
        self.joint_space = args.joint_space

        self.samples = data_group.samples

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
            box_verge = np.stack([center - height, center + height])
        else:
            box_verge = np.stack([center - width, center + width])

        camera = copy.deepcopy(sample.camera)
        camera.turn_towards(center)
        camera.undistort()
        camera.square_pixels()

        box_verge = sample.camera.image_to_world(box_verge)
        box_verge = camera.world_to_image(box_verge)
        side_crop = np.linalg.norm(box_verge[0] - box_verge[1])

        camera.zoom(self.side_in / side_crop)
        camera.center_principal((self.side_in, self.side_in))

        world_coords = sample.body_pose

        camera_coords = camera.world_to_camera(world_coords)
        image_coords = camera.camera_to_image(camera_coords)

        image = jpeg4py.JPEG(sample.image_path).decode()
        image = cameralib.reproject_image(image, sample.camera, camera, (self.side_in, self.side_in))
        image = self.transform(image.copy())

        back_rotation = np.matmul(sample.camera.R, camera.R.T)

        if self.joint_space:
            return image, camera_coords, image_coords, back_rotation, np.uint8(sample.valid), camera.intrinsics
        else:
            return image, camera_coords, back_rotation, np.uint8(sample.valid)

    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])

    def __len__(self):
        return len(self.samples)
