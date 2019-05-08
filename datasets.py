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
import augmentation
import torch.utils.data as data

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import datasets, transforms


def get_train_loader(args):
    data_group = getattr(data_groups, 'get_' + args.data_source + '_group')('train', args)

    dataset = TrainSet(data_group, args)

    return DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = args.shuffle,
            num_workers = args.workers,
            pin_memory = True), data_group.joint_info


def get_test_loader(args, phase):
    data_group = getattr(data_groups, 'get_' + args.data_source + '_group')(phase, args)

    dataset = TestSet(data_group, args)

    return DataLoader(
            dataset,
            batch_size = args.batch_size,
            shuffle = args.shuffle,
            num_workers = args.workers,
            pin_memory = True), data_group.joint_info


class TrainSet(data.Dataset):
    def __init__(self, pose_group, args):

        assert pose_group.phase == 'train'

        self.crop_factor = args.crop_factor_train
        self.side_eingabe = args.side_eingabe
        self.univ_skeleton = args.univ_skeleton
        self.do_perturbate = args.do_perturbate
        self.do_occlude = args.do_occlude
        self.chance_occlude = args.chance_occlude
        self.occ_path = args.occluder_path
        self.random_zoom = args.random_zoom

        self.occ_count = torch.load(os.path.join(self.occ_path, 'count.pth'))['count']
        
        self.joint_info = pose_group.joint_info
        self.samples = pose_group.samples

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

    def parse_train_sample(self, sample):

        center = sample.bbox[:2] + sample.bbox[2:] / 2
        
        width = np.array([sample.bbox[2] / 2, 0])
        height = np.array([0, sample.bbox[3] / 2])

        if self.do_perturbate:
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

        camera.zoom(self.side_eingabe / side_crop * self.crop_factor)
        camera.center_principal((self.side_eingabe, self.side_eingabe))

        if self.do_perturbate:
            camera.zoom(np.random.uniform(self.random_zoom, self.random_zoom ** (-1)))
            camera.rotate(roll = np.random.uniform(- np.pi / 6, np.pi / 6))

        world_coords = sample.univ_coords if self.univ_skeleton else sample.world_coords

        if np.random.rand() < 0.5:
            camera.horizontal_flip()
            camera_coords = camera.world_to_camera(world_coords)[self.joint_info.mirror]
        else:
            camera_coords = camera.world_to_camera(world_coords)

        image_coords = camera.camera_to_image(camera_coords)

        image = jpeg4py.JPEG(sample.image_path).decode()
        image = cameralib.reproject_image(image, sample.camera, camera, (self.side_eingabe, self.side_eingabe))
        image = self.transform(self.occlusion_augment(image)) if self.do_occlude else self.transform(image)

        inv_intrinsics = np.linalg.inv(camera.intrinsic_matrix).astype(np.float32)

        return image, camera_coords, inv_intrinsics

    def occlusion_augment(self, image):
        random_value = np.random.uniform()

        if random_value < self.chance_occlude / 2:
            image = augmentation.augment_object(image, np.random.choice(self.occ_count), self.occ_path)
        elif random_value < self.chance_occlude:
            image = augmentation.random_erase(image)

        return augmentation.augment_color(image)

    def __getitem__(self, index):
        return self.parse_train_sample(self.samples[index])

    def __len__(self):
        return len(self.samples)


class TestSet(data.Dataset):
    def __init__(self, pose_group, args):
        
        assert pose_group.phase != 'train'

        self.crop_factor = args.crop_factor_test
        self.side_eingabe = args.side_eingabe
        self.univ_skeleton = args.univ_skeleton

        self.joint_info = pose_group.joint_info
        self.samples = pose_group.samples

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])

    def parse_test_sample(self, sample):
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

        camera.zoom(self.side_eingabe / side_crop * self.crop_factor)
        camera.center_principal((self.side_eingabe, self.side_eingabe))

        world_coords = sample.univ_coords if self.univ_skeleton else sample.world_coords

        camera_coords = camera.world_to_camera(world_coords)
        image_coords = camera.camera_to_image(camera_coords)

        image = jpeg4py.JPEG(sample.image_path).decode()
        image = cameralib.reproject_image(image, sample.camera, camera, (self.side_eingabe, self.side_eingabe))
        image = self.transform(image.copy())

        inv_intrinsics = np.linalg.inv(camera.intrinsic_matrix).astype(np.float32)

        back_rotation = np.matmul(sample.camera.R, camera.R.T)

        return image, camera_coords, inv_intrinsics, back_rotation

    def __getitem__(self, index):
        return self.parse_test_sample(self.samples[index])

    def __len__(self):
        return len(self.samples)
