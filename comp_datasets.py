import os
import cv2
import copy
import jpeg4py
import numpy as np
import comp_groups
import torch
import mat_utils
import joint_settings
import torch.utils.data as data

from mat_utils import Mapper
from torchvision import datasets
from torchvision import transforms
from augment_colour import augment_color


def get_comp_loader(args, phase, dest_info):
    data_info, samples = getattr(comp_groups, 'get_' + args.comp_name + '_group')(phase, args)

    match = getattr(joint_settings, args.comp_name + '_' + args.data_name + '_match')

    mapper = Mapper(data_info, dest_info, match)

    dataset = Lecture(data_info, samples, mapper, args) if phase == 'train' else Exam(samples, mapper, args)

    shuffle = args.shuffle if phase == 'train' else False

    return data.DataLoader(dataset, args.batch_size, shuffle, num_workers = args.workers, pin_memory = True)


class Lecture(data.Dataset):

    def __init__(self, data_info, samples, mapper, args):

        self.data_info = data_info
        self.samples = samples
        self.mapper = mapper

        self.random_zoom = args.random_zoom
        self.side_in = args.side_in

        self.geometry = args.geometry
        self.colour = args.colour
        self.extra_channel = args.extra_channel

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])


    def parse_sample(self, sample):
        image_coords = sample.image_coords

        image = jpeg4py.JPEG(sample.image_path).decode()

        border = np.array(image.shape[:2])[::-1]

        if np.random.rand() < 0.5:

            image = image[:, ::-1].copy()

            image_coords[:, 0] = border[0] - image_coords[:, 0]

            image_coords = image_coords[self.data_info.mirror]

        roi_center = sample.bbox[:2] + sample.bbox[2:] / 2
        
        far_dist = np.amax(sample.bbox[2:])

        near_dist = np.amin(sample.bbox[2:])

        vertical = np.argmin(sample.bbox[2:]) == 0

        if self.geometry:
            image, new_coords = mat_utils.rand_rotate(center = roi_center, image = image, points = image_coords[:, :2], max_radian = np.pi / 9)

            image_coords = np.hstack(new_coords, image_coords[:, 2:])

            far_dist *= np.random.uniform(self.random_zoom, self.random_zoom ** (-1))

        roi_begin = (roi_center - far_dist / 2).astype(np.int)

        far_dist = int(np.round(far_dist))

        roi_end = roi_begin + far_dist

        scale_factor = self.side_in / float(far_dist)

        image_coords[:, :2] = (image_coords[:, :2] - roi_begin) * scale_factor

        image = cv2.resize(image[roi_begin[1]:roi_end[1], roi_begin[0]:roi_end[0]], (self.side_in, self.side_in))

        feed_in = self.transform(augment_color(image)) if self.colour else self.transform(image)

        if self.extra_channel:

            channel = np.zeros((self.side_in, self.side_in, 1))

            near_dist *= scale_factor

            near_in = int(np.round((self.side_in - near_dist) / 2.0))
            near_out = int(np.round((self.side_in + near_dist) / 2.0))

            if vertical:
                channel[:, near_in:near_out] = 1.0
            else:
                channel[near_in:near_out, :] = 1.0

            feed_in = np.concatenate([feed_in, channel.transpose(2, 0, 1)])

        image_coords = self.mapper.map_coord(image_coords)

        return feed_in, np.float32(image_coords[:, :2]), np.uint8(image_coords[:, 2] != 0)


    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])


    def __len__(self):
        return len(self.samples)


class Exam(data.Dataset):

    def __init__(self, samples, mapper, args):

        self.samples = samples
        self.mapper = mapper

        self.side_in = args.side_in

        self.mean = [0.485, 0.456, 0.406]
        self.dev = [0.229, 0.224, 0.225]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.dev)])


    def parse_sample(self, sample):
        image_coords = sample.image_coords

        image = jpeg4py.JPEG(sample.image_path).decode()

        border = np.array(image.shape[:2])[::-1]

        roi_center = sample.bbox[:2] + sample.bbox[2:] / 2

        far_dist = np.amax(sample.bbox[2:])

        roi_begin = (roi_center - far_dist / 2).astype(np.int)

        far_dist = int(np.round(far_dist))

        roi_end = roi_begin + far_dist

        scale_factor = self.side_in / float(far_dist)

        image_coords[:, :2] = (image_coords[:, :2] - roi_begin) * scale_factor

        feed_in = cv2.resize(image[roi_begin[1]:roi_end[1], roi_begin[0]:roi_end[0]], (self.side_in, self.side_in))

        image_coords = self.mapper.map_coord(image_coords)

        return self.transform(feed_in), image_coords[:, :2], np.uint8(image_coords[:, 2] != 0)


    def __getitem__(self, index):
        return self.parse_sample(self.samples[index])


    def __len__(self):
        return len(self.samples)
