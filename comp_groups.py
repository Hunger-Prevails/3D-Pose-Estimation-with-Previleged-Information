import os
import jpeg4py
import json
import cv2
import copy
import json
import numpy as np
import multiprocessing

from utils import JointInfo
from mat_utils import MatSample
from utils import PoseGroup


def get_mpii_group(phase, args):

	assert os.path.isdir(args.comp_down)
	assert phase in ['train', 'valid']

	from joint_settings import mpii_short_names as short_names
	from joint_settings import mpii_parent as parent
	from joint_settings import mpii_mirror as mirror
	from joint_settings import mpii_base_joint as base_joint

	mapper = dict(zip(short_names, range(len(short_names))))
	
	map_mirror = [mapper[mirror[name]] for name in short_names if name in mirror]
	map_parent = [mapper[parent[name]] for name in short_names if name in parent]

	_mirror = np.arange(len(short_names))
	_parent = np.arange(len(short_names))

	_mirror[np.array([name in mirror for name in short_names])] = np.array(map_mirror)
	_parent[np.array([name in parent for name in short_names])] = np.array(map_parent)

	joint_info = JointInfo(short_names, _parent, _mirror, mapper[base_joint])

	valid_images = os.path.join(args.comp_path, 'valid_images.txt')
	valid_images = [line for line in open(valid_images)]

	release = json.load(open('mpii_human_pose.json'))

	if phase == 'valid':
		release = [annotation for annotation in release if annotation['image'] in valid_images]
	else:
		release = [annotation for annotation in release if annotation['image'] not in valid_images]

	processes = []

	pool = multiprocessing.Pool(args.num_processes)

	for aid, annotation in enumerate(release):

		print 'collecting annotation [', str(aid) + '/' + str(len(release)), ']'
		
		image_path = os.path.join(args.comp_path, 'images', annotation['image'])

		for sing in annotation['singles']:

			sample = annotation['samples'][sing]

			down_path = os.path.join(args.comp_down, str(aid) + '_' + str(sing) + '.jpg')

			if sample:
				data_params = (down_path, args.side_in, args.random_zoom, args.box_margin)
				processes.append(pool.apply_async(func = make_sample, args = (sample, image_path, data_params)))

	pool.close()
	pool.join()
	samples = [process.get() for process in processes]

	return PoseGroup(phase, joint_info, [sample for sample in samples if sample])


def coord_to_box(image_coord, box_margin, border, scale):
	'''
	params
		image_coord: (19 x 3) joint coords in image space with confidence scores
	returns
		image_box: (4,) pseudo bounding box of the person
	'''
	valid = image_coord[np.where(0 != image_coord[:, 2])]

	x_min = np.amin(valid[:, 0])
	x_max = np.amax(valid[:, 0])
	y_min = np.amin(valid[:, 1])
	y_max = np.amax(valid[:, 1])

	center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
	shape = np.array([x_max - x_min, y_max - y_min])

	pad = max(scale * 180 - np.max(shape) / box_margin, 0) * (1 - box_margin)
	shape = shape / box_margin + pad

	begin = np.maximum(center - shape / 2, np.zeros(2))
	end = np.minimum(center + shape / 2, border)

	return np.hstack([begin, end - begin])


def make_sample(sample, image_path, data_params):

	down_path, side_in, random_zoom, box_margin = data_params

	image_coords = np.array(sample['joints']).reshape((16, 3))

	image = jpeg4py.JPEG(image_path).decode()

	border = np.array(image.shape[:2])[::-1]

	image_coords[:, 2] *= (0 <= image_coords[:, 0])
	image_coords[:, 2] *= (0 <= image_coords[:, 1])
	image_coords[:, 2] *= (image_coords[:, 0] < border[0])
	image_coords[:, 2] *= (image_coords[:, 1] < border[1])

	bbox = coord_to_box(image_coords, box_margin, border, sample['scale'])

	box_center = bbox[:2] + bbox[2:] / 2

	expand_side = int(np.round(np.sum(bbox[2:] ** 2) ** 0.5))

	view_begin = (box_center - expand_side / 2).astype(np.int)
	view_end = view_begin + expand_side

	crop_begin = np.maximum(view_begin, np.zeros(2).astype(np.int))
	crop_end = np.minimum(view_end, border)

	dest_begin = crop_begin - view_begin
	dest_end = crop_end - view_begin

	scale_factor = min(side_in / np.max(bbox[2:]) / random_zoom, 1.0)

	dest_side = int(np.round(scale_factor * expand_side))

	bbox[:2] -= view_begin
	image_coords[:, :2] -= view_begin

	bbox *= scale_factor
	image_coords[:, :2] *= scale_factor

	if not os.path.exists(down_path):

		view_patch = np.zeros((expand_side, expand_side, 3)).astype(np.uint8)

		view_patch[dest_begin[1]:dest_end[1], dest_begin[0]:dest_end[0]] = image[crop_begin[1]:crop_end[1], crop_begin[0]:crop_end[0]]

		save_patch = cv2.resize(view_patch, (dest_side, dest_side))

		cv2.imwrite(down_path, save_patch[:, :, ::-1])

	return MatSample(down_path, image_coords, bbox)
