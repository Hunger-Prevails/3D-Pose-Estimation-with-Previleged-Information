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

	assert os.path.isdir(args.comp_down_path)
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

	valid_images = os.path.join(args.comp_root_path, 'valid_images.txt')
	valid_images = [line.strip() for line in open(valid_images)]

	release = json.load(open('mpii_human_pose.json'))

	if phase == 'valid':
		release = [annotation for annotation in release if annotation['image'] in valid_images]
	else:
		release = [annotation for annotation in release if annotation['image'] not in valid_images]

	processes = []

	pool = multiprocessing.Pool(args.num_processes)

	for aid, annotation in enumerate(release):

		print 'collecting annotation [', str(aid) + '/' + str(len(release)), ']'
		
		image_path = os.path.join(args.comp_root_path, 'images', annotation['image'])

		for sid, sample in enumerate(annotation['samples']):
			down_path = os.path.join(args.comp_down_path, str(aid) + '_' + str(sid) + '.jpg')

			if sample:
				processes.append(pool.apply_async(func = make_sample, args = (sample, image_path, down_path, annotation['shape'], args)))

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
	shape = np.array([x_max - x_min, y_max - y_min]) / box_margin

	shape *= max(np.amax(shape), scale * 180) / np.amax(shape)

	begin = np.maximum(center - shape / 2, np.zeros(2))
	end = np.minimum(center + shape / 2, border)

	return np.hstack([begin, end - begin])


def make_sample(sample, image_path, down_path, border, args):

	image_coords = np.array(sample['joints']).reshape((16, 3))

	border = np.array(border[:2])[::-1]

	image_coords[:, 2] *= (0 <= image_coords[:, 0])
	image_coords[:, 2] *= (0 <= image_coords[:, 1])
	image_coords[:, 2] *= (image_coords[:, 0] < border[0])
	image_coords[:, 2] *= (image_coords[:, 1] < border[1])

	bbox = coord_to_box(image_coords, args.box_margin, border, sample['scale'])

	box_center = bbox[:2] + bbox[2:] / 2

	expand_side = int(np.round(np.sum((bbox[2:] / args.random_zoom) ** 2) ** 0.5))

	view_begin = (box_center - expand_side / 2).astype(np.int)
	view_end = view_begin + expand_side

	crop_begin = np.maximum(view_begin, np.zeros(2).astype(np.int))
	crop_end = np.minimum(view_end, border)

	dest_begin = crop_begin - view_begin
	dest_end = crop_end - view_begin

	scale_factor = min(args.side_in / np.max(bbox[2:]) / args.random_zoom, 1.0)

	dest_side = int(np.round(scale_factor * expand_side))

	bbox[:2] -= view_begin
	image_coords[:, :2] -= view_begin

	bbox *= scale_factor
	image_coords[:, :2] *= scale_factor

	if not os.path.exists(down_path):

		view_patch = np.zeros((expand_side, expand_side, 3)).astype(np.uint8)

		image = jpeg4py.JPEG(image_path).decode()

		view_patch[dest_begin[1]:dest_end[1], dest_begin[0]:dest_end[0]] = image[crop_begin[1]:crop_end[1], crop_begin[0]:crop_end[0]]

		save_patch = cv2.resize(view_patch, (dest_side, dest_side))

		cv2.imwrite(down_path, save_patch[:, :, ::-1])

	# show_skeleton(down_path, image_coords[:, :2].T, image_coords[:, 2], bbox = bbox)

	return MatSample(down_path, image_coords, bbox)


def show_skeleton(image, image_coord, confidence, bbox = None):
	'''
	Shows coco19 skeleton(mat)

	Args:
		image: path to image
		image_coord: (2, num_joints)
		confidence: (num_joints,)
	'''
	from joint_settings import mpii_short_names as short_names
	from joint_settings import mpii_parent as parent

	mapper = dict(zip(short_names, range(len(short_names))))

	body_edges = [mapper[parent[name]] for name in short_names if name in parent]
	body_edges = np.hstack(
		[
			np.arange(len(body_edges)).reshape(-1, 1),
			np.array(body_edges).reshape(-1, 1)
		]
	)
	import matplotlib.pyplot as plt
	import matplotlib.patches as patches

	image = plt.imread(image) if isinstance(image, str) else image

	plt.figure(figsize = (15, 15))

	ax = plt.subplot(1, 1, 1)

	plt.title('2D Body Skeleton in comp groups' + str(image.shape))
	plt.imshow(image)

	ax.set_autoscale_on(False)

	valid = (0.1 <= confidence)

	plt.plot(image_coord[0, valid], image_coord[1, valid], '.')

	for edge in body_edges:
		if valid[edge[0]] and valid[edge[1]]:
			plt.plot(image_coord[0, edge], image_coord[1, edge])

	if bbox is not None:
		rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
		ax.add_patch(rect)

	plt.draw()
	plt.show()
