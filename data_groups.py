import os
import h5py
import jpeg4py
import itertools
import boxlib
import json
import cv2
import copy
import utils
import glob
import pickle5 as pickle
import numpy as np
import cameralib
import transforms3d
import multiprocessing
import spacepy.pycdf as pycdf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ElementTree

from utils import JointInfo
from utils import PoseSample


def detect_bbox(image, rect, detector):
	det_bboxes = detector.detect(image)

	ious = np.array([boxlib.iou(rect, bbox) for bbox in det_bboxes])

	if np.all(ious < 0.5):
		return None

	return det_bboxes[np.argmax(ious)]


def make_sample(sample, camera, image, args):
	'''
	Args:
		sample: dict(skeleton = body_pose, valid = valid, image = new_path, bbox = bbox)
	'''
	box_center = boxlib.center(sample['bbox'])

	sine = np.sin(np.pi / 6)
	cosine = np.cos(np.pi / 6)

	expand_shape = np.array([[cosine, sine], [sine, cosine]]) @ sample['bbox'][2:, np.newaxis]
	expand_side = np.max(expand_shape)

	scale_factor = min(args.side_in / np.max(sample['bbox'][2:]) / args.random_zoom, 1.0)

	dest_side = int(np.round(expand_side * scale_factor))

	new_cam = copy.deepcopy(camera)
	new_cam.shift_to_center(box_center, (expand_side, expand_side))
	new_cam.scale_output(scale_factor)

	new_bbox = cameralib.reproject_points(sample['bbox'][None, :2], camera, new_cam)[0]

	new_bbox = np.concatenate([new_bbox, sample['bbox'][2:] * scale_factor])

	if not os.path.exists(sample['image']):

		new_image = cameralib.reproject_image(image, camera, new_cam, (dest_side, dest_side))

		plt.imsave(sample['image'], new_image)

	sample['bbox'] = new_bbox
	sample['camera'] = new_cam

	return sample


def get_cmu_cameras(json_file, cam_names):

	calibration = json.load(open(json_file))

	cameras = [cam for cam in calibration['cameras'] if cam['panel'] == 0]

	return dict(
		[
			(
				cam['name'],
				cameralib.Camera(
						- np.array(cam['R']).T @ np.array(cam['t']),
						np.array(cam['R']),
						np.array(cam['K']),
						np.array(cam['distCoef']),
						(0, -1, 0)
				)
			) for cam in cameras if cam['name'] in cam_names
		]
	)


def get_cmu_group(phase, args):

	assert os.path.isdir(args.data_down_path)

	sequences = dict(
		train = [
			'171026_pose1',
			'171026_pose2',
			'171204_pose1',
			'171204_pose2',
			'171204_pose4',
			'171204_pose5'
		],
		valid = [
			'171204_pose3',
			'171204_pose6'
		],
		test = [
			'171026_pose3'
		]
	)
	frame_step = dict(
		train = 10,
		valid = 10,
		test = 50
	)
	samples = []

	time_window = json.load(open(os.path.join(args.data_root_path, 'time_window.json')))

	for sequence in sequences[phase]:

		root_seq = os.path.join(args.data_root_path, sequence)

		root_image = os.path.join(root_seq, 'hdImgs')

		cam_names = [
			'00_00', '00_03', '00_05', '00_08', '00_09', '00_11', '00_12', '00_14', '00_15', '00_16',
			'00_18', '00_20', '00_21', '00_22', '00_23', '00_24', '00_25', '00_26', '00_27', '00_29'
		]
		cam_names = [cam_name for cam_name in cam_names if os.path.isdir(os.path.join(root_image, cam_name))]

		cam_folders = [os.path.join(root_image, cam_name) for cam_name in cam_names]
		cam_folders = dict(zip(cam_names, cam_folders))

		down_path = [os.path.join(args.data_down_path, sequence + '.' + cam_name) for cam_name in cam_names]
		down_path = dict(zip(cam_names, down_path))
		
		cameras = get_cmu_cameras(os.path.join(root_seq, 'calibration_' + sequence + '.json'), cam_names)
		
		root_skeleton = os.path.join(root_seq, 'hdPose3d_stage1_coco19')

		prev_pose = dict()

		for frame in range(time_window[sequence][0], time_window[sequence][1]):

			bodies = os.path.join(root_skeleton, 'body3DScene_' + str(frame).zfill(8) + '.json')
			bodies = json.load(open(bodies))['bodies']

			if not bodies:
				continue

			for body in bodies:
				body_id = body['id']

				body_pose = np.array(body['joints19']).reshape((-1, 4))

				if body_id in prev_pose:

					displacement = np.linalg.norm(prev_pose[body_id] - body_pose[:, :3], axis = 1)

					if np.all(displacement < 10.0):
						continue

				for cam_name in cam_names:
					image_path = os.path.join(cam_folders[cam_name], cam_name + '_' + str(frame).zfill(8) + '.jpg')

					if not os.path.exists(image_path):
						continue

					if not os.path.exists(down_path[cam_name]):
						os.mkdir(down_path[cam_name])

					image_coord = cameras[cam_name].world_to_image(body_pose[:, :3])

					new_path = os.path.join(down_path[cam_name], str(frame) + '.' + str(body_id) + '.jpg')

					valid = (0.2 <= body_pose[:, 3])

					if near_entry(body_pose[:, :3], valid):
						continue

					bbox = boxlib.bb_of_points(image_coord[valid])

					image = jpeg4py.JPEG(image_path).decode()

					sample = dict(skeleton = body_pose[:, :3], valid = valid, image = new_path, bbox = detect_bbox(image, bbox))

					samples.append(make_sample(sample, cameras[cam_name], image, args))

				prev_pose[body_id] = body_pose[:, :3]

			print('collecting samples [', str(time_window[sequence][0]), '-', str(frame), '-', str(time_window[sequence][1]), '] sequence', sequence)

	with open(os.path.join(args.data_root_path, 'samples.pkl'), 'wb') as file:
		pickle.dump(samples, file)


def load_coords(path, key_foots, stride):

	coords_raw = pycdf.CDF(path)['Pose']
	coords_raw = np.array(coords_raw, np.float32)[0]
	coords_raw = coords_raw.reshape((coords_raw.shape[0], -1, 3))

	return coords_raw.shape[0], coords_raw[::stride, key_foots]


def collect_data(root_part, activity, camera_id, stride):

	from joint_settings import h36m_cam_names as cam_names
	from joint_settings import h36m_key_foots as key_foots

	root_pose = os.path.join(root_part, 'MyPoseFeatures')
	path_coords = os.path.join(root_pose, 'D3_Positions', activity + '.cdf')

	n_frames, body_poses = load_coords(path_coords, key_foots, stride)

	root_image = os.path.join(root_part, 'Images', activity + '.' + cam_names[camera_id])

	image_paths = ['frame_' + str(x).zfill(6) + '.jpg' for x in range(0, n_frames, stride)]
	image_paths = [os.path.join(root_image, path) for path in image_paths]

	path_bbox = os.path.join(root_part, 'BBoxes', activity + '.' + cam_names[camera_id] + '.npy')

	bboxes = np.load(path_bbox)[::stride]

	return image_paths, body_poses / 10.0, bboxes


def get_h36m_cameras(metadata):

	def make_h36m_camera(extrinsics, intrinsics):
		x_angle, y_angle, z_angle = extrinsics[0:3]
		R = transforms3d.euler.euler2mat(x_angle, y_angle, z_angle, 'rxyz')

		t = extrinsics[3:6] / 10.0
		f = intrinsics[:2]
		c = intrinsics[2:4]
		k = intrinsics[4:7]
		p = intrinsics[7:]

		distorts = np.array([k[0], k[1], p[0], p[1], k[2]], np.float32)
		intrinsics = np.array([[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], np.float32)

		return cameralib.Camera(t, R, intrinsics, distorts)

	root = ElementTree.parse(metadata).getroot()

	cam_params_text = root.findall('w0')[0].text

	numbers = np.array([float(x) for x in cam_params_text[1:-1].split(' ')])

	extrinsic = numbers[:264].reshape(4, 11, 6)
	intrinsic = numbers[264:].reshape(4, 9)

	return [
		[
			make_h36m_camera(extrinsic[camera_id, partition], intrinsic[camera_id]) for partition in range(11)
		] for camera_id in range(4)
	]


def get_h36m_group(phase, args):

	assert os.path.isdir(args.data_down_path)

	detector = utils.Detector()

	cameras = globals()['get_' + args.data_name + '_cameras'](os.path.join(args.data_root_path, 'metadata.xml'))

	partitions = dict(
		train = [1, 5, 6, 7, 8],
		valid = [9, 11]
	)
	stride = dict(
		train = 5,
		valid = 64
	)
	def cond(root_path, elem):
		return os.path.isdir(os.path.join(root_path, elem)) and '_' not in elem

	samples = []

	for partition in partitions[phase]:

		root_part = os.path.join(args.data_root_path, 'S' + str(partition))
		root_image = os.path.join(root_part, 'Images')

		activities = [elem for elem in os.listdir(root_image) if cond(root_image, elem)]
		activities = set([elem.split('.')[0] for elem in activities])

		for index, (activity, camera_id) in enumerate(itertools.product(activities, range(4))):

			if partition == 11 and activity == 'Directions' and camera_id == 0:
				continue

			camera = cameras[camera_id][partition - 1]

			print('collecting samples', str(index) + '|' + str(len(activities) * 4), 'partition', partition)

			image_paths, body_poses, bboxes = collect_data(root_part, activity, camera_id, stride[phase])

			down_path = str(partition) + '.' + activity.replace(' ', '-') + '.' + str(camera_id)
			down_path = os.path.join(args.data_down_path, down_path)

			new_paths = [os.path.join(down_path, os.path.basename(path)) for path in image_paths]

			if not os.path.exists(down_path):
				os.mkdir(down_path)

			for image_path, new_path, body_pose, bbox in zip(image_paths, new_paths, body_poses, bboxes):

				image = jpeg4py.JPEG(image_path).decode()

				valid = np.ones(body_pose.shape[0]).astype(np.bool)

				sample = dict(skeleton = body_pose, valid = valid, image = new_path, bbox = detect_bbox(image, bbox, detector))

				if sample['bbox'] is not None:
					samples.append(make_sample(sample, camera, image, args))

	with open(os.path.join(args.data_root_path, 'samples.pkl'), 'wb') as file:
		pickle.dump(samples, file)


def show_skeleton(image, image_coord, confidence, message = '', bbox = None):
	'''
	Shows coco19 skeleton(mat)

	Args:
		image: path to image
		image_coord: (2, num_joints)
		confidence: (num_joints,)
	'''
	image = plt.imread(image) if isinstance(image, str) else image

	plt.figure(figsize = (12, 8))

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
	ax = plt.subplot(1, 1, 1)
	plt.title(message + ':' + str(image.shape))
	plt.imshow(image)
	ax.set_autoscale_on(False)

	valid = (0.1 <= confidence)

	plt.plot(image_coord[0, valid], image_coord[1, valid], '.')

	for edge in body_edges:
		if valid[edge[0]] and valid[edge[1]]:
			plt.plot(image_coord[0, edge], image_coord[1, edge])

	plt.plot(np.mean(image_coord[0, valid]), np.mean(image_coord[1, valid]), 'X', color = 'w')

	if bbox is not None:
		rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
		ax.add_patch(rect)

	plt.draw()
	plt.show()
