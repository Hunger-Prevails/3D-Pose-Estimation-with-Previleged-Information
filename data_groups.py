import os
import jpeg4py
import json
import cv2
import copy
import numpy as np
import cameralib
import multiprocessing

from utils import JointInfo
from utils import PoseSample
from utils import PoseGroup


def get_cameras(json_file, cam_names):
	
	calibration = json.load(open(json_file))

	cameras = [cam for cam in calibration['cameras'] if cam['panel'] == 0]

	return dict(
			[
				(
					cam['name'],
					cameralib.Camera(
							np.matmul(np.array(cam['R']).T, - np.array(cam['t'])),
							np.array(cam['R']),
							np.array(cam['K']),
							np.array(cam['distCoef']),
							(0, -1, 0)
					)
				) for cam in cameras if cam['name'] in cam_names
			]
		)


def make_sample(data_sample, data_params):
	'''
	params
		bbox: (4,) bounding box in original camera view
		body_pose: (19 x 3) joint coords in world space
		image_coord: (19 x 3) joint coords in image space with confidence scores
		image_path: path to image under original camera view
	returns
		pose sample with path to down-scaled image and corresponding box/image_coord
	'''
	image_path, image_coord, body_pose, camera = data_sample
	side_in, random_zoom, box_margin, essence, folder_down = data_params

	border = np.array([1920, 1080])

	cond1 = np.all(0 <= image_coord[:, :2], axis = 1)
	cond2 = np.all(image_coord[:, :2] <= border, axis = 1)

	image_coord[np.where(~(cond1 & cond2)), 2] = -1

	try:
		assert np.all(image_coord[essence, 2] != -1)
	except:
		return None

	bbox = coord_to_box(image_coord, box_margin, border)

	expand_side = np.sum(bbox[2:] ** 2) ** 0.5

	box_center = bbox[:2] + bbox[2:] / 2

	scale_factor = min(side_in / np.max(bbox[2:]) / random_zoom, 1.0)

	dest_side = int(np.round(expand_side * scale_factor))

	new_camera = copy.deepcopy(camera)
	new_camera.shift_to_center(box_center, (expand_side, expand_side))
	new_camera.scale_output(scale_factor)

	new_path = os.path.join(folder_down, os.path.basename(image_path))

	new_bbox = cameralib.reproject_points(bbox[None, :2], camera, new_camera)[0]

	new_bbox = np.concatenate((new_bbox, bbox[2:] * scale_factor))

	new_coord = cameralib.reproject_points(image_coord[:, :2], camera, new_camera)

	new_coord = np.concatenate((new_coord, image_coord[:, 2:]), axis = 1)

	if not os.path.exists(new_path):

		image = jpeg4py.JPEG(image_path).decode()

		new_image = cameralib.reproject_image(image, camera, new_camera, (dest_side, dest_side))

		cv2.imwrite(new_path, new_image[:, :, ::-1])

	return PoseSample(new_path, body_pose, new_coord, new_bbox, new_camera)


def coord_to_box(image_coord, box_margin, border):
	'''
	params
		image_coord: (19 x 3) joint coords in image space with confidence scores
	returns
		image_box: (4,) pseudo bounding box of the person
	'''
	valid = image_coord[np.where(image_coord[:, 2] != -1)]

	x_min = np.amin(valid[:, 0])
	x_max = np.amax(valid[:, 0])
	y_min = np.amin(valid[:, 1])
	y_max = np.amax(valid[:, 1])

	center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
	shape = np.array([x_max - x_min, y_max - y_min]) / box_margin

	begin = np.maximum(center - shape / 2, np.zeros(2))
	end = np.minimum(center + shape / 2, border)

	return np.hstack([begin, end - begin])


def get_cmu_group(phase, args):

	assert os.path.isdir(args.data_down_path)
	
	from joint_settings import cmu_short_names as short_names
	from joint_settings import cmu_parent as parent
	from joint_settings import cmu_mirror as mirror
	from joint_settings import cmu_base_joint as base_joint
	from joint_settings import cmu_weight as weight
	from joint_settings import cmu_overlook as overlook

	mapper = dict(zip(short_names, range(len(short_names))))
	
	map_mirror = [mapper[mirror[name]] for name in short_names if name in mirror]
	map_parent = [mapper[parent[name]] for name in short_names if name in parent]

	_mirror = np.arange(len(short_names))
	_parent = np.arange(len(short_names))

	_mirror[np.array([name in mirror for name in short_names])] = np.array(map_mirror)
	_parent[np.array([name in parent for name in short_names])] = np.array(map_parent)

	essence = np.array([False if name in overlook else True for name in short_names])

	data_info = JointInfo(short_names, _parent, _mirror, mapper[base_joint], np.array(weight), essence)

	sequences = dict(
		train = [
			'171204_pose1',
			'171204_pose2',
			'171026_pose1',
			'171026_pose2',
			'171204_pose4',
			'171204_pose5',
			'171204_pose6'],
		valid = [
			'171204_pose3'],
		test = [
			'171026_pose3']
	)
	frame_step = dict(
		train = 10,
		valid = 10,
		test = 50
	)
	processes = []

	pool = multiprocessing.Pool(args.num_processes)

	for sequence in sequences[phase]:

		root_sequence = os.path.join(args.data_root_path, sequence)
		root_image = os.path.join(root_sequence, 'hdImgs')

		cam_folders = [os.path.join(root_image, folder) for folder in os.listdir(root_image)]
		cam_folders = [folder for folder in cam_folders if os.path.isdir(folder)]
		cam_folders.sort()

		cam_names = [os.path.basename(folder) for folder in cam_folders]

		cam_files = [os.path.join(root_image, 'image_coord_' + cam_name + '.json') for cam_name in cam_names]
		cam_files = [json.load(open(file)) for file in cam_files]

		down_folders = [os.path.join(args.data_down_path, sequence + '.' + cam_name) for cam_name in cam_names]

		start_frame = cam_files[0]['start_frame']
		end_frame = cam_files[0]['end_frame']
		interval = cam_files[0]['interval']

		cam_folders = dict(zip(cam_names, cam_folders))
		cam_files = dict(zip(cam_names, cam_files))		
		down_folders = dict(zip(cam_names, down_folders))
		
		cameras = get_cameras(os.path.join(root_sequence, 'calibration_' + sequence + '.json'), cam_names)
		
		pose_idx = 0

		root_skeleton = os.path.join(root_sequence, 'hdPose3d_stage1_coco19')

		for frame_idx, frame in enumerate(xrange(start_frame, end_frame, interval)):

			skeleton = os.path.join(root_skeleton, 'body3DScene_' + str(frame).zfill(8) + '.json')
			skeleton = json.load(open(skeleton))['bodies']
			if not skeleton:
				continue

			body_pose = np.array(skeleton[0]['joints19']).reshape((-1, 4))[:, :3]

			for cam_name in cam_names:

				if (frame - start_frame) % frame_step[phase] != 0:
					continue

				if not os.path.exists(down_folders[cam_name]):
					os.mkdir(down_folders[cam_name])

				image_path = os.path.join(cam_folders[cam_name], cam_name + '_' + str(frame).zfill(8) + '.jpg')
				image_coord = np.array(cam_files[cam_name]['image_coord'][pose_idx])

				data_sample = (image_path, image_coord, body_pose, cameras[cam_name])
				data_params = (args.side_in, args.random_zoom, args.box_margin, essence, down_folders[cam_name])

				processes.append(pool.apply_async(func = make_sample, args = (data_sample, data_params)))

			print 'collecting samples [', str(frame_idx) + '/' + str((end_frame - start_frame) / interval), '] sequence', sequence

			pose_idx += 1

	pool.close()
	pool.join()
	samples = [process.get() for process in processes]

	return PoseGroup(phase, data_info, [sample for sample in samples if sample])
