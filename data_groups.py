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


def make_sample(paths, annos, args):
	'''
	params
		bbox: (4,) bounding box in original camera view
		body_pose: (19 x 3) joint coords in world space
		image_coord: (19 x 3) joint coords in image space with confidence scores
		image_path: path to image under original camera view
	returns
		pose sample with path to down-scaled image and corresponding box/image_coord
	'''
	image_path, new_path = paths
	image_coord, body_pose, camera = annos

	border = np.array([1920, 1080])

	cond1 = np.all(0 <= image_coord[:, :2], axis = 1)
	cond2 = np.all(image_coord[:, :2] < border, axis = 1)

	contains = cond1 & cond2 & (image_coord[:, 2] != -1)

	valid = (args.thresh_confid <= image_coord[:, 2]) & contains if args.confid_filter else contains

	if np.sum(valid) < args.num_valid:
		return None

	mass_center = np.array([np.mean(body_pose[valid, 0]), np.mean(body_pose[valid, 2])])
	entry_center = np.array([30.35, -254.3])

	if np.linalg.norm(mass_center - entry_center) <= 80:
		return None

	bbox = coord_to_box(image_coord[contains, :2], args.box_margin, border)

	expand_side = np.sum((bbox[2:] / args.random_zoom) ** 2) ** 0.5

	box_center = bbox[:2] + bbox[2:] / 2

	scale_factor = min(args.side_in / np.max(bbox[2:]) / args.random_zoom, 1.0)

	dest_side = int(np.round(expand_side * scale_factor))

	new_camera = copy.deepcopy(camera)
	new_camera.shift_to_center(box_center, (expand_side, expand_side))
	new_camera.scale_output(scale_factor)

	new_bbox = cameralib.reproject_points(bbox[None, :2], camera, new_camera)[0]

	new_bbox = np.concatenate((new_bbox, bbox[2:] * scale_factor))

	if not os.path.exists(new_path):

		image = jpeg4py.JPEG(image_path).decode()

		new_image = cameralib.reproject_image(image, camera, new_camera, (dest_side, dest_side))

		cv2.imwrite(new_path, new_image[:, :, ::-1])

	return PoseSample(new_path, body_pose, valid, new_bbox, new_camera)


def coord_to_box(valid_coord, box_margin, border):
	'''
	params
		image_coord: (k x 3) valid joint coords in image space
	returns
		image_box: (4,) pseudo bounding box of the person
	'''
	x_min = np.amin(valid_coord[:, 0])
	x_max = np.amax(valid_coord[:, 0])
	y_min = np.amin(valid_coord[:, 1])
	y_max = np.amax(valid_coord[:, 1])

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
			'170221_haggling_b1',
			'170221_haggling_b2',
			'170221_haggling_b3',
			'170221_haggling_m1',
			'170221_haggling_m2',
			'170221_haggling_m3',
			'170224_haggling_a2',
			'170224_haggling_a3',
			'170224_haggling_b1',
			'170224_haggling_b2',
			'170224_haggling_b3',
			'170228_haggling_a1',
			'170228_haggling_a2',
			'170228_haggling_a3',
			'170228_haggling_b1',
			'170228_haggling_b2',
			'170228_haggling_b3',
			'170404_haggling_a1',
			'170404_haggling_a2',
			'170404_haggling_a3',
			'170404_haggling_b1',
			'170404_haggling_b2',
			'170404_haggling_b3',
			'170407_haggling_a1',
			'170407_haggling_a2',
			'170407_haggling_a3',
			'170407_haggling_b1',
			'170407_haggling_b2',
			'170407_haggling_b3',
			'171026_pose1',
			'171026_pose2',
			'171026_pose3',
			'171204_pose1',
			'171204_pose2',
			'171204_pose3',
			'171204_pose4',
			'171204_pose5',
			'171204_pose6'
		],
		valid = [
			'160224_haggling1',
			'160226_haggling1'
		],
		test = [
			'160422_haggling1',
			'161202_haggling1'
		]
	)
	frame_step = dict(
		train = 10,
		valid = 10,
		test = 50
	)
	processes = []

	pool = multiprocessing.Pool(args.num_processes)

	for sequence in sequences[phase]:

		root_seq = os.path.join(args.data_root_path, sequence)

		try:
			assert os.path.exists(root_seq)
		except:
			root_seq = root_seq.replace('cmu', 'new')

		root_image = os.path.join(root_seq, 'hdImgs')

		cam_folders = [os.path.join(root_image, folder) for folder in os.listdir(root_image)]
		cam_folders = [folder for folder in cam_folders if os.path.isdir(folder)]
		cam_folders.sort()

		cam_names = [os.path.basename(folder) for folder in cam_folders]

		cam_files = [os.path.join(root_image, 'image_coord_' + cam_name + '.json') for cam_name in cam_names]
		cam_files = [json.load(open(file)) for file in cam_files]

		down_path = [os.path.join(args.data_down_path, sequence + '.' + cam_name) for cam_name in cam_names]

		start_frame = cam_files[0]['start_frame']
		end_frame = cam_files[0]['end_frame']
		interval = cam_files[0]['interval']

		cam_folders = dict(zip(cam_names, cam_folders))
		cam_files = dict(zip(cam_names, cam_files))		
		down_path = dict(zip(cam_names, down_path))
		
		cameras = get_cameras(os.path.join(root_seq, 'calibration_' + sequence + '.json'), cam_names)
		
		pose_idx = 0

		root_skeleton = os.path.join(root_seq, 'hdPose3d_stage1_coco19')

		prev_pose = dict()

		for frame_idx, frame in enumerate(xrange(start_frame, end_frame, interval)):

			bodies = os.path.join(root_skeleton, 'body3DScene_' + str(frame).zfill(8) + '.json')
			bodies = json.load(open(bodies))['bodies']

			if not bodies:
				print 'empty frame skipped'
				continue

			for body_pose in bodies:

				if (frame - start_frame) % frame_step[phase] != 0:
					pose_idx += 1
					continue

				body_id = body_pose['id']

				body_pose = np.array(body_pose['joints19']).reshape((-1, 4))[:, :3]

				if args.static_filter and body_id in prev_pose:

					displacement = np.linalg.norm(prev_pose[body_id] - body_pose, axis = 1)

					if np.all(displacement < args.thresh_static):
						print 'static pose skipped', body_id
						pose_idx += 1
						continue

				for cam_name in cam_names:

					image_path = os.path.join(cam_folders[cam_name], cam_name + '_' + str(frame).zfill(8) + '.jpg')

					if not os.path.exists(image_path):
						continue

					if not os.path.exists(down_path[cam_name]):
						os.mkdir(down_path[cam_name])

					image_coord = np.array(cam_files[cam_name]['image_coord'][pose_idx])

					new_path = os.path.join(down_path[cam_name], str(frame) + '.' + str(body_id) + '.jpg')

					paths = (image_path, new_path)
					annos = (image_coord, body_pose, cameras[cam_name])

					processes.append(pool.apply_async(func = make_sample, args = (paths, annos, args)))

				prev_pose[body_id] = body_pose

				pose_idx += 1

			print 'collecting samples [', str(frame_idx) + '/' + str((end_frame - start_frame) / interval), '] sequence', sequence

	pool.close()
	pool.join()
	samples = [process.get() for process in processes]

	return PoseGroup(phase, data_info, [sample for sample in samples if sample])
