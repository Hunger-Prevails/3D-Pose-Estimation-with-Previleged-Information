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
from builtins import zip as xzip


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
		image_coord: (19 x 3) joint coords in image space with confidence scores
		body_pose: (19 x 3) joint coords in world space
	returns
		pose sample with path to down-scaled image and corresponding box/image_coord
	'''
	image_path, new_path = paths
	image_coord, body_pose, camera = annos

	border = np.array([1920, 1080])

	cond1 = np.all(0 <= image_coord[:, :2], axis = 1)
	cond2 = np.all(image_coord[:, :2] < border, axis = 1)

	cond3 = cond1 & cond2 & (image_coord[:, 2] != -1)

	valid = (args.thresh_confid <= image_coord[:, 2]) & cond3 if args.confid_filter else cond3

	if np.sum(valid) < args.num_valid:
		return None

	mass_center = np.array([np.mean(body_pose[valid, 0]), np.mean(body_pose[valid, 2])])
	entry_center = np.array([30.35, -254.3])

	if np.linalg.norm(mass_center - entry_center) <= 80:
		return None

	bbox = coord_to_box(image_coord[cond3, :2], args.box_margin, border)

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

	mapper = dict(zip(short_names, range(len(short_names))))
	
	map_mirror = [mapper[mirror[name]] for name in short_names if name in mirror]
	map_parent = [mapper[parent[name]] for name in short_names if name in parent]

	_mirror = np.arange(len(short_names))
	_parent = np.arange(len(short_names))

	_mirror[np.array([name in mirror for name in short_names])] = np.array(map_mirror)
	_parent[np.array([name in parent for name in short_names])] = np.array(map_parent)

	data_info = JointInfo(short_names, _parent, _mirror, mapper[base_joint])

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

		cam_names = [
			'00_00', '00_03', '00_05', '00_08', '00_09', '00_11', '00_12', '00_14', '00_15', '00_16',
			'00_18', '00_20', '00_21', '00_22', '00_23', '00_24', '00_25', '00_26', '00_27', '00_29'
		]
		cam_names = [cam_name for cam_name in cam_names if os.path.isdir(os.path.join(root_image, cam_name))]

		cam_folders = [os.path.join(root_image, cam_name) for cam_name in cam_names]

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

		for frame_idx, frame in enumerate(range(start_frame, end_frame, interval)):

			bodies = os.path.join(root_skeleton, 'body3DScene_' + str(frame).zfill(8) + '.json')
			bodies = json.load(open(bodies))['bodies']

			if not bodies:
				print('empty frame skipped')
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
						print('static pose skipped', body_id)
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

			print('collecting samples [', str(frame_idx) + '/' + str((end_frame - start_frame) / interval), '] sequence', sequence)

	pool.close()
	pool.join()

	samples = [process.get() for process in processes]
	samples = [sample for sample in samples if sample]

	return data_info, samples


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


def make_h36m_sample(paths, annos, args):
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
	bbox, body_pose, camera = annos

	valid = np.ones(args.num_joints).astype(np.bool)

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


def get_h36m_group(phase, args):

	assert os.path.isdir(args.data_down_path)

	cameras = get_h36m_cameras(os.path.join(args.data_root_path, 'metadata.xml'))

	from joint_settings import h36m_short_names as short_names
	from joint_settings import h36m_parent as parent
	from joint_settings import h36m_mirror as mirror
	from joint_settings import h36m_base_joint as base_joint

	mapper = dict(zip(short_names, range(len(short_names))))

	map_mirror = [mapper[mirror[name]] for name in short_names if name in mirror]
	map_parent = [mapper[parent[name]] for name in short_names if name in parent]

	_mirror = np.arange(len(short_names))
	_parent = np.arange(len(short_names))

	_mirror[np.array([name in mirror for name in short_names])] = np.array(map_mirror)
	_parent[np.array([name in parent for name in short_names])] = np.array(map_parent)

	data_info = JointInfo(short_names, _parent, _mirror, mapper[base_joint])

	partitions = dict(
		train = [1, 5, 6, 7, 8],
		valid = [9, 11],
		test = [9, 11]
	)
	stride = dict(
		train = 5,
		valid = 64,
		test = 64
	)
	def cond(root_path, elem):
		return os.path.isdir(os.path.join(root_path, elem)) and '_' not in elem

	processes = []

	pool = multiprocessing.Pool(args.num_processes)

	for partition in partitions[phase]:

		root_part = os.path.join(args.data_root_path, 'S' + str(partition))
		root_image = os.path.join(root_part, 'Images')

		activities = [elem for elem in os.listdir(root_image) if cond(root_image, elem)]
		activities = set([elem.split('.')[0] for elem in activities])

		for index, (activity, camera_id) in enumerate(itertools.product(activities, range(4))):

			if partition == 11 and activity == 'Directions' and camera_id == 0:
				continue

			camera = cameras[camera_id][partition - 1]

			print('collecting samples', str(index) + '/' + str(len(activities) * 4), 'partition', partition)

			image_paths, body_poses, bboxes = collect_data(root_part, activity, camera_id, stride[phase])

			down_path = str(partition) + '.' + activity.replace(' ', '-') + '.' + str(camera_id)
			down_path = os.path.join(args.data_down_path, down_path)

			new_paths = [os.path.join(down_path, os.path.basename(path)) for path in image_paths]

			if not os.path.exists(down_path):
				os.mkdir(down_path)

			for image_path, new_path, body_pose, bbox in xzip(image_paths, new_paths, body_poses, bboxes):

				paths = (image_path, new_path)
				annos = (bbox, body_pose, camera)

				processes.append(pool.apply_async(func = make_h36m_sample, args = (paths, annos, args)))

	pool.close()
	pool.join()
	samples = [process.get() for process in processes]

	return data_info, samples


def load_ntu_cameras(args):

	with open(os.path.join(args.data_root_path, 'cameras.pkl'), 'rb') as file:
		color_cameras = pickle.load(file)

	with open(os.path.join(args.data_root_path, 'depth_cameras.pkl'), 'rb') as file:
		depth_cameras = pickle.load(file)

	return color_cameras, depth_cameras


def by_sequence(phase, sample_file):

	partitions = dict(
		train = ['S001', 'S002', 'S003', 'S004', 'S005', 'S006', 'S007', 'S008', 'S009', 'S010', 'S011'],
		valid = ['S012', 'S013', 'S014'],
		test = ['S015', 'S016', 'S017']
	)
	cam_id = os.path.basename(sample_file).split('.')[0]

	return cam_id[:4] in partitions[phase]


def make_ntu_sample(sample, cameras, image, args):
	'''
	Args:
		sample: dict(skeleton = pose_coord, color = color_coord, depth = depth_coord, frame = frame, video = video_id, bbox = bbox)
		cameras: tuple(color_cam, depth_cam)
	'''
	color_cam, depth_cam = cameras

	box_center = boxlib.center(sample['bbox'])

	depth_bbox = utils.transfer_bbox(sample['bbox'], color_cam, depth_cam)

	sine = np.sin(np.pi / 6)
	cosine = np.cos(np.pi / 6)

	expand_shape = np.array([[cosine, sine], [sine, cosine]]) @ sample['bbox'][2:, np.newaxis]
	expand_side = np.max(expand_shape)

	scale_factor = min(args.side_in / np.max(sample['bbox'][2:]) / args.random_zoom, 1.0)

	dest_side = int(np.round(expand_side * scale_factor))

	new_cam = copy.deepcopy(color_cam)
	new_cam.shift_to_center(box_center, (expand_side, expand_side))
	new_cam.scale_output(scale_factor)

	new_bbox = cameralib.reproject_points(sample['bbox'][None, :2], color_cam, new_cam)[0]

	new_bbox = np.concatenate([new_bbox, sample['bbox'][2:] * scale_factor])

	new_path = os.path.join(args.down_path, str(sample['frame']) + '.jpg')

	if not os.path.exists(new_path):

		new_image = cameralib.reproject_image(image, color_cam, new_cam, (dest_side, dest_side))

		cv2.imwrite(new_path, new_image[:, :, ::-1])

	sample['image'] = new_path
	sample['bbox'] = new_bbox
	sample['new_cam'] = new_cam
	sample['depth_bbox'] = depth_bbox

	return sample


def get_ntu_group(phase, args):

	assert os.path.isdir(args.data_down_path)

	color_cameras, depth_cameras = load_ntu_cameras(args)

	sample_files = glob.glob(os.path.join(args.data_root_path, 'midway_samples', '*.pkl'))

	sample_files = [file for file in sample_files if by_sequence(phase, file)]

	for i_cam, sample_file in enumerate(sample_files):

		processes = []

		pool = multiprocessing.Pool(args.num_processes)

		cam_id = os.path.basename(sample_file).split('.')[0]

		print('=> handles camera[', cam_id, ']: [', i_cam, '|', len(sample_files), ']')

		cameras = (color_cameras[cam_id], depth_cameras[cam_id])

		with open(sample_file, 'rb') as file:
			samples_cur_cam = pickle.load(file)

		samples_by_video = utils.groupby(samples_cur_cam, lambda sample: sample['video'])

		for i_vid, (video_id, samples_cur_video) in enumerate(samples_by_video.items()):

			print('\t => handles video[', video_id, ']: [', i_vid, '|', len(samples_by_video), ']')

			samples_by_frame = utils.groupby(samples_cur_video, lambda sample: sample['frame'])

			video_path = os.path.join(args.data_root_path, 'nturgb+d_rgb', video_id + '_rgb.avi')

			down_path = os.path.join(args.data_down_path, video_id)

			if not os.path.exists(down_path):
				os.mkdir(down_path)

			args.down_path = down_path

			for frame, image in enumerate(utils.prefetch(video_path, 10)):
				if frame in samples_by_frame:
					for sample in samples_by_frame[frame]:
						processes.append(pool.apply_async(func = make_ntu_sample, args = (sample, cameras, image, args)))

		pool.close()
		pool.join()
		samples = [process.get() for process in processes]

		with open(sample_file.replace('midway', 'final'), 'wb') as file:
			pickle.dump(samples, file)


def get_ntu_info():
	from joint_settings import h36m_short_names as short_names
	from joint_settings import h36m_parent as parent
	from joint_settings import h36m_mirror as mirror
	from joint_settings import h36m_base_joint as base_joint

	mapper = dict(zip(short_names, range(len(short_names))))

	map_mirror = [mapper[mirror[name]] for name in short_names if name in mirror]
	map_parent = [mapper[parent[name]] for name in short_names if name in parent]

	_mirror = np.arange(len(short_names))
	_parent = np.arange(len(short_names))

	_mirror[np.array([name in mirror for name in short_names])] = np.array(map_mirror)
	_parent[np.array([name in parent for name in short_names])] = np.array(map_parent)

	data_info = JointInfo(short_names, _parent, _mirror, mapper[base_joint])

	return data_info


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
