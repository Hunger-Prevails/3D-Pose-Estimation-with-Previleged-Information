import os
import itertools
import jpeg4py
import h5py
import cv2
import copy
import numpy as np
import transforms3d
import cameralib
import multiprocessing

from utils import JointInfo
from utils import PoseSample
from utils import PoseGroup
from functools import partial

pose_folder = 'hdPose3d_stage1_coco19'

def parse_json(json_path):
	ids = []
	poses = []

	with open(json_path) as json_file:
		bodies = json.load(json_file)['bodies']
		for body in bodies:
			ids.append(body['id'])
			poses.append(body['joints19'])

	poses = [np.array(pose).reshape((19, 4)) for pose in poses]
	return dict(ids = ids, poses = np.stack(poses).tolist())


def load_poses(seq_path, start_frame, end_frame, interval):
	assert os.path.isfile(os.path.join(seq_path, pose_folder, 'body3DScene_' + str(start_frame).zfill(8) + '.json'))
	assert os.path.isfile(os.path.join(seq_path, pose_folder, 'body3DScene_' + str(end_frame).zfill(8) + '.json'))

	sample_frames = range(start_frame, end_frame, interval)
	json_files = [os.path.join(seq_path, pose_folder, 'body3DScene_' + str(frame).zfill(8) + '.json') for frame in sample_frames]

	return [parse_json(json_file) for json_file in json_files]


def get_cameras(json_file):
	
    calibration = json.load(open(json_file))

    cameras = [cam for cam in calibration['cameras'] if cam['panel'] == 0]

    return [cameralib.Camera(
    						np.array(cam['t']).flatten(),
    						np.array(cam['R']),
    						np.array(cam['K']),
    						np.array(cam['distCoef'])
    			) for cam in cameras]


def make_sample(data_sample, camera, args):

	image_path, image_coord, bbox, body_pose, folder_down, phase = data_sample

	max_rotate = np.pi / 6

	rotation = np.array([
							[np.cos(max_rotate), np.sin(max_rotate)]
							[np.sin(max_rotate), np.cos(max_rotate)]
						])
	
	expand_side = np.max(np.matmul(rotation, bbox[2:]))

	box_center = bbox[:2] + bbox[2:] / 2

	scale_factor = min(base_dst_side / np.max(bbox[2:]) / args.random_zoom, 1.0)

	dest_side = expand_side * scale_factor

	new_camera = copy.deepcopy(camera)
	new_camera.shift_to_center(box_center, (expand_side, expand_side))
	new_camera.scale_output(scale_factor)

	new_path = os.path.join(folder_down, os.path.basename(image_path))

	if not os.path.exists(new_path):
		image = jpeg4py.JPEG(image_path).decode()
		new_image = cameralib.reproject_image(image, camera, new_camera, (dest_side, dest_side))
		cv2.imwrite(new_path, new_image[:, :, ::-1])

	new_bbox = cameralib.reproject_points(bbox[None, :2], camera, new_camera)[0]
	new_bbox = np.concatenate((new_bbox, bbox[2:] * scale_factor))

	return PoseSample(new_path, body_pose, image_coord, new_bbox, new_camera)


def coord_to_box(image_coord, box_margin):
	'''
	params
		image_coord: (19 x 3) joint coords in image space with confidence scores
	returns
		image_box: (4,) pseudo bounding box of the person
	'''
	x_min = np.min(image_coord[:, 0])
	x_max = np.max(image_coord[:, 0])
	y_min = np.min(image_coord[:, 1])
	y_min = np.max(image_coord[:, 1])

	center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
	shape = np.array([x_max - x_min, y_max - y_min])

	return np.hstack([center - shape * box_margin / 2, shape * box_margin])

def get_cmu_panoptic_group(phase, args):

	assert os.path.isdir(args.root_down)
	
	from joint_settings import cmu_panoptic_short_names as short_names
	from joint_settings import cmu_panoptic_parents as parents
	from joint_settings import cmu_panoptic_pairs as pairs

	mapper = dict(zip(short_names, range(len(short_names))))
	mapped = [mapper[pairs[name]] for name in short_names if name in pairs]

	mirror = np.arange(len(short_names))
	mirror[np.array([name in pairs for name in short_names])] = np.array(mapped)

	joint_info = JointInfo(short_names, parents, mirror)

	sequences = dict(
		train = ['171204_pose1', '171204_pose2', '171026_pose1', '171026_pose2'],
		validation = ['171204_pose3'],
		test = ['171026_pose3']
	)
	frame_step = dict(
		train = 10,
		validation = 10,
		test = 50
	)
	processes = []

	pool = multiprocessing.Pool(args.num_processes)

	for sequence in sequences[phase]:

		root_sequence = os.path.join(args.root_path, sequence)
		root_image = os.path.join(root_sequence, 'hdImgs')

		cam_files = [os.path.join(root_image, file) for file in os.listdir(root_image)]
		cam_files = [file for file in cam_files if os.path.isfile(file)]
		cam_files.sort()
		cam_files = [json.load(open(cam_file)) for cam_file in cam_files]

		cam_folders = [os.path.join(root_image, folder) for folder in os.listdir(root_image)]
		cam_folders = [folder for folder in cam_folders if os.path.isdir(folder)]
		cam_folders.sort()
		cam_names = [os.path.basename(folder) for folder in cam_folders]

		start_frame = cam_files[0]['start_frame']
		end_frame = cam_files[0]['end_frame']
		interval = cam_files[0]['interval']

		bodies = load_poses(root_sequence, start_frame, end_frame, interval)
		cameras = get_cameras(os.path.join(root_sequence, 'calibration_' + seq_name + '.json'))[:len(cam_files)]

		pose_frame_idx = 0

		cam_folder_down = [os.path.join(args.root_down, sequence + '.' + cam_name) for cam_name in cam_names]

		for frame_idx, body in enumerate(bodies):

			if not body['poses']:
				continue

			body_pose = np.array(body['poses'][0])

			for cam_idx, cam_folder in enumerate(cam_folders):

				frame = frame_idx * interval + start_frame

				if frame_idx * interval % frame_step[phase] != 0:
					continue

				image_path = os.path.join(root_image, cam_folder, cam_folder + '_' + str(frame).zfill(8) + '.jpg')
				image_coord = np.array(cam_files[cam_idx]['image_coord'][pose_frame_idx])
				image_box = coord_to_box(image_coord, args.box_margin)

				data_sample = (image_path, image_coord, image_box, body_pose, cam_folder_down[cam_idx], phase)
				processes.append(pool.apply_async(func = make_sample, args = (data_sample, camera[cam_idx], args.side_eingabe)))

			print 'collecting samples [', str(frame_idx) + '/' + str(len(bodies)), '] sequence', sequence

			pose_frame_idx += 1

	pool.close()
	pool.join()
	samples = [process.get() for process in processes]

	return PoseGroup(phase, joint_info, samples)


def get_mpi_3dhp_group(phase, args):
	pass;
