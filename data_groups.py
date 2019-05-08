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


pose_folder = 'hdPose3d_stage1_coco19'


def get_cameras(json_file, cam_names):
	
	calibration = json.load(open(json_file))

	cameras = [cam for cam in calibration['cameras'] if cam['panel'] == 0]

	return dict(
			[
				(
					cam['name'],
					cameralib.Camera(
							np.array(cam['t']).flatten(),
							np.array(cam['R']),
							np.array(cam['K']),
							np.array(cam['distCoef'])
					)
				) for cam in cameras if cam['name'] in cam_names
			]
		)


def make_sample(data_sample, data_params, camera):

	image_path, image_coord, bbox, body_pose = data_sample
	folder_down, side_eingabe, random_zoom = data_params

	max_rotate = np.pi / 6

	rotation = np.array([
							[np.cos(max_rotate), np.sin(max_rotate)],
							[np.sin(max_rotate), np.cos(max_rotate)]
						])
	expand_side = np.max(np.matmul(rotation, bbox[2:]))

	box_center = bbox[:2] + bbox[2:] / 2

	scale_factor = min(side_eingabe / np.max(bbox[2:]) / random_zoom, 1.0)

	dest_side = int(np.round(expand_side * scale_factor))

	new_camera = copy.deepcopy(camera)
	new_camera.shift_to_center(box_center, (expand_side, expand_side))
	new_camera.scale_output(scale_factor)

	new_path = os.path.join(folder_down, os.path.basename(image_path))

	new_bbox = cameralib.reproject_points(bbox[None, :2], camera, new_camera)[0]
	new_bbox = np.concatenate((new_bbox, bbox[2:] * scale_factor))

	def draw(image, box):
		image = image.copy()
		
		round_box = [int(np.round(x)) for x in box]

		pt_a = (round_box[0],          round_box[1])
		pt_b = (round_box[0] + round_box[2], round_box[1])
		pt_c = (round_box[0] + round_box[2], round_box[1] + round_box[3])
		pt_d = (round_box[0],          round_box[1] + round_box[3])
		
		cv2.line(image, pt_a, pt_b, color = (0, 255, 255), thickness = 2)
		cv2.line(image, pt_b, pt_c, color = (0, 255, 255), thickness = 2)
		cv2.line(image, pt_c, pt_d, color = (0, 255, 255), thickness = 2)
		cv2.line(image, pt_d, pt_a, color = (0, 255, 255), thickness = 2)

		return image

	if not os.path.exists(new_path):
		image = jpeg4py.JPEG(image_path).decode()

		new_image = cameralib.reproject_image(image, camera, new_camera, (dest_side, dest_side))

		# cv2.imwrite(new_path, draw(new_image[:, :, ::-1], new_bbox))
		cv2.imwrite(new_path, new_image[:, :, ::-1])

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
	y_max = np.max(image_coord[:, 1])

	center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
	shape = np.array([x_max - x_min, y_max - y_min])

	return np.hstack([center - shape / box_margin / 2, shape / box_margin])

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

		cam_folders = [os.path.join(root_image, folder) for folder in os.listdir(root_image)]
		cam_folders = [folder for folder in cam_folders if os.path.isdir(folder)]
		cam_folders.sort()

		cam_names = [os.path.basename(folder) for folder in cam_folders]

		cam_files = [os.path.join(root_image, 'image_coord_' + cam_name + '.json') for cam_name in cam_names]
		cam_files = [json.load(open(file)) for file in cam_files]

		down_folders = [os.path.join(args.root_down, sequence + '.' + cam_name) for cam_name in cam_names]

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

			body_pose = np.array(skeleton[0]['joints19']).reshape((-1,4))[:, :3]

			for cam_name in cam_names:

				if (frame - start_frame) % frame_step[phase] != 0:
					continue

				if not os.path.exists(down_folders[cam_name]):
					os.mkdir(down_folders[cam_name])

				image_path = os.path.join(cam_folders[cam_name], cam_name + '_' + str(frame).zfill(8) + '.jpg')
				image_coord = np.array(cam_files[cam_name]['image_coord'][pose_idx])
				image_box = coord_to_box(image_coord, args.box_margin)

				data_sample = (image_path, image_coord, image_box, body_pose)
				data_params = (down_folders[cam_name], args.side_eingabe, args.random_zoom)

				processes.append(pool.apply_async(func = make_sample, args = (data_sample, data_params, cameras[cam_name])))

			print 'collecting samples [', str(frame_idx) + '/' + str((end_frame - start_frame) / interval), '] sequence', sequence

			pose_idx += 1

	pool.close()
	pool.join()
	samples = [process.get() for process in processes]

	return PoseGroup(phase, joint_info, samples)


def get_mpi_3dhp_group(phase, args):
	pass;
