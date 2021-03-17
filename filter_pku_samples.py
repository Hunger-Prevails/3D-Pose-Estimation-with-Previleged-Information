import os
import sys
import glob
import json
import boxlib
import cameralib
import multiprocessing 
import numpy as np
import pickle5 as pickle

from functools import partial
from filter_ntu_samples import are_changes_sufficient_and_update


def exclude(exclusions, anno_file):
	for exc_file in exclusions:
		if exc_file in anno_file:
			return False
	return True


def kinect_to_box(camera, skel):
	direction = np.array([1.0, -1.0, 1.0])
	image_coords = camera.camera_to_image(np.multiply(skel, direction))
	image_coords[:, 0] = 1920 - image_coords[:, 0]
	return boxlib.bb_of_points(image_coords)


def reap_by_iou(infer_skels, kinect_skels, camera):
	kinect_boxes = [kinect_to_box(camera, skel) for skel in kinect_skels if np.all(skel[:, 2] != 0.0)]
	infer_skels = [skel for skel in infer_skels if not np.any(np.isnan(skel))]
	infer_boxes = [boxlib.bb_of_points(camera.camera_to_image(skel)) for skel in infer_skels]

	ret = []

	for kinect_box in kinect_boxes:
		iou_scores = [boxlib.iou(kinect_box, infer_box) for infer_box in infer_boxes]
		best_match = np.argmax(iou_scores)

		if iou_scores[best_match] > 0.5:
			ret.append(infer_skels[best_match])

	return ret


def filter_samples(anno_file, camera):
	skeletons = np.load(anno_file)
	indices = [63, 4, 7, 38, 3, 6, 5, 47, 24, 27, 42, 17, 19, 67, 18, 20, 52]
	skeletons = skeletons[:, :, indices]

	n_frames = skeletons.shape[1]

	video_id = os.path.basename(anno_file)[:6]

	print('collect samples from video: [', video_id, ']')

	label_file = os.path.join('/globalwork/data/pkummd/Train_Label_PKU_final', video_id + '.txt')

	with open(label_file) as file:
		lines = [line.strip() for line in file.readlines()]

	begin_frames = [int(line.split(',')[1]) for line in lines]
	end_frames = [int(line.split(',')[2]) for line in lines]

	origin_file = os.path.join('/globalwork/data/pkummd/PKU_Skeleton_Renew', video_id + '.txt')

	with open(origin_file) as file:
		lines = [line.strip() for line in file.readlines()]

	origin_skels = np.stack([np.asarray([float(val) for val in line.split(' ')]).reshape(2, 25, 3) for line in lines], axis = 1)

	samples = []

	for begin, end in zip(begin_frames, end_frames):

		prev_poses = []

		for frame in range(begin, end):
			cur_poses = reap_by_iou(skeletons[:, frame], origin_skels[:, frame], camera)

			are_changes_sufficient = are_changes_sufficient_and_update(prev_poses, cur_poses)

			for idx in np.where(are_changes_sufficient)[0]:

				pose_coord = cur_poses[idx]
				color_coord = camera.world_to_image(pose_coord)
				bbox = boxlib.expand(boxlib.bb_of_points(color_coord), 1.25)

				valid = camera.is_visible(pose_coord, [1920, 1080]) & (200.0 <= pose_coord[:, 2])

				if np.count_nonzero(valid) >= 15:
					samples.append(dict(skeleton = pose_coord, valid = valid, frame = frame, video = video_id, bbox = bbox))

	return samples


def main(root, anno_path):
	anno_files = sorted(glob.glob(os.path.join(anno_path, '*.npy')))

	exclusions = json.load(open(os.path.join(root, 'exclusions.json')))

	anno_files = list(filter(partial(exclude, exclusions), anno_files))

	intrinsics = np.array([[1.03e3, 0, 9.80e2], [0, 1.03e3, 5.50e2], [0, 0, 1]])
	
	camera = cameralib.Camera(intrinsic_matrix = intrinsics, world_up = (0, -1, 0))

	processes = []

	pool = multiprocessing.Pool(6)

	for anno_file in anno_files:
		processes.append(pool.apply_async(func = filter_samples, args = (anno_file, camera)))

	pool.close()
	pool.join()

	samples = []

	for process in processes:
		samples += process.get()

	with open(os.path.join(root, 'midway_samples.pkl'), 'wb') as file:
		pickle.dump(samples, file)

if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])
