import os
import sys
import glob
import utils
import collections
import pickle5 as pickle
import boxlib
import cameralib
import numpy as np
import scipy
import multiprocessing
import matplotlib.pyplot as plt

from utils import prefetch


def get_cam_id(anno_file):
	video_id = os.path.basename(anno_file).split('.')[0]
	return video_id[:8]


def sufficient_pose_change(prev_pose, current_pose):
	if prev_pose is None:
		return True
	dists = np.linalg.norm(prev_pose - current_pose, axis = -1)
	return np.sum(dists >= 100) >= 3


def non_empty(pose):
	return not np.any(np.isnan(pose))


def are_changes_sufficient_and_update(prev_poses, current_poses):
	'''
	Function:
		associates current poses with previous poses
		update pose and mark as positive only if matched pair is sufficiently dissimilar
		mark all unmatched current poses as positive and insert them into the list of previous poses
	Args:
		prev_poses: list of previous poses, each a (n_joints, 3) np.array
		current_poses: list of poses from the current frame, each a (n_joints, 3) np.array
	'''

	result = [True] * len(current_poses)
	if not prev_poses:
		prev_poses.extend(current_poses)
		return result

	def pose_distance(p1, p2):
		return np.nanmean(np.linalg.norm(p1 - p2, axis=-1))

	dist_matrix = np.array([[pose_distance(p1, p2) for p1 in current_poses] for p2 in prev_poses])

	prev_indices, current_indices = scipy.optimize.linear_sum_assignment(dist_matrix)

	for pi, ci in zip(prev_indices, current_indices):
		result[ci] = sufficient_pose_change(prev_poses[pi], current_poses[ci])
		if result[ci]:
			prev_poses[pi] = current_poses[ci]

	for i, current_pose in enumerate(current_poses):
		if i not in current_indices:
			prev_poses.append(current_pose)

	return result


def filter_samples(anno_files, cam_id, camera, root_path):
	
	samples = []
	anno_files.sort()

	for anno_file in anno_files:
		prev_poses = []
		video_id = os.path.basename(anno_file).split('.')[0]

		indices = [63, 4, 7, 38, 3, 6, 5, 47, 24, 27, 42, 17, 19, 67, 18, 20, 52]

		skeletons = np.load(anno_file)[:, :, indices]

		n_frames = skeletons.shape[1]

		print('collect samples from video:', video_id)

		for frame in range(n_frames):
			cur_poses = list(filter(non_empty, skeletons[:, frame]))

			are_changes_sufficient = are_changes_sufficient_and_update(prev_poses, cur_poses)

			for idx in np.where(are_changes_sufficient)[0]:

				pose_coord = cur_poses[idx]
				color_coord = camera.world_to_image(pose_coord)
				bbox = boxlib.expand(boxlib.bb_of_points(color_coord), 1.25)

				valid = camera.is_visible(pose_coord, [1920, 1080]) & (200.0 <= pose_coord[:, 2])

				if np.count_nonzero(valid) >= 15:
					samples.append(dict(skeleton = pose_coord, valid = valid, frame = frame, video = video_id, bbox = bbox))

	with open(os.path.join(root_path, 'midway_samples', cam_id + '.pkl'), 'wb') as file:
		pickle.dump(samples, file)


def main(root_path, skeleton_path):
	with open(os.path.join(root_path, 'cameras.pkl'), 'rb') as file:
		cameras = pickle.load(file)

	anno_files = glob.glob(os.path.join(skeleton_path, '*.npy'))

	anno_files_by_cam = utils.groupby(anno_files, get_cam_id)

	pool = multiprocessing.Pool(6)

	for cam_id, annos in anno_files_by_cam.items():
		pool.apply_async(func = filter_samples, args = (annos, cam_id, cameras[cam_id], root_path))

	pool.close()
	pool.join()


def show_mat(image_coord, ax, bbox = None):
	'''
	Shows skeleton(mat)

	Args:
	    image_coord: (21, 2)
	'''
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
	ax.plot(image_coord[:, 0], image_coord[:, 1], '.', color = 'yellow')

	for edge in body_edges:
		ax.plot(image_coord[edge, 0], image_coord[edge, 1], '--', color = 'b')

	if bbox is not None:
		rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
		ax.add_patch(rect)


def visualize(image, skeletons, camera):
	plt.figure(figsize = (12, 8))

	ax = plt.subplot(1, 1, 1)
	ax.imshow(image)

	for skeleton in skeletons:
		show_mat(camera.world_to_image(skeleton), ax)

	plt.show()


if __name__ == '__main__':
	main(sys.argv[1], sys.argv[2])
