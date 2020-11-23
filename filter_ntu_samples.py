import os
import sys
import glob
import collections
import pickle5 as pickle
import boxlib
import cameralib
import numpy as np
import scipy
import multiprocessing

def groupby(items, key):
	result = collections.defaultdict(list)
	for item in items:
		result[key(item)].append(item)
	return result


def get_cam_id(anno_file):
	video_id = os.path.basename(anno_file).split('.')[0]
	return video_id[:8]


def sufficient_pose_change(prev_pose, current_pose):
    if prev_pose is None:
        return True
    dists = np.linalg.norm(prev_pose - current_pose, axis=-1)
    return np.count_nonzero(dists[~np.isnan(dists)] >= 100) >= 3


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

    dist_matrix = np.array([[pose_distance(p1, p2)
                             for p1 in current_poses]
                            for p2 in prev_poses])
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

	for anno_file in anno_files:
		# print('handle video:', anno_file)

		sample_count = 0

		prev_poses = []
		video_id = os.path.basename(anno_file).split('.')[0]

		anno = np.load(anno_file, allow_pickle = True, encoding = 'latin1').item()

		n_frames = len(anno['nbodys'])

		for frame in range(n_frames):
			cur_poses = [anno['skel_body' + str(body)][frame] for body in range(anno['nbodys'][frame])]
			cur_color_coords = [anno['rgb_body' + str(body)][frame] for body in range(anno['nbodys'][frame])]
			cur_depth_coords = [anno['depth_body' + str(body)][frame] for body in range(anno['nbodys'][frame])]

			cur_poses = [cur_pose * np.array([1000.0, -1000.0, 1000.0]) for cur_pose in cur_poses]

			are_changes_sufficient = are_changes_sufficient_and_update(prev_poses, cur_poses)

			for idx in np.where(are_changes_sufficient)[0]:

				pose_coord = cur_poses[idx]  # (25, 3)
				color_coord = cur_color_coords[idx]  # (25, 2)
				depth_coord = cur_depth_coords[idx]  # (25, 2)
				bbox = boxlib.expand(boxlib.bb_of_points(color_coord), 1.25)

				valid = camera.is_visible(pose_coord, [1920, 1080]) & (200.0 <= pose_coord[:, 2])

				if np.count_nonzero(valid) >= 15:
					samples.append(dict(skeleton = pose_coord, color = color_coord, depth = depth_coord, frame = frame, video = video_id, bbox = bbox))
					sample_count += 1

		# print('sample_count:', sample_count)

	with open(os.path.join(root_path, 'final_samples', cam_id + '.pkl'), 'wb') as file:
		pickle.dump(samples, file)

def main(root_path):
	with open(os.path.join(root_path, 'cameras.pkl'), 'rb') as file:
		cameras = pickle.load(file)

	anno_files = glob.glob(os.path.join(root_path, 'numpy_skeletons', '*.skeleton.npy'))

	anno_files_by_cam = groupby(anno_files, get_cam_id)

	pool = multiprocessing.Pool(6)

	for cam_id, annos in anno_files_by_cam.items():
		pool.apply_async(func = filter_samples, args = (annos, cam_id, cameras[cam_id], root_path))

	pool.close()
	pool.join()

if __name__ == '__main__':
	main(sys.argv[1])
