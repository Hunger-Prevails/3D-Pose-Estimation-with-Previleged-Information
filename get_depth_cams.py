import os
import sys
import glob
import numpy as np
import random
import pickle
import cameralib
import collections


np.seterr('raise')


def groupby(items, key):
	result = collections.defaultdict(list)
	for item in items:
		result[key(item)].append(item)
	return result


def get_cam_id(anno_file):
	video_id = os.path.basename(anno_file).split('.')[0]
	return video_id[:8]


def main(path):
	anno_files = glob.glob(os.path.join(path, '*.skeleton.npy'))

	anno_files_by_cam = groupby(anno_files, get_cam_id)

	cameras = {cam_id: get_camera(annos, cam_id) for cam_id, annos in anno_files_by_cam.items()}

	with open('/globalwork/data/NTU_RGBD/depth_cameras.pkl', 'wb') as file:
		pickle.dump(cameras, file)


def get_camera(anno_files, cam_id):

	print('compute intrinsics for camera:', cam_id)

	chosen_files = random.sample(anno_files, min(200, len(anno_files)))

	n_rows = 25 * len(chosen_files) * 2
	A = np.empty((n_rows, 4), dtype=np.float32)
	b = np.empty((n_rows, 1), dtype=np.float32)
	i = 0

	discard_rows = []

	for chosen_file in chosen_files:

		# print('gather equations from file:', chosen_file)

		anno = np.load(chosen_file, allow_pickle = True, encoding = 'latin1').item()

		n_frames = len(anno['nbodys'])

		frame = random.randrange(0, n_frames)

		coord_on_depth = anno['depth_body0'][frame]  # (25, 2)
		coord_cam = anno['skel_body0'][frame] * np.array([1000.0, -1000.0, 1000.0])  # (25, 3)

		for coords2d, coords3d in zip(coord_on_depth, coord_cam):
			x, y = coords2d
			x3, y3, z3 = coords3d

			try:
				A[i] = [x3 / z3, 0, 1, 0]
				A[i + 1] = [0, y3 / z3, 0, 1]
			except FloatingPointError:
				discard_rows += [i, i + 1]

			b[i] = [x]
			b[i + 1] = [y]
			i += 2

	A = np.delete(A, discard_rows, axis = 0)
	b = np.delete(b, discard_rows, axis = 0)

	print('A:', A.shape, '| b:', b.shape)

	if (A.shape[0] < 5000):
		print('too few valid videos chosen for camera:', cam_id)
		exit(0)

	try:
		rms_A = np.sqrt(np.mean(np.square(A), axis=0))
		rms_b = np.sqrt(np.mean(np.square(b), axis=0))

		result, residual, rank, sv = np.linalg.lstsq(A / rms_A, b / rms_b, rcond = None)
		result = result[:, 0] * rms_b / rms_A
		fx, fy, cx, cy = result

	except np.linalg.LinAlgError:
		print('LinAlgError!')
		exit()

	intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype = np.float32)
	return cameralib.Camera(intrinsic_matrix = intrinsics, world_up = (0, -1, 0))


if __name__ == '__main__':
	main(sys.argv[1])