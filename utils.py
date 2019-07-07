import os
import torch
import numpy as np

from builtins import zip as xzip

class PoseSample:
	
	def __init__(self, image_path, body_pose, image_coords, bbox, camera):
		self.image_path = image_path
		self.body_pose = body_pose
		self.image_coords = image_coords
		self.bbox = bbox
		self.camera = camera


class PoseGroup:
	
	def __init__(self, phase, joint_info, samples):
		assert phase in ['train', 'valid', 'test']

		self.phase = phase
		self.joint_info = joint_info
		self.samples = samples


class JointInfo:
	def __init__(self, short_names, parent, mirror, key_index, weight = None):
		self.short_names = short_names
		self.parent = parent
		self.mirror = mirror
		self.key_index = key_index
		self.weight = weight


def to_heatmap(ausgabe, depth, num_joints, height, width):
	'''
	performs axis permutation and numerically stable softmax to output feature map

	args:
		ausgabe: (batch_size, depth x num_joints, height, width)

	returns:
		volumetric heatmap of shape(batch_size, num_joints, height, width, depth)
	'''
	heatmap = ausgabe.view(-1, depth, num_joints, height, width)
	heatmap = heatmap.permute(0, 2, 3, 4, 1).contiguous()
	
	heatmap = heatmap.view(-1, num_joints, height * width * depth)

	max_val = torch.max(heatmap, dim = 2, keepdim = True)[0]  # (batch_size, num_joints)

	heatmap = torch.exp(heatmap - max_val)

	heatmap = heatmap / torch.sum(heatmap, dim = 2, keepdim = True)
	
	return heatmap.view(-1, num_joints, height, width, depth)


def decode(heatmap, depth_range):
	'''
	performs position interpolation over each axis
	'''
	heat_y = torch.sum(heatmap, dim = (3, 4))
	heat_x = torch.sum(heatmap, dim = (2, 4))
	heat_z = torch.sum(heatmap, dim = (2, 3))

	grid_y = torch.linspace(0.0, 2.0, heat_y.size(-1), device = heat_y.device).view(1, 1, -1)
	grid_x = torch.linspace(0.0, 2.0, heat_x.size(-1), device = heat_x.device).view(1, 1, -1)
	grid_z = torch.linspace(0.0, 2.0, heat_z.size(-1), device = heat_z.device).view(1, 1, -1)

	coord_y = torch.sum(grid_y * heat_y, dim = 2)
	coord_x = torch.sum(grid_x * heat_x, dim = 2)
	coord_z = torch.sum(grid_z * heat_z, dim = 2)

	return torch.stack((coord_x, coord_y, coord_z), dim = 2) * depth_range


def statistics(original, mirrored, tangential, thresh):

	dist = dict(
		original = original,
		mirrored = mirrored,
		tangential = tangential
	)
	def count_and_eliminate(condition):
		remains = np.nonzero(np.logical_not(condition))

		dist['original'] = dist['original'][remains]
		dist['mirrored'] = dist['mirrored'][remains]
		dist['tangential'] = dist['tangential'][remains]

		return np.count_nonzero(condition)

	count = float(dist['original'].size)

	keys = ('solid', 'close', 'jitter', 'depth', 'switch', 'fail')

	solid = count_and_eliminate(dist['original'] <= thresh['solid']) / count
	
	close = count_and_eliminate(dist['original'] <= thresh['close']) / count
	
	jitter = count_and_eliminate(dist['original'] <= thresh['rough']) / count
	
	depth = count_and_eliminate(dist['tangential'] <= thresh['close']) / count

	switch = count_and_eliminate(dist['mirrored'] <= thresh['rough']) / count

	return dict(zip(keys, (solid, close, jitter, depth, switch, dist['original'].size / count)))


def parse_epoch(stats, total):

	keys = ('solid', 'close', 'jitter', 'depth', 'switch', 'fail')
	keys += ('score_pck', 'score_auc', 'cam_mean', 'batch_size')

	values = np.array([[patch[key] for patch in stats] for key in keys])

	return dict(zip(keys[:-1], np.sum(values[-1] * values[:-1], axis = 1) / total))


def analyze(spec_cam, true_cam, valid_mask, mirror, thresh):
	'''
	Analyzes spec_cam against true_cam under shifted original camera

	Args:
		spec_cam: (batch_size, num_joints, 3)
		true_cam: (batch_size, num_joints, 3)
		valid_mask: (batch_size, num_joints)
		mirror: (num_joints,)

	Returns:
		dict containing batch_size | scores | statistics

	'''
	original = np.linalg.norm(spec_cam - true_cam, axis = -1)
	mirrored = np.linalg.norm(spec_cam - true_cam[:, mirror], axis = -1)
	tangential = np.linalg.norm(spec_cam[:, :, :2] - true_cam[:, :, :2], axis = -1)

	valid = np.where(valid_mask.flatten() == 1.0)

	original = original.flatten()[valid]
	mirrored = mirrored.flatten()[valid]
	tangential = tangential.flatten()[valid]

	cam_mean = np.mean(original)
	score_pck = np.mean(original / thresh['rough'] <= 1.0)
	score_auc = np.mean(np.maximum(0, 1 - original / thresh['rough']))

	stats = statistics(original, mirrored, tangential, thresh)

	stats.update(
		dict(
			batch_size = spec_cam.shape[0],
			score_pck = score_pck,
			score_auc = score_auc,
			cam_mean = cam_mean
		)
	)
	return stats


def least_square(A, b, weight):
	'''
	Performs weighted least square regression

	Args:
		A: (num_valid x 2, 3)
		b: (num_valid x 2,)
		weight: (num_valid,)
	'''
	weight = np.tile(weight.reshape(-1, 1) ** 0.5, (1, 2))  # (num_valid, 2)

	A = A * weight.reshape(-1, 1)
	b = b * weight.reshape(-1)

	return np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))


def get_deter_cam(_spec_mat, _relat_cam, _valid_mask, _intrinsics, weight):
	'''
	Reconstructs the reference point location.

	Args:
		_spec_mat: (batch_size, num_joints, 2) estimated image coordinates
		_relat_cam: (batch_size, num_joints, 3) estimated relative camera coordinates with respect to an unknown reference point
		_valid_mask: (batch_size, num_joints)
		_intrinsics: (batch_size, 3, 3) camera intrinsics

	Returns:
		(batch_size, num_joints, 3) estimation of camera coordinates
	'''
	deter_cams = []

	batch_zip = xzip(_spec_mat, _relat_cam, _valid_mask, _intrinsics)

	for (spec_mat, relat_cam, valid_mask, intrinsics) in batch_zip:

		valid = np.where(valid_mask == 1.0)

		spec_mat = spec_mat[valid].copy()

		num_valid = spec_mat.shape[0]

		assert num_valid != 0 and num_valid != 1

		normalized = np.hstack([spec_mat, np.ones((num_valid, 1))])
		normalized = np.matmul(normalized, np.linalg.inv(intrinsics).T)[:, :2]

		A = np.hstack([np.vstack([np.eye(2, 2)] * num_valid), - normalized.reshape(-1, 1)])  # (num_valid x 2, 3)

		b = (normalized * relat_cam[valid, 2:] - relat_cam[valid, :2]).reshape(-1)  # (num_valid x 2,)

		deter_cams.append(relat_cam + least_square(A, b, weight[valid]))

	return np.stack(deter_cams)
