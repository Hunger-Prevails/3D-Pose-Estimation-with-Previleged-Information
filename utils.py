import os
import torch
import numpy as np

class PoseSample:
	
	def __init__(self, image_path, body_pose, image_coords, bbox, camera):
		self.image_path = image_path
		self.body_pose = body_pose
		self.image_coords = image_coords
		self.bbox = bbox
		self.camera = camera


class PoseGroup:
	
	def __init__(self, phase, joint_info, samples):
		assert phase in ['train', 'validation', 'test']

		self.phase = phase
		self.joint_info = joint_info
		self.samples = samples


class JointInfo:
	def __init__(self, short_names, parents, mirror, key_index):
		self.short_names = short_names
		self.parents = parents
		self.mirror = mirror
		self.key_index = key_index


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


def statistics(cubics, reflects, tangents, thresholds):

	dist = dict(
		cubics = cubics,
		reflects = reflects,
		tangents = tangents
	)

	def count_and_eliminate(condition):
		remains = np.nonzero(np.logical_not(condition))

		dist['cubics'] = dist['cubics'][remains]
		dist['reflects'] = dist['reflects'][remains]
		dist['tangents'] = dist['tangents'][remains]

		return np.count_nonzero(condition)

	count = float(dist['cubics'].size)

	stats = ('perfect', 'good', 'jitter', 'switch', 'depth', 'fail')

	perfect = count_and_eliminate(dist['cubics'] <= thresholds['perfect']) / count
	
	good = count_and_eliminate(dist['cubics'] <= thresholds['good']) / count
	
	jitter = count_and_eliminate(dist['cubics'] <= thresholds['jitter']) / count
	
	switch = count_and_eliminate(dist['reflects'] <= thresholds['jitter']) / count
	
	depth_condition = (dist['tangents'] <= thresholds['jitter'] * (2 / 3) ** 0.5) & (thresholds['jitter'] < dist['cubics'])

	depth = count_and_eliminate(depth_condition) / count

	return dict(zip(stats, (perfect, good, jitter, switch, depth, dist['cubics'].size / count)))


def parse_epoch(scores_and_stats, total):

	keys = ('perfect', 'good', 'jitter', 'switch', 'depth', 'fail')
	keys += ('score_pck', 'score_auc', 'cam_mean', 'batch_size')

	values = np.array([[patch[key] for patch in scores_and_stats] for key in keys])

	return dict(zip(keys[:-1], np.sum(values[-1] * values[:-1], axis = 1) / total))


def analyze(spec_cam, true_cam, valid_mask, mirror, key_index, thresholds):
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
	spec_cam -= spec_cam[:, key_index:key_index + 1]
	true_cam -= true_cam[:, key_index:key_index + 1]

	cubics = np.linalg.norm(spec_cam - true_cam, axis = -1)
	reflects = np.linalg.norm(spec_cam - true_cam[:, mirror], axis = -1)
	tangents = np.linalg.norm(spec_cam[:, :, :2] - true_cam[:, :, :2], axis = -1)

	valid = np.where(valid_mask.flatten() == 1.0)[0]

	cubics = cubics.flatten()[valid]
	reflects = reflects.flatten()[valid]
	tangents = tangents.flatten()[valid]

	cam_mean = np.mean(cubics)
	score_pck = np.mean(cubics / thresholds['score'] <= 1.0)
	score_auc = np.mean(np.maximum(0, 1 - cubics / thresholds['score']))

	stats = statistics(cubics, reflects, tangents, thresholds)

	stats.update(
		dict(
			batch_size = spec_cam.shape[0],
			score_pck = score_pck,
			score_auc = score_auc,
			cam_mean = cam_mean
		)
	)
	return stats
