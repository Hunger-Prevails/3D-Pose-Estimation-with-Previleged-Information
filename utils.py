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
	def __init__(self, short_names, parents, mirror):
		self.short_names = short_names
		self.parents = parents
		self.mirror = mirror


def to_heatmap(ausgabe, depth, num_joints, height, width):
	heatmap = ausgabe.view(-1, depth, num_joints, height, width)
	heatmap = heatmap.permute(0, 2, 3, 4, 1).contiguous()
	
	heatmap = heatmap.view(-1, num_joints, height * width * depth)

	max_val = torch.max(heatmap, dim = 2, keepdim = True)[0]
	heatmap = torch.exp(heatmap - max_val)
	heatmap = heatmap / torch.sum(heatmap, dim = 2, keepdim = True)
	
	return heatmap.view(-1, num_joints, height, width, depth)


def decode(heatmap, side_eingabe, depth_range, cuda_device):
	height_map = torch.sum(heatmap, dim = (3, 4))
	width_map = torch.sum(heatmap, dim = (2, 4))
	depth_map = torch.sum(heatmap, dim = (2, 3))

	height_grid = torch.linspace(0.0, 1.0, height_map.size(-1)).view(1, 1, -1).to(cuda_device)
	width_grid = torch.linspace(0.0, 1.0, width_map.size(-1)).view(1, 1, -1).to(cuda_device)
	depth_grid = torch.linspace(0.0, 2.0, depth_map.size(-1)).view(1, 1, -1).to(cuda_device)

	height = torch.sum(height_grid * height_map, dim = 2)
	width = torch.sum(width_grid * width_map, dim = 2)
	depth = torch.sum(depth_grid * depth_map, dim = 2)

	planar_coords = torch.stack((width, height), dim = -1) * side_eingabe
	depth = depth_range * depth - depth_range

	return planar_coords, depth
	

def to_coordinate(planar_coords, depth, true_coords, inv_intrinsics, cuda_device):
	'''
	Maps planar coords and depth to prediction under rotated camera.
	
	Args:
		planar_coords: (batch_size, num_joints, 2)
		depth: (batch_size, num_joints)
		true_coords: (batch_size, num_joints, 3)
		return: (batch_size, num_joints, 3)
	'''
	return torch.einsum(
		'bij,bcj->bci',
		inv_intrinsics,
		torch.cat(
			(
				planar_coords,
				torch.ones(planar_coords.size()[:-1]).unsqueeze(-1).to(cuda_device)
			), 
			dim = -1
		)
	) * (depth + true_coords[:, -1:, 2]).unsqueeze(-1)


def statistics(dist_cubic, dist_mirrored, dist_planar, thresholds):

	dist = dict(
		cubic = dist_cubic,
		mirrored = dist_mirrored,
		planar = dist_planar
	)

	def count_and_eliminate(condition):
		remainings = np.nonzero(np.logical_not(condition))

		dist['cubic'] = dist['cubic'][remainings]
		dist['mirrored'] = dist['mirrored'][remainings]
		dist['planar'] = dist['planar'][remainings]

		return np.count_nonzero(condition)

	count = float(dist['cubic'].size)
	stats = ('perfect', 'good', 'jitter', 'switch', 'depth', 'fail')

	perfect = count_and_eliminate(dist['cubic'] <= thresholds['perfect']) / count
	
	good = count_and_eliminate(dist['cubic'] <= thresholds['good']) / count
	
	jitter = count_and_eliminate(dist['cubic'] <= thresholds['jitter']) / count
	
	switch = count_and_eliminate(dist['mirrored'] <= thresholds['jitter']) / count
	
	depth_condition = (dist['planar'] <= thresholds['jitter'] * (2 / 3) ** 0.5) & (thresholds['jitter'] < dist['cubic'])
	depth = count_and_eliminate(depth_condition) / count

	return dict(zip(stats, (perfect, good, jitter, switch, depth, dist['cubic'].size / count)))


def parse_epoch(scores_and_stats, total):

	keys = ('perfect', 'good', 'jitter', 'switch', 'depth', 'fail')
	keys += ('score_pck', 'score_auc', 'overall_mean', 'batch_size')

	values = np.array([[patch[key] for patch in scores_and_stats] for key in keys])

	return dict(zip(keys[:-1], np.sum(values[-1] * values[:-1], axis = 1) / total))