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

	returns:
		volumetric heatmap of shape(batch_size, num_joints, height, width, depth)
	'''
	heatmap = ausgabe.view(-1, depth, num_joints, height, width)
	heatmap = heatmap.permute(0, 2, 3, 4, 1).contiguous()
	
	heatmap = heatmap.view(-1, num_joints, height * width * depth)

	max_val = torch.max(heatmap, dim = 2, keepdim = True)[0]
	heatmap = torch.exp(heatmap - max_val)
	heatmap = heatmap / torch.sum(heatmap, dim = 2, keepdim = True)
	
	return heatmap.view(-1, num_joints, height, width, depth)


def to_coordinate(heatmap, side_eingabe, depth_range, intrinsics, key_depth, cuda_device):
	'''
	Function:
		Performs position interpolation over each axis
		Maps interpolation results over x and y axis to a range of [0, side_eingabe]
		Maps interpolation results over z axis to a range of [- depth_range, depth_range]
		Transforms homogeneous 2D coordinates into 3D camera coordinates
	Args:
		heatmap: (batch_size, num_joints, height, width, depth)
		key_depth: (batch_size, 1)
	'''
	height_map = torch.sum(heatmap, dim = (3, 4))
	width_map = torch.sum(heatmap, dim = (2, 4))
	depth_map = torch.sum(heatmap, dim = (2, 3))

	height_grid = torch.linspace(0.0, 1.0, height_map.size(-1)).view(1, 1, -1).to(cuda_device)
	width_grid = torch.linspace(0.0, 1.0, width_map.size(-1)).view(1, 1, -1).to(cuda_device)
	depth_grid = torch.linspace(0.0, 2.0, depth_map.size(-1)).view(1, 1, -1).to(cuda_device)

	height = torch.sum(height_grid * height_map, dim = 2) * side_eingabe
	width = torch.sum(width_grid * width_map, dim = 2) * side_eingabe
	depth = torch.sum(depth_grid * depth_map, dim = 2) * depth_range - depth_range
	extension = torch.ones_like(height, device = cuda_device)

	prediction = torch.einsum('bij,bcj->bci', intrinsics, torch.stack((width, height, extension), dim = -1))  # (batch_size, num_joints, 3)
	return prediction * (depth + key_depth).unsqueeze(-1)  # (batch_size, num_joints, 3)


def decode(heatmap, depth_range, cuda_device):
	'''
	performs position interpolation over each axis
	'''
	height_map = torch.sum(heatmap, dim = (3, 4))
	width_map = torch.sum(heatmap, dim = (2, 4))
	depth_map = torch.sum(heatmap, dim = (2, 3))

	height_grid = torch.linspace(0.0, 2.0, height_map.size(-1)).view(1, 1, -1).to(cuda_device)
	width_grid = torch.linspace(0.0, 2.0, width_map.size(-1)).view(1, 1, -1).to(cuda_device)
	depth_grid = torch.linspace(0.0, 2.0, depth_map.size(-1)).view(1, 1, -1).to(cuda_device)

	height = torch.sum(height_grid * height_map, dim = 2)
	width = torch.sum(width_grid * width_map, dim = 2)
	depth = torch.sum(depth_grid * depth_map, dim = 2)

	return torch.stack((width, height, depth), dim = 2) * depth_range


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
	keys += ('score_pck', 'score_auc', 'overall_mean', 'batch_size')

	values = np.array([[patch[key] for patch in scores_and_stats] for key in keys])

	return dict(zip(keys[:-1], np.sum(values[-1] * values[:-1], axis = 1) / total))