import os
import torch
import numpy as np
import helpers

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


def to_heatmap(ausgabe, depth, num_joints, height, width, unimodal = False):
	'''
	performs axis permutation and numerically stable softmax to output feature map
	suppresses non-neighbors of the maximum if unimodal

	args:
		ausgabe: (batch_size, depth x num_joints, height, width)

	returns:
		volumetric heatmap of shape(batch_size, num_joints, height, width, depth)
	'''
	heatmap = ausgabe.view(-1, depth, num_joints, height, width)
	heatmap = heatmap.permute(0, 2, 3, 4, 1).contiguous()
	
	heatmap = heatmap.view(-1, num_joints, height * width * depth)

	max_val, max_idx = torch.max(heatmap, dim = 2, keepdim = True)  # (batch_size, num_joints)

	def fetch_idx(max_idx, dim_width, dim_depth):

		depth = max_idx % dim_depth

		width = ((max_idx - depth) / dim_depth) % dim_width

		height = (((max_idx - depth) / dim_depth) - width) / dim_width

		return torch.stack((height, width, depth), dim = -1)
	
	if unimodal:
		heatmap = heatmap.view(-1, num_joints, height, width, depth)

		max_idx = helpers.fetch_idx(max_idx, width, depth)  # (batch_size, num_joints, 3)
		max_idx = max_idx.unsqueeze(2)
		max_idx = max_idx.unsqueeze(2)
		max_idx = max_idx.unsqueeze(2)

		mesh_grid = helpers.mesh_grid(heatmap.size())[2:]
		mesh_grid = torch.stack(mesh_grid, dim = -1).to(max_idx.device)  # (batch_size, num_joints, height, width, depth, 3)

		neighborhood = torch.sum((dist - mesh_grid) ** 2, dim = -1) <= 3  # (batch_size, num_joints, height, width, depth)

		heatmap[neighborhood] = 0

		heatmap = heatmap.view(-1, num_joints, height * width * depth)

	heatmap = torch.exp(heatmap - max_val)
	heatmap = heatmap / torch.sum(heatmap, dim = 2, keepdim = True)
	
	return heatmap.view(-1, num_joints, height, width, depth)


def to_coordinate(heatmap, side_eingabe, depth_range, intrinsics, key_depth):
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
	heat_y = torch.sum(heatmap, dim = (3, 4))
	heat_x = torch.sum(heatmap, dim = (2, 4))
	heat_z = torch.sum(heatmap, dim = (2, 3))

	grid_y = torch.linspace(0.0, 1.0, heat_y.size(-1)).view(1, 1, -1).to(heat_y.device)
	grid_x = torch.linspace(0.0, 1.0, heat_x.size(-1)).view(1, 1, -1).to(heat_x.device)
	grid_z = torch.linspace(0.0, 2.0, heat_z.size(-1)).view(1, 1, -1).to(heat_z.device)

	height = torch.sum(grid_y * heat_y, dim = 2) * side_eingabe
	width = torch.sum(grid_x * heat_x, dim = 2) * side_eingabe
	depth = torch.sum(grid_z * heat_z, dim = 2) * depth_range - depth_range

	extension = torch.ones_like(depth, device = depth.device)

	prediction = torch.einsum('bij,bcj->bci', intrinsics, torch.stack((width, height, extension), dim = -1))  # (batch_size, num_joints, 3)
	return prediction * (depth + key_depth).unsqueeze(-1)  # (batch_size, num_joints, 3)


def decode(heatmap, depth_range):
	'''
	performs position interpolation over each axis
	'''
	heat_y = torch.sum(heatmap, dim = (3, 4))
	heat_x = torch.sum(heatmap, dim = (2, 4))
	heat_z = torch.sum(heatmap, dim = (2, 3))

	grid_y = torch.linspace(0.0, 2.0, heat_y.size(-1)).view(1, 1, -1).to(heat_y.device)
	grid_x = torch.linspace(0.0, 2.0, heat_x.size(-1)).view(1, 1, -1).to(heat_x.device)
	grid_z = torch.linspace(0.0, 2.0, heat_z.size(-1)).view(1, 1, -1).to(heat_z.device)

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
	keys += ('score_pck', 'score_auc', 'overall_mean', 'batch_size')

	values = np.array([[patch[key] for patch in scores_and_stats] for key in keys])

	return dict(zip(keys[:-1], np.sum(values[-1] * values[:-1], axis = 1) / total))