import os
import torch
import numpy as np

from builtins import zip as xzip

class PoseSample:
	
	def __init__(self, image_path, body_pose, valid, bbox, camera):
		self.image_path = image_path
		self.body_pose = body_pose
		self.valid = valid
		self.bbox = bbox
		self.camera = camera


class JointInfo:
	def __init__(self, short_names, parent, mirror, key_index, weight = None, essence = None):
		self.short_names = short_names
		self.parent = parent
		self.mirror = mirror
		self.key_index = key_index
		self.weight = weight
		self.essence = essence


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

	max_val = torch.max(heatmap, dim = 2, keepdim = True)[0]  # (batch_size, num_joints, 1)

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


def statistics(basic, flip, tangent, thresh):

	dist = dict(
		basic = basic,
		flip = flip,
		tangent = tangent
	)
	def count_and_eliminate(condition):
		remains = np.nonzero(np.logical_not(condition))

		dist['basic'] = dist['basic'][remains]
		dist['flip'] = dist['flip'][remains]
		dist['tangent'] = dist['tangent'][remains]

		return np.count_nonzero(condition)

	count = float(dist['basic'].size)

	keys = ('solid', 'close', 'jitter', 'depth', 'switch', 'fail')

	solid = count_and_eliminate(dist['basic'] <= thresh['solid']) / count
	
	close = count_and_eliminate(dist['basic'] <= thresh['close']) / count
	
	jitter = count_and_eliminate(dist['basic'] <= thresh['rough']) / count
	
	depth = count_and_eliminate(dist['tangent'] <= thresh['close']) / count

	switch = count_and_eliminate(dist['flip'] <= thresh['rough']) / count

	return dict(zip(keys, (solid, close, jitter, depth, switch, dist['basic'].size / count)))


def parse_epoch(stats):

	keys = ('solid', 'close', 'jitter', 'depth', 'switch', 'fail')
	keys += ('score_pck', 'score_auc', 'cam_mean', 'batch_size')

	values = np.array([[patch[key] for patch in stats] for key in keys])

	return dict(zip(keys[:-1], np.sum(values[-1] * values[:-1], axis = 1) / np.sum(values[-1])))


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
	valid = valid_mask.flatten()

	dist = np.linalg.norm(spec_cam - true_cam, axis = -1)
	dist = dist.flatten()[valid]
	
	dist_flip = np.linalg.norm(spec_cam - true_cam[:, mirror], axis = -1)
	dist_flip = dist_flip.flatten()[valid]
	
	dist_tangent = np.linalg.norm(spec_cam[:, :, :2] - true_cam[:, :, :2], axis = -1)
	dist_tangent = dist_tangent.flatten()[valid]

	cam_mean = np.mean(dist)
	score_pck = np.mean(dist / thresh['rough'] <= 1.0)
	score_auc = np.mean(np.maximum(0, 1 - dist / thresh['rough']))

	stats = statistics(dist, dist_flip, dist_tangent, thresh)

	stats.update(
		dict(
			batch_size = dist.shape[0],
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


def get_deter_cam(spec_mat, relat_cam, valid, intrinsics, attention):
	'''
	reconstructs the reference point location at train time.

	Args:
		spec_mat: (batch_size, num_joints, 2) estimated image coordinates
		relat_cam: (batch_size, num_joints, 3) estimated relative camera coordinates with respect to an unknown reference point
		valid: (batch_size, num_joints)
		intrinsics: (batch_size, 3, 3) camera intrinsics
		attention: (batch_size, num_joints) attention weights

	Returns:
		(batch_size, num_joints, 3) estimation of camera coordinates
	'''
	dim_batch, dim_joint = valid.shape

	assert (np.sum(valid, axis = 1) != 0).all()
	assert (np.sum(valid, axis = 1) != 1).all()

	unproject = np.linalg.inv(intrinsics).transpose(0, 2, 1)

	augment = np.ones((dim_batch, dim_joint, 1))  # (batch_size, num_joints, 1)

	normalized = np.concatenate([spec_mat, augment], axis = -1)  # (batch_size, num_joints, 3)

	normalized = np.einsum('bij,bjk->bik', normalized, unproject)[:, :, :2]  # (batch_size, num_joints, 2)

	A = np.tile(np.eye(2), (dim_batch, dim_joint, 1))  # (batch_size, num_joints x 2, 2)

	A = np.concatenate([A, - normalized.reshape(dim_batch, -1, 1)], axis = -1)  # (batch_size, num_joints x 2, 3)

	b = (normalized * relat_cam[:, :, 2:] - relat_cam[:, :, :2]).reshape(dim_batch, -1, 1)  # (batch_size, num_joints x 2, 1)

	attention = (valid * attention ** 0.5).reshape(-1, 1)  # (batch_size x num_joints, 1)
	attention = np.tile(attention, (1, 2)).reshape(dim_batch, -1, 1)  # (batch_size, num_joints x 2, 1)

	A = A * attention
	b = b * attention

	refer = np.linalg.inv(np.einsum('bij,bjk->bik', A.transpose(0, 2, 1), A))  # (batch_size, 3, 3)

	refer = np.einsum('bij,bjk->bik', refer, np.einsum('bij,bjk->bik', A.transpose(0, 2, 1), b))  # (batch_size, 3, 1)

	return relat_cam + refer.transpose(0, 2, 1)


def get_recon_cam(spec_mat, relat_cam, valid, intrinsics, attention):
	'''
	reconstructs the reference point location at train time.

	Args:
		spec_mat: (batch_size, num_joints, 2) estimated image coordinates
		relat_cam: (batch_size, num_joints, 3) estimated relative camera coordinates with respect to an unknown reference point
		valid: (batch_size, num_joints)
		intrinsics: (batch_size, 3, 3) camera intrinsics
		attention: (batch_size, num_joints) attention weights

	Returns:
		(batch_size, num_joints, 3) estimation of camera coordinates
	'''
	dim_batch, dim_joint = valid.size()

	assert (torch.sum(valid, dim = 1) != 0).all()
	assert (torch.sum(valid, dim = 1) != 1).all()

	unproject = torch.inverse(intrinsics).permute(0, 2, 1)

	augment = torch.ones((dim_batch, dim_joint, 1)).to(spec_mat.device)  # (batch_size, num_joints, 1)

	normalized = torch.cat([spec_mat, augment], dim = -1)  # (batch_size, num_joints, 3)

	normalized = torch.einsum('bij,bjk->bik', normalized, unproject)[:, :, :2]  # (batch_size, num_joints, 2)

	A = torch.eye(2).repeat((dim_batch, dim_joint, 1)).to(spec_mat.device)  # (batch_size, num_joints x 2, 2)

	A = torch.cat([A, - normalized.contiguous().view(dim_batch, -1, 1)], dim = -1)  # (batch_size, num_joints x 2, 3)

	b = (normalized * relat_cam[:, :, 2:] - relat_cam[:, :, :2]).view(dim_batch, -1, 1)  # (batch_size, num_joints x 2, 1)

	attention = (valid.float() * attention ** 0.5).view(-1, 1)  # (batch_size x num_joints, 1)
	attention = attention.repeat(1, 2).view(dim_batch, -1, 1)  # (batch_size, num_joints x 2, 1)

	A = A * attention
	b = b * attention

	refer = torch.inverse(torch.einsum('bij,bjk->bik', A.permute(0, 2, 1), A))  # (batch_size, 3, 3)

	refer = torch.einsum('bij,bjk->bik', refer, torch.einsum('bij,bjk->bik', A.permute(0, 2, 1), b))  # (batch_size, 3, 1)

	return relat_cam + refer.permute(0, 2, 1)
