import torch

def to_heatmap(ausgabe, num_joints, height, width):

	heatmap = ausgabe.view(-1, num_joints, height * width)  # (batch_size, num_joints, height x width)

	max_val = torch.max(heatmap, dim = 2, keepdim = True)[0]  # (batch_size, num_joints)

	heatmap = torch.exp(heatmap - max_val)  # (batch_size, num_joints, height x width)

	heatmap /= torch.sum(heatmap, dim = 2, keepdim = True)  # (batch_size, num_joints, height x width)

	return heatmap.view(-1, num_joints, height, width)


def decode(heatmap, map_range):

	heat_x = torch.sum(heatmap, dim = 2)  # (batch_size, num_joints, width)
	heat_y = torch.sum(heatmap, dim = 3)  # (batch_size, num_joints, height)

	grid_x = torch.linspace(0.0, 1.0, heat_x.size(-1), device = heat_x.device).view(1, 1, -1)
	grid_y = torch.linspace(0.0, 1.0, heat_z.size(-1), device = heat_z.device).view(1, 1, -1)

	coord_x = torch.sum(grid_x * heat_x, dim = -1)  # (batch_size, num_joints)
	coord_y = torch.sum(grid_y * heat_y, dim = -1)  # (batch_size, num_joints)

	return torch.stack((coord_x, coord_y), dim = 2) * map_range  # (batch_size, num_joints, 2)


def coord_to_area(true_mat, box_margin):
	'''
	utilizes true image coords to compute relative scale of a pose instance in terms of its area

	Args:
		true_mat: (batch_size, num_joints, 2)
	'''
	x_min = np.min(image_coord[:, :, 0], axis = -1)  # (batch_size,)
	x_max = np.max(image_coord[:, :, 0], axis = -1)  # (batch_size,)
	y_min = np.min(image_coord[:, :, 1], axis = -1)  # (batch_size,)
	y_max = np.max(image_coord[:, :, 1], axis = -1)  # (batch_size,)

	return (x_max - x_min) * (y_max - y_min) / box_margin / box_margin


def analyze(spec_mat, true_mat, valid_mask, box_margin):
	'''
	Analyzes spec_mat against true_mat under current camera setting

	Args:
		spec_mat: (batch_size, num_joints, 2)
		true_mat: (batch_size, num_joints, 2)
		valid_mask: (batch_size, num_joints)

	Returns:
		dict containing batch_size mean and statistics
	'''
	dist = np.sum((spec_mat - true_mat) ** 2, axis = -1)  # (batch_size, num_joints)

	mat_mean = np.mean(dist ** 0.5)

	area = coord_to_area(true_mat, box_margin)  # (batch_size,)

	oks = np.exp(- dist / area.expand_dims(axis = -1) / 2)  # (batch_size, num_joints)

	oks = np.sum(oks * valid_mask, axis = -1) / np.sum(valid_mask, axis = -1)  # (batch_size,)

	return dict(
		mat_mean = mat_mean,
		score_oks = np.mean(oks),
		batch_size = spec_mat.shape[0]
	)


def parse_epoch(scores, total):

	keys = ('score_oks', 'mat_mean', 'batch_size')

	values = np.array([[patch[key] for patch in scores] for key in keys])

	return dict(zip(keys[:-1], np.sum(values[-1] * values[:-1], axis = 1) / total))