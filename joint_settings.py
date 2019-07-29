cmu_short_names = [
	'Neck',
	'Nose',
	'BodyCenter',
	'lShoulder',
	'lElbow',
	'lWrist',
	'lHip',
	'lKnee',
	'lAnkle',
	'rShoulder',
	'rElbow',
	'rWrist',
	'rHip',
	'rKnee',
	'rAnkle',
	'lEye',
	'lEar',
	'rEye',
	'rEar'
]
cmu_parent = dict(
	[
		('BodyCenter', 'BodyCenter'),
		('Nose', 'Neck'),
		('lShoulder', 'Neck'),
		('lElbow', 'lShoulder'),
		('lWrist', 'lElbow'),
		('Neck', 'BodyCenter'),
		('lHip', 'BodyCenter'),
		('lKnee', 'lHip'),
		('lAnkle', 'lKnee'),
		('rHip', 'BodyCenter'),
		('rKnee', 'rHip'),
		('rAnkle', 'rKnee'),
		('rShoulder', 'Neck'),
		('rElbow', 'rShoulder'),
		('rWrist', 'rElbow'),
		('lEar', 'lEye'),
		('lEye', 'Nose'),
		('rEar', 'rEye'),
		('rEye', 'Nose')
	]
)
cmu_mirror = dict(
	[
		('lShoulder', 'rShoulder'),
		('rShoulder', 'lShoulder'),
		('lElbow', 'rElbow'),
		('rElbow', 'lElbow'),
		('lWrist', 'rWrist'),
		('rWrist', 'lWrist'),
		('lHip', 'rHip'),
		('rHip', 'lHip'),
		('lKnee', 'rKnee'),
		('rKnee', 'lKnee'),
		('lAnkle', 'rAnkle'),
		('rAnkle', 'lAnkle'),
		('lEar', 'rEar'),
		('rEar', 'lEar'),
		('lEye', 'rEye'),
		('rEye', 'lEye')
	]
)
cmu_weight = [
	0.341,
	0.210,
	4.721,
	0.217,
	0.074,
	0.029,
	1.388,
	0.143,
	0.034,
	0.233,
	0.069,
	0.031,
	1.377,
	0.164,
	0.038,
	0.223,
	0.243,
	0.230,
	0.236
]
cmu_overlook = [
	'lWrist',
	'lAnkle',
	'rWrist',
	'rAnkle'
]
cmu_base_joint = 'BodyCenter'

mpii_short_names = [
	'rAnkle',
	'rKnee',
	'rHip',
	'lHip',
	'lKnee',
	'lAnkle',
	'Pelvis',
	'Thorax',
	'Neck',
	'Head',
	'rWrist',
	'rElbow',
	'rShoulder',
	'lShoulder',
	'lElbow',
	'lWrist'
]
mpii_parent = dict(
	[
		('Neck', 'Neck'),
		('Head', 'Head'),
		('Thorax', 'Thorax'),
		('lShoulder', 'Thorax'),
		('rShoulder', 'Thorax'),
		('lElbow', 'lShoulder'),
		('rElbow', 'rShoulder'),
		('lWrist', 'lElbow'),
		('rWrist', 'rElbow'),
		('Pelvis', 'Thorax'),
		('lHip', 'Pelvis'),
		('rHip', 'Pelvis'),
		('lKnee', 'lHip'),
		('rKnee', 'rHip'),
		('lAnkle', 'lKnee'),
		('rAnkle', 'rKnee'),
	]
)
mpii_mirror = dict(
	[
		('lShoulder', 'rShoulder'),
		('rShoulder', 'lShoulder'),
		('lElbow', 'rElbow'),
		('rElbow', 'lElbow'),
		('lWrist', 'rWrist'),
		('rWrist', 'lWrist'),
		('lHip', 'rHip'),
		('rHip', 'lHip'),
		('lKnee', 'rKnee'),
		('rKnee', 'lKnee'),
		('lAnkle', 'rAnkle'),
		('rAnkle', 'lAnkle'),
	]
)
mpii_base_joint = 'Thorax'

mpii_cmu_match = dict(
	[
		('rAnkle', 'rAnkle'),
		('rKnee', 'rKnee'),
		('lKnee', 'lKnee'),
		('lAnkle', 'lAnkle'),
		('Thorax', 'Neck'),
		('rWrist', 'rWrist'),
		('rElbow', 'rElbow'),
		('rShoulder', 'rShoulder'),
		('lShoulder', 'lShoulder'),
		('lElbow', 'lElbow'),
		('lWrist', 'lWrist')
	]
)
