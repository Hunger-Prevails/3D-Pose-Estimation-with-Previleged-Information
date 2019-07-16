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
	'rEye',
	'lEye',
	'rEar',
	'lEar'
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
	0.34112873,
	0.2097354,
	4.72104386,
	0.21656642,
	0.07430816,
	0.02899673,
	1.38761942,
	0.14282832,
	0.03364499,
	0.23299864,
	0.06865945,
	0.03135558,
	1.37655607,
	0.16439174,
	0.03809168,
	0.22265177,
	0.24344002,
	0.22986239,
	0.23612062
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
