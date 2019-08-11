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
	0.5268,
	0.4613,
	3.5125,
	0.3597,
	0.1922,
	0.0778,
	0.6934,
	0.2874,
	0.2256,
	0.3401,
	0.2030,
	0.0838,
	0.6925,
	0.2833,
	0.2213,
	0.4622,
	0.4530,
	0.4642,
	0.4597
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
