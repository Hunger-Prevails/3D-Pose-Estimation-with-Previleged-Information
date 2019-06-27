cmu_panoptic_short_names = [
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
cmu_panoptic_parent = dict(
	[
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
cmu_panoptic_mirror = dict(
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
cmu_panoptic_base_joint = 'BodyCenter'

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
		('Neck', 'Thorax'),
		('Head', 'Neck'),
		('lShoulder', 'Neck'),
		('rShoulder', 'Neck'),
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
