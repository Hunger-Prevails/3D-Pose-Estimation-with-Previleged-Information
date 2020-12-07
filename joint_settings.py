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
		('rHip', 'rHip'),
		('lHip', 'lHip'),
		('lKnee', 'lKnee'),
		('lAnkle', 'lAnkle'),
		('Pelvis', 'BodyCenter'),
		('Thorax', 'Neck'),
		('rWrist', 'rWrist'),
		('rElbow', 'rElbow'),
		('rShoulder', 'rShoulder'),
		('lShoulder', 'lShoulder'),
		('lElbow', 'lElbow'),
		('lWrist', 'lWrist')
	]
)

mpii_h36m_match = dict(
	[
		('rAnkle', 'rank'),
		('rKnee', 'rkne'),
		('rHip', 'rhip'),
		('lHip', 'lhip'),
		('lKnee', 'lkne'),
		('lAnkle', 'lank'),
		('Pelvis', 'pelv'),
		('Thorax', 'neck'),
		('rWrist', 'rwri'),
		('rElbow', 'relb'),
		('rShoulder', 'rsho'),
		('lShoulder', 'lsho'),
		('lElbow', 'lelb'),
		('lWrist', 'lwri')
	]
)

h36m_short_names = [
	'rhip',
	'rkne',
	'rank',
	'lhip',
	'lkne',
	'lank',
	'tors',
	'neck',
	'head',
	'htop',
	'lsho',
	'lelb',
	'lwri',
	'rsho',
	'relb',
	'rwri',
	'pelv'
]

h36m_parent = dict(
	[
		('htop', 'head'),
		('head', 'neck'),
		('lsho', 'neck'),
		('lelb', 'lsho'),
		('lwri', 'lelb'),
		('rsho', 'neck'),
		('relb', 'rsho'),
		('rwri', 'relb'),
		('neck', 'tors'),
		('tors', 'pelv'),
		('lhip', 'pelv'),
		('lkne', 'lhip'),
		('lank', 'lkne'),
		('rhip', 'pelv'),
		('rkne', 'rhip'),
		('rank', 'rkne'),
		('pelv', 'pelv')
	]
)

h36m_mirror = dict(
	[
		('lsho', 'rsho'),
		('rsho', 'lsho'),
		('lelb', 'relb'),
		('relb', 'lelb'),
		('lwri', 'rwri'),
		('rwri', 'lwri'),
		('lhip', 'rhip'),
		('rhip', 'lhip'),
		('lkne', 'rkne'),
		('rkne', 'lkne'),
		('lank', 'rank'),
		('rank', 'lank')
	]
)

h36m_cam_names = ['54138969', '55011271', '58860488', '60457274']
h36m_key_foots = [1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27, 0]
h36m_base_joint = 'pelv'

ntu_short_names = [
	'Pelvis',
	'Spine',
	'Neck',
	'Head',
	'rShoulder',
	'rElbow',
	'rWrist',
	'rHand',
	'lShoulder',
	'lElbow',
	'lWrist',
	'lHand',
	'rHip',
	'rKnee',
	'rAnkle',
	'rFoot',
	'lHip',
	'lKnee',
	'lAnkle',
	'lFoot',
	'Clavicle'
]

ntu_parent = dict(
	[
		('Pelvis', 'Pelvis'),
		('Spine', 'Pelvis'),
		('Neck', 'Clavicle'),
		('Head', 'Neck'),
		('rShoulder', 'Clavicle'),
		('rElbow', 'rShoulder'),
		('rWrist', 'rElbow'),
		('rHand', 'rWrist'),
		('lShoulder', 'Clavicle'),
		('lElbow', 'lShoulder'),
		('lWrist', 'lElbow'),
		('lHand', 'lWrist'),
		('rHip', 'Pelvis'),
		('rKnee', 'rHip'),
		('rAnkle', 'rKnee'),
		('rFoot', 'rAnkle'),
		('lHip', 'Pelvis'),
		('lKnee', 'lHip'),
		('lAnkle', 'lKnee'),
		('lFoot', 'lAnkle'),
		('Clavicle', 'Spine')
	]
)
