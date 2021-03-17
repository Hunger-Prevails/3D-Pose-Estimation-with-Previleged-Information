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
ntu_mirror = dict(
	[
		('rShoulder', 'lShoulder'),
		('rElbow', 'lElbow'),
		('rWrist', 'lWrist'),
		('rHand', 'lHand'),
		('lShoulder', 'rShoulder'),
		('lElbow', 'rElbow'),
		('lWrist', 'rWrist'),
		('lHand', 'rHand'),
		('rHip', 'lHip'),
		('rKnee', 'lKnee'),
		('rAnkle', 'lAnkle'),
		('rFoot', 'lFoot'),
		('lHip', 'rHip'),
		('lKnee', 'rKnee'),
		('lAnkle', 'rAnkle'),
		('lFoot', 'rFoot')
	]
)
ntu_base_joint = 'Pelvis'
