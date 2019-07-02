import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')

# bool options
parser.add_argument('-shuffle', action='store_true', help='Reshuffle data at each epoch')
parser.add_argument('-save_record', action='store_true', help='Path to save train record')
parser.add_argument('-test_only', action='store_true', help='only performs test')
parser.add_argument('-val_only', action='store_true', help='only performs validation')
parser.add_argument('-pretrained', action='store_true', help='Loads a pretrained model')
parser.add_argument('-resume', action='store_true', help='Continues from a previous checkpoint')
parser.add_argument('-flip_test', action='store_true', help='whether to perform flip test')
parser.add_argument('-do_perturbate', action='store_true', help='whether to perform perturbation augmentation')
parser.add_argument('-do_occlude', action='store_true', help='whether to perform occlusion augmentation')
parser.add_argument('-do_complement', action='store_true', help='whether to use complement dataset')
parser.add_argument('-valid_check', action='store_true', help='whether to threshold uncertain keypoints off')
parser.add_argument('-joint_space', action='store_true', help='whether to allow joint-space train data')

# required options
parser.add_argument('-model', required=True, help='Backbone architecture')
parser.add_argument('-model_path', required=True, help='Path to pretrained model')
parser.add_argument('-suffix', required=True, help='Model suffix')
parser.add_argument('-data_name', required=True, help='name of dataset')
parser.add_argument('-comp_name', required=True, help='name of complement dataset')
parser.add_argument('-data_root_path', required=True, help='Root path to dataset')
parser.add_argument('-comp_root_path', required=True, help='Root path to complement dataset')
parser.add_argument('-data_down_path', required=True, help='Root path to post images of the dataset')
parser.add_argument('-comp_down_path', required=True, help='Root path to post images of the complement dataset')
parser.add_argument('-occluder_path', required=True, help='Root path to occluders')
parser.add_argument('-save_path', required=True, help='Path to save train record')
parser.add_argument('-criterion', required=True, help='Type of objective function')

# integer options
parser.add_argument('-n_epochs', default=20, type=int, help='Training epochs')
parser.add_argument('-batch_size', default=48, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-n_cudas', default=2, type=int, help='Number of cuda devices for training')
parser.add_argument('-workers', default=6, type=int, help='Number of subprocesses to to load data')
parser.add_argument('-num_processes', default=6, type=int, help='Number of subprocesses in the process pool')
parser.add_argument('-side_in', default=257, type=int, help='side of input image')
parser.add_argument('-stride', default=16, type=int, help='stride of network for train')
parser.add_argument('-num_joints', default=19, type=int, help='number of joints in the dataset')
parser.add_argument('-depth', default=16, type=int, help='depth side of volumetric heatmap')

# train options
parser.add_argument('-learn_rate', default=4e-5, type=float, help='Base learning rate of training')
parser.add_argument('-grad_norm', default=5.0, type=float, help='norm for gradient clip')
parser.add_argument('-momentum', default=0.9, type=float, help='Momentum for training')
parser.add_argument('-weight_decay', default=4e-5, type=float, help='Weight decay for training')
parser.add_argument('-box_margin', default=0.8, type=float, help='scale factor for pseudo bbox')
parser.add_argument('-comp_loss_weight', default=0.2, type=float, help='loss weight for complement train samples')

# evaluation options
parser.add_argument('-valid_thresh', default=0.1, type=float, help='threshold for valid joints')
parser.add_argument('-score_thresh', default=5.0, type=float, help='threshold for score analysis')
parser.add_argument('-perfect_thresh', default=1.0, type=float, help='threshold for perfect prediction')
parser.add_argument('-good_thresh', default=3.0, type=float, help='threshold for good prediction')
parser.add_argument('-jitter_thresh', default=10.0, type=float, help='threshold for jittered prediction')
parser.add_argument('-depth_range', default=100.0, type=float, help='depth range of prediction')

# augmentation options
parser.add_argument('-crop_factor_train', default=1.0, type=float, help='crop factor for train')
parser.add_argument('-crop_factor_test', default=1.0, type=float, help='crop factor for test')
parser.add_argument('-chance_occlude', default=0.8, type=float, help='chance for performing an occlusion augmentation')
parser.add_argument('-random_zoom', default=0.9, type=float, help='scale for random zoom operation')

args = parser.parse_args()
