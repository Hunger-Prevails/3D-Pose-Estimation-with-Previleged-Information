import argparse

parser = argparse.ArgumentParser(description='Parser for all the training options')

# bool options
parser.add_argument('-shuffle', action='store_true', help='Reshuffle data at each epoch')
parser.add_argument('-half_acc', action='store_true', help='whether to use float16 for speed-up')
parser.add_argument('-save_record', action='store_true', help='Path to save train record')
parser.add_argument('-test_only', action='store_true', help='only performs test')
parser.add_argument('-val_only', action='store_true', help='only performs validation')
parser.add_argument('-pretrain', action='store_true', help='whether to load an imagenet pre-train')
parser.add_argument('-resume', action='store_true', help='whether to continue from a previous checkpoint')
parser.add_argument('-out_of_view', action='store_true', help='whether to train cam regressor on out-of-view keypoints')
parser.add_argument('-extra_channel', action='store_true', help='whether to append an extra channel that masks the bbox')
parser.add_argument('-confid_filter', action='store_true', help='whether to threshold uncertain keypoints off')
parser.add_argument('-static_filter', action='store_true', help='whether to threshold static frames off')
parser.add_argument('-joint_space', action='store_true', help='whether to allow joint-space train data')
parser.add_argument('-do_track', action='store_true', help='whether to track cam coords via least square optim')
parser.add_argument('-do_attention', action='store_true', help='whether to learn weights for reference joint regression')
parser.add_argument('-do_company', action='store_true', help='whether to use complement dataset')

# augmentation options
parser.add_argument('-geometry', action='store_true', help='whether to perform geometry augmentation')
parser.add_argument('-colour', action='store_true', help='whether to perform colour augmentation')
parser.add_argument('-eraser', action='store_true', help='whether to perform eraser augmentation')
parser.add_argument('-occluder', action='store_true', help='whether to perform occluder augmentation')

# required options
parser.add_argument('-model', required=True, help='Backbone architecture')
parser.add_argument('-model_path', required=True, help='Path to an imagenet pre-train or checkpoint')
parser.add_argument('-suffix', required=True, help='Model suffix')
parser.add_argument('-data_name', required=True, help='name of dataset')
parser.add_argument('-comp_name', required=True, help='name of complement dataset')
parser.add_argument('-data_root_path', required=True, help='Root path to dataset')
parser.add_argument('-comp_root_path', required=True, help='Root path to complement dataset')
parser.add_argument('-data_down_path', required=True, help='Root path to crop images of the dataset')
parser.add_argument('-comp_down_path', required=True, help='Root path to crop images of the complement dataset')
parser.add_argument('-occluder_path', required=True, help='Root path to occluders')
parser.add_argument('-save_path', required=True, help='Path to save train record')
parser.add_argument('-criterion', required=True, help='Type of objective function')

# integer options
parser.add_argument('-n_epochs', default=20, type=int, help='Training epochs')
parser.add_argument('-batch_size', default=64, type=int, help='Size of mini-batches for each iteration')
parser.add_argument('-n_cudas', default=2, type=int, help='Number of cuda devices available')
parser.add_argument('-workers', default=2, type=int, help='Number of subprocesses to to load data')
parser.add_argument('-num_processes', default=6, type=int, help='Number of subprocesses in the process pool')
parser.add_argument('-side_in', default=257, type=int, help='side of input image')
parser.add_argument('-stride', default=16, type=int, help='stride of network for train')
parser.add_argument('-num_joints', default=19, type=int, help='number of joints in the dataset')
parser.add_argument('-num_valid', default=7, type=int, help='minimum number of valid joints for a valid sample')
parser.add_argument('-depth', default=16, type=int, help='depth side of volumetric heatmap')

# train options
parser.add_argument('-learn_rate', default=5e-5, type=float, help='Base learning rate for train')
parser.add_argument('-grad_norm', default=5.0, type=float, help='norm for gradient clip')
parser.add_argument('-grad_scaling', default=32.0, type=float, help='magnitude of loss scaling when performing float16 computation')
parser.add_argument('-momentum', default=0.9, type=float, help='Momentum for training')
parser.add_argument('-weight_decay', default=4e-5, type=float, help='Weight decay for training')
parser.add_argument('-box_margin', default=0.6, type=float, help='factor for generating pseudo bbox from image coords')
parser.add_argument('-comp_loss_weight', default=0.5, type=float, help='loss weight for complement train samples')
parser.add_argument('-depth_range', default=100.0, type=float, help='depth range of prediction')
parser.add_argument('-random_zoom', default=0.9, type=float, help='scale for random zoom operation')

# evaluation options
parser.add_argument('-thresh_confid', default=0.1, type=float, help='threshold for a confident annotation')
parser.add_argument('-thresh_static', default=10.0, type=float, help='threshold for a subject to be deemed as static')
parser.add_argument('-thresh_solid', default=0.5, type=float, help='threshold for a solid estimation')
parser.add_argument('-thresh_close', default=2.0, type=float, help='threshold for a close estimation')
parser.add_argument('-thresh_rough', default=5.0, type=float, help='threshold for a rough estimation')

args = parser.parse_args()
