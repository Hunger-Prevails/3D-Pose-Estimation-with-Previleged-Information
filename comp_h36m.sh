source ../.local/stow/cdf37_0/bin/definitions.B

export PATH=$PATH:/home/liu/Downloads/libjpeg-turbo/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liu/Downloads/libjpeg-turbo/lib64

export CUDA_VISIBLE_DEVICES=0

python main.py \
				-do_company \
				-half_acc \
				-shuffle \
				-save_record \
				-pretrain \
				-joint_space \
				-do_track \
				-model resnet50 \
				-model_path /home/liu/camera_pose/models/resnet50.pth \
				-suffix do_atn_h36m_mpii \
				-data_name h36m \
				-comp_name mpii \
				-data_root_path /globalwork/data/human3.6m \
				-comp_root_path /globalwork/data/mpii \
				-data_down_path /globalwork/liu/h36m_down \
				-comp_down_path /globalwork/liu/mpii_down \
				-save_path /globalwork/liu/pose_track \
				-criterion SmoothL1 \
				-batch_size 32 \
				-learn_rate 2e-5 \
				-n_cudas 1 \
				-n_epochs 30 \
				-num_joints 17
