export PATH=$PATH:/home/liu/Downloads/libjpeg-turbo/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liu/Downloads/libjpeg-turbo/lib64

python main.py \
				-shuffle \
				-val_only \
				-valid_check \
				-joint_space \
				-do_track \
				-do_attention \
				-model resnet50 \
				-model_path /home/liu/pose_track/models/resnet50.pth \
				-suffix do_attention \
				-data_name cmu \
				-data_root_path /globalwork/data/cmu-panoptic \
				-data_down_path /globalwork/liu/cmu_panoptic_down \
				-comp_name mpii \
				-comp_root_path /globalwork/data/mpii \
				-comp_down_path /globalwork/liu/mpii_down \
				-occluder_path /globalwork/liu/pascal_occluders \
				-save_path /globalwork/liu/pose_track \
				-criterion SmoothL1 \
				-batch_size 64 \
				-n_cudas 1
