export PATH=$PATH:/home/liu/Downloads/libjpeg-turbo/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liu/Downloads/libjpeg-turbo/lib64

export CUDA_VISIBLE_DEVICES=0
python main.py \
				-shuffle \
				-val_only \
				-confid_filter \
				-static_filter \
				-joint_space \
				-do_track \
				-do_attention \
				-model resnet50 \
				-model_path /home/liu/pose_track/models/resnet50.pth \
				-suffix do_atn_full_baseline \
				-data_name cmu \
				-comp_name mpii \
				-data_root_path /globalwork/data/cmu-panoptic \
				-comp_root_path /globalwork/data/mpii \
				-data_down_path /globalwork/liu/cmu_down \
				-comp_down_path /globalwork/liu/mpii_down \
				-occluder_path /globalwork/liu/pascal_occluders \
				-save_path /globalwork/liu/pose_track \
				-criterion SmoothL1 \
				-batch_size 64 \
				-n_cudas 1
