export PATH=$PATH:/home/liu/Downloads/libjpeg-turbo/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liu/Downloads/libjpeg-turbo/lib64

export CUDA_VISIBLE_DEVICES=0
python main.py \
				-extra_channel \
				-half_acc \
				-shuffle \
				-save_record \
				-pretrain \
				-static_filter \
				-joint_space \
				-do_track \
				-model resnet50 \
				-model_path /home/liu/camera_pose/models/resnet50.pth \
				-suffix do_atn_full_ecoov \
				-data_name cmu \
				-data_root_path /globalwork/data/cmu-panoptic \
				-data_down_path /globalwork/liu/cmu_down \
				-save_path /globalwork/liu/pose_track \
				-criterion SmoothL1 \
				-batch_size 64 \
				-learn_rate 2e-5 \
				-n_cudas 1
