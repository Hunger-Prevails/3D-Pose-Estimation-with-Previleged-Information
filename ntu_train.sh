export PATH=$PATH:/home/liu/Downloads/libjpeg-turbo/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liu/Downloads/libjpeg-turbo/lib64

export CUDA_VISIBLE_DEVICES=0
python3 depth_main.py \
				-shuffle \
				-save_record \
				-half_acc \
				-pretrain \
				-model resnet50 \
				-model_path /home/liu/camera_pose/models/resnet50.pth \
				-suffix debug \
				-data_name ntu \
				-data_root_path /globalwork/data/NTU_RGBD \
				-data_down_path /globalwork/liu/ntu_down \
				-save_path /globalwork/liu/ntu_train \
				-criterion SmoothL1 \
				-num_joints 21 \
				-depth_range 1e3 \
				-loss_div 1e1 \
				-n_cudas 1
