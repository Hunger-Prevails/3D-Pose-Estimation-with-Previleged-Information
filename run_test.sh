export PATH=$PATH:/home/liu/Downloads/libjpeg-turbo/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liu/Downloads/libjpeg-turbo/lib64

python main.py \
                                -shuffle \
                                -val_only \
				-valid_check \
				-joint_space \
				-do_track \
                                -model resnet50 \
                                -model_path /home/liu/pose_volumetric/models/resnet50.pth \
                                -suffix small_batch \
                                -data_name cmu \
                                -data_root_path /globalwork/liu/cmu_panoptic \
                                -data_down_path /globalwork/liu/cmu_panoptic_down \
				-comp_name mpii \
                                -comp_root_path /globalwork/data/mpii \
                                -comp_down_path /globalwork/liu/mpii_down \
                                -occluder_path /globalwork/liu/pascal_occluders \
                                -save_path /globalwork/liu/pose_track \
                                -criterion SmoothL1
