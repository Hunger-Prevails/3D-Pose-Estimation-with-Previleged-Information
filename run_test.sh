export PATH=$PATH:/home/liu/Downloads/libjpeg-turbo/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/liu/Downloads/libjpeg-turbo/lib64

python main.py \
                                -shuffle \
                                -test_only \
                                -model resnet50 \
                                -model_path /home/liu/pose_volumetric/models/resnet50.pth \
                                -suffix baseline \
                                -data_source cmu_panoptic \
                                -root_path /globalwork/liu/cmu_panoptic \
                                -root_down /globalwork/liu/cmu_panoptic_down \
                                -occluder_path /globalwork/liu/pascal_occluders \
                                -save_path /globalwork/liu/pose_track \
                                -criterion SmoothL1
