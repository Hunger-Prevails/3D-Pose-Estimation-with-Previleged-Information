import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os


data_path = '/globalwork/data/cmu-panoptic'


def projectPoints(X, cam):
    """
    Projects points X (3xN) using camera intrinsics K (3x3),
    extrinsics (R,t) and distortion parameters Kd=[k1,k2,p1,p2,k3].

    Roughly, x = K*(R*X + t) + distortion
    """
    K = cam['K']
    R = cam['R']
    t = cam['t']
    Kd = cam['distCoef']

    x = np.asarray(R*X + t)

    x[0:2,:] = x[0:2,:]/x[2,:]

    r = x[0,:]*x[0,:] + x[1,:]*x[1,:]

    x[0,:] = x[0,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[2]*x[0,:]*x[1,:] + Kd[3]*(r + 2*x[0,:]*x[0,:])
    x[1,:] = x[1,:]*(1 + Kd[0]*r + Kd[1]*r*r + Kd[4]*r*r*r) + 2*Kd[3]*x[0,:]*x[1,:] + Kd[2]*(r + 2*x[1,:]*x[1,:])

    x[0,:] = K[0,0]*x[0,:] + K[0,1]*x[1,:] + K[0,2]
    x[1,:] = K[1,0]*x[0,:] + K[1,1]*x[1,:] + K[1,2]

    return x


def show_skeleton(image, image_coord, confidence, message = '', bbox = None):
    '''
    Shows coco19 skeleton(mat)

    Args:
        image: path to image
        image_coord: (2, num_joints)
        confidence: (num_joints,)
    '''
    image = plt.imread(image) if isinstance(image, str) else image

    plt.figure(figsize = (12, 8))

    from joint_settings import cmu_short_names as short_names
    from joint_settings import cmu_parent as parent

    mapper = dict(zip(short_names, range(len(short_names))))

    body_edges = [mapper[parent[name]] for name in short_names]
    body_edges = np.hstack(
        [
            np.arange(len(body_edges)).reshape(-1, 1),
            np.array(body_edges).reshape(-1, 1)
        ]
    )
    ax = plt.subplot(1, 1, 1)
    plt.title(message + ':' + str(image.shape))
    plt.imshow(image)
    ax.set_autoscale_on(False)

    valid = (0.1 <= confidence)

    plt.plot(image_coord[0, valid], image_coord[1, valid], '.')

    for edge in body_edges:
        if valid[edge[0]] and valid[edge[1]]:
            plt.plot(image_coord[0, edge], image_coord[1, edge])

    plt.plot(np.mean(image_coord[0, valid]), np.mean(image_coord[1, valid]), 'X', color = 'w')

    if bbox is not None:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth = 2, edgecolor = 'r', facecolor = 'none')
        ax.add_patch(rect)

    plt.draw()
    plt.show()


def get_image_coords(seq_name, start_frame, end_frame, interval):
    pose_folder = os.path.join(data_path, seq_name, 'hdPose3d_stage1_coco19')
    image_root = os.path.join(data_path, seq_name, 'hdImgs')

    cam_folders = [os.path.join(image_root, folder) for folder in os.listdir(image_root)]
    cam_folders = [folder for folder in cam_folders if os.path.isdir(folder)]
    cam_folders.sort()

    cam_names = [os.path.basename(folder) for folder in cam_folders]

    cam_folders = dict(zip(cam_names, cam_folders))
    image_coords = dict([(name, []) for name in cam_names])

    calib = os.path.join(data_path, seq_name, 'calibration_' + seq_name + '.json')
    calib = json.load(open(calib))

    cameras = [cam for cam in calib['cameras'] if cam['panel'] == 0]

    for cam in cameras:
        cam['K'] = np.matrix(cam['K'])
        cam['distCoef'] = np.array(cam['distCoef'])
        cam['R'] = np.matrix(cam['R'])
        cam['t'] = np.array(cam['t'])

    cameras = dict([(cam['name'], cam) for cam in cameras if cam['name'] in cam_names])

    for frame in xrange(start_frame, end_frame, interval):
        bodies = os.path.join(pose_folder, 'body3DScene_' + str(frame).zfill(8) + '.json')
        bodies = json.load(open(bodies))['bodies']
        
        if not bodies:
            continue

        for skeleton in bodies:
            skeleton = np.array(skeleton['joints19'])
            skeleton = skeleton.reshape((-1,4)).transpose()  # (4 x 19)

            for name in cam_names:
                image_coord = projectPoints(skeleton[:3], cameras[name])  # (3 x 19)

                image_coord = np.concatenate((image_coord[:2], skeleton[3:]), axis = 0)  # (3 x 19)

                image_coords[name].append(image_coord.transpose())  # (19 x 3)

        print 'frame [', start_frame, '-', frame, '|', end_frame, '] processed'

    print 'saving collected image coords'

    for name in cam_names:
        save_path = os.path.join(image_root, 'image_coord_' + name + '.json')
        image_coord = np.stack(image_coords[name]).tolist()

        with open(save_path, 'w') as file:
            file.write(
                json.dumps(
                    dict(
                        start_frame = start_frame,
                        end_frame = end_frame,
                        interval = interval,
                        image_coord = image_coord
                    )
                )
            )
            file.close()


def main(interval, seq_name, start_frame, end_frame):
    get_image_coords(seq_name, int(start_frame), int(end_frame), int(interval))


if __name__ == '__main__':
    assert len(sys.argv[1:]) == 4
    main(*sys.argv[1:])
