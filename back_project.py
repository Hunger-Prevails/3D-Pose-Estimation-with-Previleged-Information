import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import os

data_path = '/globalwork/liu/cmu_panoptic'

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


def show_skeleton(image_path, image_coord, confidence):
    image = plt.imread(image_path)

    plt.figure(figsize = (15, 15))

    body_edges = np.array(
                    [
                        [1, 2],
                        [1, 4],
                        [4, 5],
                        [5, 6],
                        [1, 3],
                        [3, 7],
                        [7, 8],
                        [8, 9],
                        [3, 13],
                        [13, 14],
                        [14, 15],
                        [1, 10],
                        [10, 11],
                        [11, 12]
                    ]
                ) - 1

    plt.subplot(1, 3, 1)
    plt.title('3D Body Projection on HD view (' + image_path + ')')
    plt.imshow(image)
    currentAxis = plt.gca()
    currentAxis.set_autoscale_on(False)

    valid = (0.1 <= confidence)

    plt.plot(image_coord[0, valid], image_coord[1, valid], '.')

    for edge in body_edges:
        if valid[edge[0]] and valid[edge[1]]:
            plt.plot(image_coord[0, edge], image_coord[1, edge])

    plt.draw()
    plt.show()


def get_image_coords(seq_names, start_frames, end_frames, interval, n_cams):

    assert len(seq_names) == len(start_frames)
    assert len(seq_names) == len(end_frames)

    for seq_idx, seq_name in enumerate(seq_names):
        skeleton_dir = os.path.join(data_path, seq_name, 'hdPose3d_stage1_coco19')
        image_root = os.path.join(data_path, seq_name, 'hdImgs')

        calib = os.path.join(data_path, seq_name, 'calibration_' + seq_name + '.json')
        calib = json.load(open(calib))

        cameras = [cam for cam in calib['cameras'] if cam['panel'] == 0][:n_cams]

        for cam in cameras:
            cam['K'] = np.matrix(cam['K'])
            cam['distCoef'] = np.array(cam['distCoef'])
            cam['R'] = np.matrix(cam['R'])
            cam['t'] = np.array(cam['t'])

        image_coords = [[] for x in xrange(len(cameras))]

        camera_folders = [os.path.join(image_root, cam['name']) for cam in cameras]

        for frame in xrange(start_frames[seq_idx], end_frames[seq_idx], interval):

            skeleton = os.path.join(skeleton_dir, 'body3DScene_' + str(frame).zfill(8) + '.json')
            skeleton = json.load(open(skeleton))['bodies']
            if not skeleton:
                continue

            skeleton = np.array(skeleton[0]['joints19'])
            skeleton = skeleton.reshape((-1,4)).transpose()  # (4, 19)

            for cam_idx, cam in enumerate(cameras):

                image_coord = projectPoints(skeleton[:3], cam)  # (3, 19)

                # image_path = os.path.join(camera_folders[cam_idx], cam['name'] + '_' + str(frame).zfill(8) + '.jpg')
                # show_skeleton(image_path, image_coord, skeleton[3])

                image_coord = np.concatenate((image_coord[:2], skeleton[3:]), axis = 0)  # (3, 19)
                image_coords[cam_idx].append(image_coord.transpose())  # (19, 3)

            print 'frame [', start_frames[seq_idx], '-', frame, '|', end_frames[seq_idx], '] processed'

        print 'saving collected image coords'

        for cam_idx, cam in enumerate(cameras):
            save_path = os.path.join(image_root, 'image_coord_' + cam['name'] + '.json')
            image_coord = np.stack(image_coords[cam_idx]).tolist()

            with open(save_path, 'w') as file:
                print >> file, json.dumps(
                                dict(
                                    start_frame = start_frames[seq_idx],
                                    end_frame = end_frames[seq_idx],
                                    interval = interval,
                                    image_coord = image_coord))
                file.close()


def main(interval, n_cams, seq_info):
    get_image_coords(seq_info[::3], [int(x) for x in seq_info[1::3]], [int(x) for x in seq_info[2::3]], int(interval), int(n_cams))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
