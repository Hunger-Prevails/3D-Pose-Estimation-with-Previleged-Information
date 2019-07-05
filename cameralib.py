import cv2
import numpy as np
import transforms3d


def support_single(f):
    """
    Enables a function that transforms multiple points to accept also a single point
    """

    def wrapped(self, points, *args, **kwargs):
        if np.array(points).ndim == 2:
            return f(self, points, *args, **kwargs)
        else:
            return f(self, [points], *args, **kwargs)[0]

    return wrapped


class Camera:
    def __init__(
            self, optical_center, rot_matrix, intrinsic_matrix, distortion_coeffs, world_up = (0, 0, 1)):
        """
        Initializes camera.

        The camera coordinate system has the following axes:
          x points to the right
          y points down
          z points forwards

        The world z direction is assumed to point up by default, but 'world_up' can also be
         specified differently.

        Args:
            optical_center: position of the camera in world coordinates (eye point)
            rot_matrix: 3x3 rotation matrix for transforming column vectors
                from being expressed in world reference frame to being expressed in camera
                reference frame as follows:
                column_point_cam = rot_matrix @ (column_point_world - optical_center)
            intrinsic_matrix: 3x3 matrix that maps 3D points in camera space to homogeneous
                coordinates in image (pixel) space. Its last row must be (0,0,1).
            distortion_coeffs: parameters describing radial and tangential lens distortions,
                following OpenCV's model and order: k1, k2, p1, p2, k3 or None,
                if the camera has no distortion.
            world_up: a world vector that is designated as "pointing up", for use when
                the camera wants to roll itself upright.
        """
        self.R = np.asarray(rot_matrix, np.float32)
        self.t = np.asarray(optical_center.flatten(), np.float32)

        # self._extrinsic_matrix = build_extrinsic_matrix(self.R, self.t)

        self.intrinsics = np.asarray(intrinsic_matrix, np.float32)
        if distortion_coeffs is None:
            self.distortion_coeffs = None
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, np.float32)

        self.world_up = np.asarray(world_up)

        if not np.allclose(self.intrinsics[2, :], [0, 0, 1]):
            raise Exception('Bottom row of intrinsic matrix should be (0,0,1), got ' + str(self.intrinsics[2, :]) + 'instead.')

    @staticmethod
    def create2D(imshape):
        intrinsics = np.eye(3)
        intrinsics[:2, 2] = [imshape[1] / 2, imshape[0] / 2]
        return Camera([0, 0, 0], np.eye(3), intrinsics, None)

    def rotate(self, yaw=0, pitch=0, roll=0):
        mat = transforms3d.euler.euler2mat(-yaw, -pitch, -roll, 'syxz')
        self.R = np,matmul(mat, self.R)

    @support_single
    def camera_to_image(self, points):
        """
        Transforms points from 3D camera coordinate space to image space.
        The steps involved are:
            1. Projection
            2. Distortion (radial and tangential)
            3. Applying focal length and principal point (intrinsic matrix)

        Equivalently:

        projected = points[:, :2] / points[:, 2:]

        if self.distortion_coeffs is not None:
            r2 = np.sum(projected[:, :2] ** 2, axis=1, keepdims=True)

            k = self.distortion_coeffs[[0, 1, 4]]
            radial = 1 + np.hstack([r2, r2 ** 2, r2 ** 3]) @ k

            p_flipped = self.distortion_coeffs[[3, 2]]
            tagential = projected @ (p_flipped * 2)
            distorted = projected * np.expand_dims(radial + tagential, -1) + p_flipped * r2
        else:
            distorted = projected

        return distorted @ self.intrinsics[:2, :2].T + self.intrinsics[:2, 2]
        """

        zeros = np.zeros(3, np.float32)
        return cv2.projectPoints(
            np.expand_dims(points, 0), zeros, zeros, self.intrinsics,
            self.distortion_coeffs)[0][:, 0, :]

    @support_single
    def world_to_camera(self, points):
        points = np.asarray(points, np.float32)
        return np.matmul(points - self.t, self.R.T)

    @support_single
    def camera_to_world(self, points):
        points = np.asarray(points, np.float32)
        return np.matmul(points, self.R) + self.t

    @support_single
    def world_to_image(self, points):
        return self.camera_to_image(self.world_to_camera(points))

    @support_single
    def image_to_camera(self, points):
        points = np.expand_dims(np.asarray(points, np.float32), 0)
        new_image_points = cv2.undistortPoints(
            points, self.intrinsics, self.distortion_coeffs, None, None, None)
        return cv2.convertPointsToHomogeneous(new_image_points)[:, 0, :]

    @support_single
    def image_to_world(self, points):
        return self.camera_to_world(self.image_to_camera(points))

    def is_visible(self, world_points, imshape):
        imshape = np.asarray(imshape)
        cam_points = self.world_to_camera(world_points)
        im_points = self.camera_to_image(cam_points)

        is_within_frame = np.all(np.logical_and(0 <= im_points, im_points < imshape), axis=1)
        is_in_front_of_camera = cam_points[..., 2] > 0
        return np.logical_and(is_within_frame, is_in_front_of_camera)

    def zoom(self, factor):
        """
        Zooms the camera (factor > 1 makes objects look larger)
        while keeping the principal point fixed (scaling anchor is the principal point).
        """
        self.intrinsics[:2, :2] *= factor

    def scale_output(self, factor):
        """
        Adjusts the camera such that the images become scaled by 'factor'. It's a scaling with
        the origin as anchor point.
        The difference with 'self.zoom' is that this method also moves the principal point,
        multiplying its coordinates by 'factor'.
        """
        self.intrinsics[:2] *= factor

    def undistort(self):
        self.distortion_coeffs = None

    def square_pixels(self):
        """
        Adjusts the intrinsic matrix such that the pixels correspond to squares on the
        image plane.
        """
        fx = self.intrinsics[0, 0]
        fy = self.intrinsics[1, 1]
        fmean = 0.5 * (fx + fy)
        multiplier = np.array([[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]])
        self.intrinsics = np.matmul(multiplier, self.intrinsics)

    def horizontal_flip(self):
        self.R[0] *= -1

    def center_principal(self, imshape):
        """
        Adjusts the intrinsic matrix so that the principal point becomes located at the center
        of an image sized imshape (height, width)
        """
        self.intrinsics[:2, 2] = [imshape[1] / 2, imshape[0] / 2]

    def shift_to_center(self, desired_center, imshape):
        """
        Shifts the principal point such that what's currently at 'desired_center'
        will be shown in the image center of an image shaped 'imshape'.
        """
        self.intrinsics[:2, 2] -= (
                desired_center - np.float32([imshape[1], imshape[0]]) / 2)

    def turn_towards(self, target_image_point=None, target_world_point=None):
        """
        Turns the camera so that its optical axis goes through a desired target point.
        It resets any roll or horizontal flip applied previously. The resulting camera
        will not have horizontal flip and will be upright (0 roll).
        """
        assert (target_image_point is None) != (target_world_point is None)
        if target_image_point is not None:
            target_world_point = self.image_to_world(target_image_point)

        def unit_vec(v):
            return v / np.linalg.norm(v)

        new_z = unit_vec(target_world_point - self.t)
        new_x = unit_vec(np.cross(new_z, self.world_up))
        new_y = np.cross(new_z, new_x)

        # row_stack because we need the inverse transform (we make a matrix that transforms
        # points from one coord system to another), which is the same as the transpose
        # for rotation matrices.
        self.R = np.row_stack([new_x, new_y, new_z]).astype(np.float32)

    def horizontal_orbit_around(self, world_point, angle_radians):
        """
        Rotates the camera around a vertical axis passing through 'world point' by 'angle_radians'.
        """

        # TODO: 1 or -1 in the following line?
        rot_matrix = cv2.Rodrigues(-self.world_up * angle_radians)[0]
        # The eye position rotates simply as any point
        self.t = np.matmul(rot_matrix, self.t - world_point) + world_point

        # R is rotated by a transform expressed in world coords, so it (its inverse since its a
        # coord transform matrix, not a point transform matrix) is applied on the right.
        # (inverse = transpose for rotation matrices, they are orthogonal)
        self.R = np.matmul(self.R, rot_matrix.T)


def build_extrinsic_matrix(rot_world_to_cam, optical_center_world):
    R = rot_world_to_cam
    t = optical_center_world
    return np.block([[R, np.matmul(-R, t.expand_dims(-1))], [0, 0, 0, 1]])


def camera_in_new_world(camera, new_world_camera):
    new_world_up = new_world_camera.world_to_camera(camera.world_up)
    R = np.matmul(camera.R, new_world_camera.R.T)
    t = np.matmul(new_world_camera.R, camera.optical_center - new_world_camera.optical_center)
    return Camera(t, R, camera.intrinsic_matrix, camera.distortion_coeffs, new_world_up)


def reproject_points(points, old_camera, new_camera):
    """
    Transforms keypoints of an image captured with 'old_camera' to the corresponding
    keypoints of an image captured with 'new_camera'.
    The world position (optical center) of the cameras must be the same, otherwise
    we'd have parallax effects and no unique way to construct the output image.
    """
    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            'The optical center of the camera must not change, else warping is not enough!')

    world_points = old_camera.image_to_world(points)
    return new_camera.world_to_image(world_points)


def reproject_image(
        image, old_camera, new_camera, output_imshape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0):
    """
    Transforms an image captured with 'old_camera' to look like it was captured by
    'new_camera'.
    The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unique way to construct the output.
    """
    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            'The optical center of the camera must not change, else warping is not enough!')

    output_size = (output_imshape[1], output_imshape[0])

    # 1. Simplest case: if only the intrinsics have changed we can use an affine warp
    if (np.allclose(new_camera.R, old_camera.R) and
            allclose_or_nones(new_camera.distortion_coeffs, old_camera.distortion_coeffs)):
        relative_intrinsics = np.matmul(old_camera.intrinsic_matrix, np.linalg.inv(new_camera.intrinsic_matrix))
        return cv2.warpAffine(
            image, relative_intrinsics[:2], output_size, flags=cv2.WARP_INVERSE_MAP)
        # borderMode=border_mode, borderValue=border_value)

    # 2. If the new camera has no distortions we can use cv2.initUndistortRectifyMap
    if new_camera.distortion_coeffs is None:
        relative_rotation = np.matmul(new_camera.R, old_camera.R.T)
        map1, map2 = cv2.initUndistortRectifyMap(
            old_camera.intrinsic_matrix, old_camera.distortion_coeffs, relative_rotation,
            new_camera.intrinsic_matrix, output_size, cv2.CV_32FC1)
        return cv2.remap(
            image, map1, map2, cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)

    # 3. The general case (i.e. new distortions) are handled by dense transformation and remapping.
    y, x = np.mgrid[0:output_imshape[0], 0:output_imshape[1]].astype(np.float32)
    new_maps = np.stack([x, y], axis=-1)
    newim_coords = new_maps.reshape([-1, 2])
    world_coords = new_camera.image_to_world(newim_coords)
    oldim_coords = old_camera.world_to_image(world_coords)
    old_maps = oldim_coords.reshape(new_maps.shape)
    # For cv2.remap, we need to provide a grid of lookup pixel coordinates for
    # each output pixel.
    return cv2.remap(
        image, old_maps, None, cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)


def get_affine(src_camera, dst_camera):
    """
    Returns the affine transformation matrix that brings points from src_camera frame
    to dst_camera frame. Only works for in-plane rotations, translation and zoom.
    Throws if the transform would need a homography (due to out of plane rotation).
    """
    # Check that the optical center and look direction stay the same
    if (not np.allclose(src_camera.t, dst_camera.t) or
            not np.allclose(src_camera.R[2], dst_camera.R[2])):
        raise Exception(
            'The optical center of the camera and its look '
            'direction may not change in the affine case!')

    src_points = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    dst_points = reproject_image_points(src_points, src_camera, dst_camera)
    return np.append(cv2.getAffineTransform(src_points, dst_points), [[0, 0, 1]], axis=0)


def get_homography(src_camera, dst_camera):
    """
    Returns the homography matrix that brings points from src_camera frame
    to dst_camera frame. 
    The world position (optical center) of the cameras must be the same,
    otherwise we'd have parallax effects and no unique way to construct the output
    image.
    """
    # Check that the optical center and look direction stay the same
    if (not np.allclose(src_camera.t, dst_camera.t) or
            not np.allclose(src_camera.R[2], dst_camera.R[2])):
        raise Exception(
            'The optical centers of the cameras are different, a homography can not model this!')

    src_points = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], np.float32)
    dst_points = reproject_image_points(src_points, src_camera, dst_camera)
    return cv2.findHomography(src_points, dst_points, method=0)[0]


def allclose_or_nones(a, b):
    if a is None and b is None:
        return True

    if a is None or b is None:
        return False

    return np.allclose(a, b)
