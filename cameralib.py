import copy
import functools

import cv2
import numpy as np
import scipy.optimize
import transforms3d

import boxlib


# TODO: tfcamera


def support_single(f):
    """Makes a function that transforms multiple points accept also a single point"""

    def wrapped(self, points, *args, **kwargs):
        points = np.asarray(points, np.float32)
        ndim = points.ndim
        if ndim == 1:
            return f(self, points[np.newaxis], *args, **kwargs)[0]
        else:
            return f(self, points, *args, **kwargs)
        # else:
        #    raise Exception(f'Wrong number of dimensions in points array: {ndim}, should be 1 or
        #    2')

    return wrapped


# def invalidates_cache(f):
#     """Makes a function that transforms multiple points accept also a single point"""
#
#     def wrapped(self, *args, **kwargs):
#         self.cache
#
#     return wrapped
#
# def cachable(f):
#     def wrapped(self, *args, **kwargs):
#         self.cache[]

class Camera:
    def __init__(
            self, optical_center=None, rot_world_to_cam=None, intrinsic_matrix=np.eye(3),
            distortion_coeffs=None, world_up=(0, 0, 1), extrinsic_matrix=None):
        """Initializes camera.

        The camera coordinate system has the following axes:
          x points to the right
          y points down
          z points forwards

        The world z direction is assumed to point up by default, but `world_up` can also be
         specified differently.

        Args:
            optical_center: position of the camera in world coordinates (eye point)
            rot_world_to_cam: 3x3 rotation matrix for transforming column vectors
                from being expressed in world reference frame to being expressed in camera
                reference frame as follows:
                column_point_cam = rot_matrix_world_to_cam @ (column_point_world - optical_center)
            intrinsic_matrix: 3x3 matrix that maps 3D points in camera space to homogeneous
                coordinates in image (pixel) space. Its last row must be (0,0,1).
            distortion_coeffs: parameters describing radial and tangential lens distortions,
                following OpenCV's model and order: k1, k2, p1, p2, k3 or None,
                if the camera has no distortion.
            world_up: a world vector that is designated as "pointing up", for use when
                the camera wants to roll itself upright.
        """

        if optical_center is not None and extrinsic_matrix is not None:
            raise Exception('At most one of `optical_center` and `extrinsic_matrix` needs to be '
                            'provided!')
        if extrinsic_matrix is not None and rot_world_to_cam is not None:
            raise Exception('At most one of `rot_world_to_cam` and `extrinsic_matrix` needs to be '
                            'provided!')

        if (optical_center is None) and (extrinsic_matrix is None):
            optical_center = np.zeros(3)

        if (rot_world_to_cam is None) and (extrinsic_matrix is None):
            rot_world_to_cam = np.eye(3)

        if extrinsic_matrix is not None:
            self.R = np.asarray(extrinsic_matrix[:3, :3], np.float32)
            self.t = (-self.R.T @ extrinsic_matrix[:3, 3]).astype(np.float32)
        else:
            self.R = np.asarray(rot_world_to_cam, np.float32)
            self.t = np.asarray(optical_center, np.float32)

        self.intrinsic_matrix = np.asarray(intrinsic_matrix, np.float32)
        if distortion_coeffs is None:
            self.distortion_coeffs = None
        else:
            self.distortion_coeffs = np.asarray(distortion_coeffs, np.float32)
        self.world_up = np.asarray(world_up)

        if not np.allclose(self.intrinsic_matrix[2, :], [0, 0, 1]):
            raise Exception(f'Bottom row of camera\'s intrinsic matrix must be (0,0,1), '
                            f'got {self.intrinsic_matrix[2, :]}.')

    @staticmethod
    def create2D(imshape=(0, 0)):
        intrinsics = np.eye(3)
        intrinsics[:2, 2] = [imshape[1] / 2, imshape[0] / 2]
        return Camera([0, 0, 0], np.eye(3), intrinsics, None)

    def rotate(self, yaw=0, pitch=0, roll=0):
        mat = transforms3d.euler.euler2mat(yaw, pitch, roll, 'ryxz').T
        self.R = mat @ self.R

    def absolute_rotate(self, yaw=0, pitch=0, roll=0):
        def unit_vec(v):
            return v / np.linalg.norm(v)

        if self.world_up[0] > self.world_up[1]:
            world_forward = unit_vec(np.cross(self.world_up, [0, 1, 0]))
        else:
            world_forward = unit_vec(np.cross(self.world_up, [1, 0, 0]))
        world_right = np.cross(world_forward, self.world_up)

        R = np.row_stack([world_right, -self.world_up, world_forward]).astype(np.float32)
        mat = transforms3d.euler.euler2mat(-yaw, -pitch, -roll, 'syxz')
        self.R = mat @ R

    @support_single
    def camera_to_image(self, points):
        """Transforms points from 3D camera coordinate space to image space.
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

        return distorted @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]
        """
        # points = np.asarray(points, np.float32)
        # zeros = np.zeros(3, np.float32)
        # return cv2.projectPoints(
        #     np.expand_dims(points, 0), zeros, zeros, self.intrinsic_matrix,
        #     self.distortion_coeffs)[0][:, 0, :]
        #
        # points = np.asarray(points, np.float32)

        if self.distortion_coeffs is not None:
            result = project_points(points, self.distortion_coeffs, self.intrinsic_matrix)
            return result
        else:
            projected = points[:, :2] / points[:, 2:]
            return projected @ self.intrinsic_matrix[:2, :2].T + self.intrinsic_matrix[:2, 2]

        # zeros = np.zeros(3, np.float32)
        # return cv2.projectPoints(
        #     np.expand_dims(points, 0), zeros, zeros, self.intrinsic_matrix,
        #     self.distortion_coeffs)[0][:, 0, :]

    @support_single
    def world_to_camera(self, points):
        points = np.asarray(points, np.float32)
        return (points - self.t) @ self.R.T

    @support_single
    def camera_to_world(self, points):
        points = np.asarray(points, np.float32)
        return points @ np.linalg.inv(self.R).T + self.t

    @support_single
    def world_to_image(self, points):
        return self.camera_to_image(self.world_to_camera(points))

    @support_single
    def image_to_camera(self, points, depth=1):
        points = np.asarray(points, np.float32)

        if self.distortion_coeffs is None:
            new_image_points = ((points - self.intrinsic_matrix[:2, 2]) @
                                np.linalg.inv(self.intrinsic_matrix[:2, :2]).T)
        else:
            new_image_points = cv2.undistortPoints(
                np.expand_dims(points, 0), self.intrinsic_matrix, self.distortion_coeffs,
                None, None, None)

        return cv2.convertPointsToHomogeneous(new_image_points)[:, 0, :] * depth

    @support_single
    def image_to_world(self, points, camera_depth=1):
        return self.camera_to_world(self.image_to_camera(points, camera_depth))

    @support_single
    def is_visible(self, world_points, imsize):
        imsize = np.asarray(imsize)
        cam_points = self.world_to_camera(world_points)
        im_points = self.camera_to_image(cam_points)

        is_within_frame = np.all(np.logical_and(0 <= im_points, im_points < imsize), axis=1)
        is_in_front_of_camera = cam_points[..., 2] > 0
        return np.logical_and(is_within_frame, is_in_front_of_camera)

    def zoom(self, factor):
        """Zooms the camera (factor > 1 makes objects look larger),
        while keeping the principal point fixed (scaling anchor is the principal point)."""
        self.intrinsic_matrix[:2, :2] *= np.expand_dims(factor, -1)

    def scale_output(self, factor):
        """Adjusts the camera such that the images become scaled by `factor`. It's a scaling with
        the origin as anchor point.
        The difference with `self.zoom` is that this method also moves the principal point,
        multiplying its coordinates by `factor`."""
        self.intrinsic_matrix[:2] *= np.expand_dims(factor, -1)

    def undistort(self):
        self.distortion_coeffs = None

    def square_pixels(self):
        """Adjusts the intrinsic matrix such that the pixels correspond to squares on the
        image plane."""
        fx = self.intrinsic_matrix[0, 0]
        fy = self.intrinsic_matrix[1, 1]
        fmean = 0.5 * (fx + fy)
        multiplier = np.array([[fmean / fx, 0, 0], [0, fmean / fy, 0], [0, 0, 1]])
        self.intrinsic_matrix = multiplier @ self.intrinsic_matrix

    def unskew_pixels(self):
        self.intrinsic_matrix[1, 0] = 0
        self.intrinsic_matrix[0, 1] = 0

    def horizontal_flip(self):
        self.R[0] *= -1

    def center_principal_point(self, imshape):
        """Adjusts the intrinsic matrix so that the principal point becomes located at the center
        of an image sized imshape (height, width)"""

        self.intrinsic_matrix[:2, 2] = [imshape[1] / 2, imshape[0] / 2]

    def shift_to_center(self, desired_center_image_point, imshape):
        """Shifts the principal point such that what's currently at `desired_center_image_point`
        will be shown in the image center of an image shaped `imshape`."""

        current_coords_of_the_point = desired_center_image_point
        target_coords_of_the_point = np.float32([imshape[1], imshape[0]]) / 2
        self.intrinsic_matrix[:2, 2] += (
                target_coords_of_the_point - current_coords_of_the_point)

    def shift_to_desired(self, current_coords_of_the_point, target_coords_of_the_point):
        """Shifts the principal point such that what's currently at `desired_center_image_point`
        will be shown in the image center of an image shaped `imshape`."""

        self.intrinsic_matrix[:2, 2] += (
                target_coords_of_the_point - current_coords_of_the_point)

    def turn_towards(self, target_image_point=None, target_world_point=None):
        """Turns the camera so that its optical axis goes through a desired target point.
        It resets any roll or horizontal flip applied previously. The resulting camera
        will not have horizontal flip and will be upright (0 roll)."""

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

    def upright(self):
        """Turns the camera so that its optical axis goes through a desired target point.
        It resets any roll or horizontal flip applied previously. The resulting camera
        will not have horizontal flip and will be upright (0 roll)."""

        def unit_vec(v):
            return v / np.linalg.norm(v)

        new_z = self.R[2]
        new_x = unit_vec(np.cross(new_z, self.world_up))
        new_y = np.cross(new_z, new_x)

        # row_stack because we need the inverse transform (we make a matrix that transforms
        # points from one coord system to another), which is the same as the transpose
        # for rotation matrices.
        self.R = np.row_stack([new_x, new_y, new_z]).astype(np.float32)

    def orbit_around(self, world_point, angle_radians, axis='vertical'):
        """Rotates the camera around a vertical axis passing through `world point` by
        `angle_radians`."""

        # TODO: 1 or -1 in the following line?
        if axis == 'vertical':
            axis = -self.world_up
        else:
            lookdir = self.R[2]
            axis = np.cross(lookdir, self.world_up)

        rot_matrix = cv2.Rodrigues(axis * angle_radians)[0]
        # The eye position rotates simply as any point
        self.t = (rot_matrix @ (self.t - world_point)) + world_point

        # R is rotated by a transform expressed in world coords, so it (its inverse since its a
        # coord transform matrix, not a point transform matrix) is applied on the right.
        # (inverse = transpose for rotation matrices, they are orthogonal)
        self.R = self.R @ rot_matrix.T

    def crop_from(self, point):
        self.intrinsic_matrix[:2, 2] -= point

    def get_projection_matrix(self):
        extrinsic_projection = np.append(self.R, -self.R @ np.expand_dims(self.t, 1), axis=1)
        return self.intrinsic_matrix @ extrinsic_projection

    def get_extrinsic_matrix(self):
        return build_extrinsic_matrix(self.R, self.t)

    def copy(self):
        return copy.deepcopy(self)


def build_extrinsic_matrix(rot_world_to_cam, optical_center_world):
    R = rot_world_to_cam
    t = optical_center_world
    return np.block([[R, -R @ np.expand_dims(t, -1)], [0, 0, 0, 1]])


def camera_in_new_world(camera, new_world_camera):
    new_world_up = new_world_camera.world_to_camera(camera.world_up) - new_world_camera.t
    R = camera.R @ new_world_camera.R.T
    t = new_world_camera.R @ (camera.optical_center - new_world_camera.optical_center)
    return Camera(t, R, camera.intrinsic_matrix, camera.distortion_coeffs, new_world_up)


def reproject_image_points(points, old_camera, new_camera):
    """Transforms keypoints of an image captured with `old_camera` to the corresponding
    keypoints of an image captured with `new_camera`.
    The world position (optical center) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output image."""

    if (old_camera.distortion_coeffs is None and new_camera.distortion_coeffs is None and
            points.ndim == 2):
        return reproject_image_points_fast(points, old_camera, new_camera)

    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            'The optical center of the camera must not change, else warping is not enough!')

    if (np.allclose(new_camera.R, old_camera.R) and
            allclose_or_nones(new_camera.distortion_coeffs, old_camera.distortion_coeffs)):
        relative_intrinsics = (
                new_camera.intrinsic_matrix @ np.linalg.inv(old_camera.intrinsic_matrix))
        return points @ relative_intrinsics[:2, :2].T + relative_intrinsics[:2, 2]

    world_points = old_camera.image_to_world(points)
    return new_camera.world_to_image(world_points)


def reproject_image(
        image, old_camera, new_camera, output_imshape, border_mode=cv2.BORDER_CONSTANT,
        border_value=0, interp=None, antialias=False, dst=None):
    """Transforms an image captured with `old_camera` to look like it was captured by
    `new_camera`. The optical center (3D world position) of the cameras must be the same, otherwise
    we'd have parallax effects and no unambiguous way to construct the output."""

    if old_camera.distortion_coeffs is None and new_camera.distortion_coeffs is None:
        return reproject_image_fast(
            image, old_camera, new_camera, output_imshape, border_mode, border_value, antialias,
            dst)

    if not np.allclose(old_camera.t, new_camera.t):
        raise Exception(
            'The optical center of the camera must not change, else warping is not enough!')

    output_size = (output_imshape[1], output_imshape[0])

    # 1. Simplest case: if only the intrinsics have changed we can use an affine warp
    if (np.allclose(new_camera.R, old_camera.R) and
            allclose_or_nones(new_camera.distortion_coeffs, old_camera.distortion_coeffs)):
        relative_intrinsics_inv = (
                old_camera.intrinsic_matrix @ np.linalg.inv(new_camera.intrinsic_matrix))
        scaling_factor = 1 / np.linalg.norm(relative_intrinsics_inv[:2, 0])
        if interp is None:
            interp = cv2.INTER_LINEAR if scaling_factor > 1 else cv2.INTER_AREA
        return cv2.warpAffine(
            image, relative_intrinsics_inv[:2], output_size, flags=cv2.WARP_INVERSE_MAP | interp,
            borderMode=border_mode, borderValue=border_value)

    # 2. The general case handled by transforming the coordinates of every pixel
    # (i.e. computing the source pixel coordinates for each destination pixel)
    # and remapping (i.e. resampling the image at the resulting coordinates)
    y, x = np.mgrid[:output_imshape[0], :output_imshape[1]].astype(np.float32)
    new_maps = np.stack([x, y], axis=-1)
    newim_coords = new_maps.reshape([-1, 2])

    if new_camera.distortion_coeffs is None:
        partial_homography = (
                old_camera.R @ np.linalg.inv(new_camera.R) @
                np.linalg.inv(new_camera.intrinsic_matrix))
        new_im_homogeneous = cv2.convertPointsToHomogeneous(newim_coords)[:, 0, :]
        old_camera_coords = new_im_homogeneous @ partial_homography.T
        oldim_coords = old_camera.camera_to_image(old_camera_coords)
    else:
        world_coords = new_camera.image_to_world(newim_coords)
        oldim_coords = old_camera.world_to_image(world_coords)

    old_maps = oldim_coords.reshape(new_maps.shape).astype(np.float32)
    # For cv2.remap, we need to provide a grid of lookup pixel coordinates for
    # each output pixel.
    if interp is None:
        interp = cv2.INTER_LINEAR

    if dst is None:
        remapped = cv2.remap(
            image, old_maps, None, interp, borderMode=border_mode, borderValue=border_value)
    else:
        remapped = cv2.remap(
            image, old_maps, None, interp, borderMode=border_mode, borderValue=border_value,
            dst=dst)

    if remapped.ndim < image.ndim:
        return np.expand_dims(remapped, -1)

    return remapped


def get_affine(src_camera, dst_camera):
    """Returns the affine transformation matrix that brings points from src_camera frame
    to dst_camera frame. Only works for in-plane rotations, translation and zoom.
    Throws if the transform would need a homography (due to out of plane rotation)."""

    # Check that the optical center and look direction stay the same
    if (not np.allclose(src_camera.t, dst_camera.t) or
            not np.allclose(src_camera.R[2], dst_camera.R[2])):
        raise Exception(
            'The optical center of the camera and its look '
            'direction may not change in the affine case!')

    src_points = np.array([[0, 0], [1, 0], [0, 1]], np.float32)
    dst_points = reproject_image_points(src_points, src_camera, dst_camera)
    return np.append(cv2.getAffineTransform(src_points, dst_points), [[0, 0, 1]], axis=0)


def undistort_points(cam, points):
    if cam.distortion_coeffs is None:
        return cam.copy(), points

    cam_undistorted = cam.copy()
    cam_undistorted.undistort()
    points_undistorted = reproject_image_points(points, cam, cam_undistorted)
    return cam_undistorted, points_undistorted


def calibrate_extrinsics(points2d, points3d):
    # Based on Hartley-Zisserman: Multiple-View Geometry, p181, Algo 7.1
    if not len(points2d) == len(points3d):
        raise Exception('The point lists must have the same length')

    points2d = np.array(points2d)
    points3d = np.array(points3d)
    n_points = len(points3d)

    hp2d = np.concatenate([points2d, np.ones(n_points, 1)], axis=1)
    hp3d = np.concatenate([points3d, np.ones(n_points, 1)], axis=1)

    def normalize(p):
        s = p.shape[-1]
        mean = np.mean(p, axis=1, keepdims=True)
        std = np.std(p)
        backward_mat = np.eye((s, s))
        backward_mat[:-1, :-1] *= std
        backward_mat[:-1, -1:] = mean
        forward_mat = np.linalg.inv(backward_mat)
        return p @ forward_mat.T, forward_mat, backward_mat

    np2d, forw2d, back2d = normalize(hp2d)
    np3d, forw3d, back3d = normalize(hp3d)

    blocks = [np.outer([0, -1, p2d[1], 1, 0, p2d[0]], p3d) for p2d, p3d in zip(np2d, np3d)]
    A = np.concatenate(blocks, axis=0).reshape(n_points * 2, 12)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    p_linear_estimate = np.reshape(vh[-1], [3, 4])

    R_guess = p_linear_estimate[:3, :3]
    eye_guess = -R_guess.T @ p_linear_estimate[:3, 3:]
    quat_guess = transforms3d.quaternions.mat2quat(R_guess)
    guess = np.concatenate([eye_guess, quat_guess])

    def residual(vec):
        eye = vec[:3]
        quat = vec[3:]
        R = transforms3d.quaternions.quat2mat(quat)
        P = np.append(R, -R @ np.expand_dims(eye, 1), axis=1)
        projected_homog = np3d @ P.T
        projected = projected_homog[:, :2] / projected_homog[:, 2:]
        residuals = (projected - np2d).reshape([-1])
        return residuals

    solution = scipy.optimize.least_squares(residual, x0=guess)
    vec = solution.x
    eye = vec[:3]
    quat = vec[3:]
    R = transforms3d.quaternions.quat2mat(quat)
    P = np.append(R, -R @ np.expand_dims(eye, 1), axis=1)
    P_unnormalized = back2d @ P @ forw3d

    det = np.linalg.det(P[:3, :3])
    return P_unnormalized / det


def triangulate(cameras, pointlists):
    cameras, pointlists = zip(*[undistort_points(c, p) for c, p in zip(cameras, pointlists)])
    proj_matrices = [c.get_projection_matrix() for c in cameras]

    n_points = len(pointlists[0])
    if not all(len(pointlist) == n_points for pointlist in pointlists):
        raise Exception('The point lists must have the same length')

    triangulated = np.empty(shape=(n_points, 3), dtype=np.float32)
    for i in range(n_points):
        points = [pointlist[i] for pointlist in pointlists]
        blocks = [(np.expand_dims(point, 1) @ pr[2:] - pr[:2]) for point, pr in
                  zip(points, proj_matrices)]
        A = np.concatenate(blocks, axis=0)
        _, _, vh = np.linalg.svd(A, full_matrices=False)
        triangulated[i] = vh[3, :3] / vh[3, 3:]

    return triangulated


def triangulate_single(cameras, points):
    cameras, points = zip(*[undistort_points(c, [p]) for c, p in zip(cameras, points)])
    points = [p[0] for p in points]
    proj_matrices = [c.get_projection_matrix() for c in cameras]
    blocks = [(np.expand_dims(point, 1) @ pr[2:] - pr[:2]) for point, pr in
              zip(points, proj_matrices)]
    A = np.concatenate(blocks, axis=0)
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    return vh[3, :3] / vh[3, 3:]


def look_at_box(orig_cam, box, output_side):
    cam = orig_cam.copy()
    center_point = boxlib.center(box)

    if box[2] < box[3]:
        delta_y = np.array([0, box[3] / 2])
        sidepoints = np.stack([center_point - delta_y, center_point + delta_y])
    else:
        delta_x = np.array([box[2] / 2, 0])
        sidepoints = np.stack([center_point - delta_x, center_point + delta_x])

    cam.turn_towards(target_image_point=center_point)
    cam.undistort()
    cam.square_pixels()
    cam_sidepoints = reproject_image_points(sidepoints, orig_cam, cam)
    if box[2] < box[3]:
        crop_side = np.abs(cam_sidepoints[0, 1] - cam_sidepoints[1, 1])
    else:
        crop_side = np.abs(cam_sidepoints[0, 0] - cam_sidepoints[1, 0])
    cam.zoom(output_side / crop_side)
    cam.center_principal_point((output_side, output_side))
    return cam


def look_at_skeleton(orig_cam, cam_coords, output_side):
    world_coords = orig_cam.camera_to_world(cam_coords)
    cam = orig_cam.copy()

    def make_box(camera):
        cam3d_coords = camera.world_to_camera(world_coords)
        im_coords = camera.camera_to_image(cam3d_coords)
        box = boxlib.expand_to_square(boxlib.bb_of_points(im_coords))
        topleft = camera.image_to_world(box[:2], cam3d_coords[-1, 2])
        bottomright = camera.image_to_world(box[:2] + box[2:] / 2, cam3d_coords[-1, 2])
        diag_mm = np.linalg.norm(bottomright - topleft)
        result = boxlib.expand(box, max(1.15, 1700 / diag_mm))
        if np.min(result[2:]) < 1:
            return [0, 0, 5, 5]
        return result

    for i in range(5):
        box = make_box(cam)
        cam = look_at_box(cam, box, output_side)

    return cam


def get_homography(src_camera, dst_camera):
    """Returns the homography matrix that brings points from src_camera frame
    to dst_camera frame. The world position (optical center) of the cameras must be the same,
    otherwise we'd have parallax effects and no unambiguous way to construct the output
    image."""

    # Check that the optical center and look direction stay the same
    if not np.allclose(src_camera.t, dst_camera.t):
        raise Exception(
            'The optical centers of the cameras are different, a homography can not model this!')

    return (src_camera.intrinsic_matrix @ src_camera.R @ np.linalg.inv(dst_camera.R) @
            np.linalg.inv(dst_camera.intrinsic_matrix))


def allclose_or_nones(a, b):
    if a is None and b is None:
        return True

    if a is None:
        return np.min(b) == np.max(b) == 0

    if b is None:
        return np.min(b) == np.max(b) == 0

    return np.allclose(a, b)


def project_points(points, distortion_coeffs, intrinsic_matrix):
    intrinsic_matrix = intrinsic_matrix.astype(np.float32)
    dist_coeff = distortion_coeffs
    points = points.astype(np.float32)
    projected = points[:, :2] / points[:, 2:]
    r_pow2 = np.sum(projected * projected, axis=1)
    r_pow4 = r_pow2 * r_pow2

    distorter = dist_coeff[0] * r_pow2
    distorter += dist_coeff[1] * r_pow4
    r_pow6 = r_pow4
    r_pow6 *= r_pow2
    distorter += dist_coeff[4] * r_pow6
    distorter += np.float32(1.0)
    distorter += projected[:, 0] * (2 * dist_coeff[3])
    distorter += projected[:, 1] * (2 * dist_coeff[2])

    projected[:, 0] *= distorter
    projected[:, 0] += r_pow2 * dist_coeff[3]
    projected[:, 1] *= distorter
    projected[:, 1] += r_pow2 * dist_coeff[2]

    return (projected @ intrinsic_matrix[:2, :2].T + intrinsic_matrix[:2, 2]).astype(np.float32)


@functools.lru_cache()
def get_grid_coords(output_imshape):
    y, x = np.mgrid[:output_imshape[0], :output_imshape[1]].astype(np.float32)
    return np.stack([x, y, np.ones_like(x)], axis=0).reshape([3, -1])


def reproject_image_fast(
        image, old_camera, new_camera, output_imshape, border_mode=None, border_value=None,
        antialias=False, dst=None):
    """Like reproject_image, but assumes no distortions."""

    old_matrix = old_camera.intrinsic_matrix @ old_camera.R
    new_matrix = new_camera.intrinsic_matrix @ new_camera.R
    homography = (old_matrix @ np.linalg.inv(new_matrix)).astype(np.float32)

    if antialias:
        center = np.array([output_imshape[1] / 2, output_imshape[0] / 2, 1])
        down = center + [0, 1, 0]
        right = center + [1, 0, 0]
        center_src, down_src, right_src = np.stack([center, down, right], axis=0) @ homography.T
        y_factor = min(1 / np.linalg.norm(center_src - down_src) * 1.5, 1)
        x_factor = min(1 / np.linalg.norm(center_src - right_src) * 1.5, 1)
        factor = np.sqrt(y_factor * x_factor)
        if factor < 1:
            scaled_size = (int(np.round(factor * image.shape[1])),
                           int(np.round(factor * image.shape[0])))
            image = cv2.resize(
                image, dsize=scaled_size, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)
            homography[:2] *= [[factor], [factor]]

    coords = get_grid_coords(tuple(list(output_imshape)))
    coords = homography @ coords
    coords = coords[:2] / coords[2:]
    coords = coords.reshape(2, output_imshape[0], output_imshape[1])

    if border_mode is None:
        border_mode = cv2.BORDER_CONSTANT
    if border_value is None:
        border_value = 0

    if dst is None:
        remapped = cv2.remap(
            image, *coords, cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value)
    else:
        remapped = cv2.remap(
            image, *coords, cv2.INTER_LINEAR, borderMode=border_mode, borderValue=border_value,
            dst=dst)

    if image.ndim == 2:
        return np.expand_dims(remapped, -1)
    return remapped


def is_all_visible_in_new_camera(image, old_camera, new_camera, output_imshape):
    """Like reproject_image, but assumes no distortions."""
    old_matrix = old_camera.intrinsic_matrix @ old_camera.R
    new_matrix = new_camera.intrinsic_matrix @ new_camera.R
    homography = (old_matrix @ np.linalg.inv(new_matrix)).astype(np.float32)

    coords = get_grid_coords(tuple(list(output_imshape)))
    coords = homography @ coords
    coords = coords[:2] / coords[2:]
    x, y = coords.reshape(2, output_imshape[0], output_imshape[1])
    return np.logical_and(np.logical_and(np.all(0 <= x), np.all(x < image.shape[1])),
                          np.logical_and(np.all(0 <= y), np.all(y < image.shape[0])))


def reproject_image_points_fast(points, old_camera, new_camera):
    old_matrix = old_camera.intrinsic_matrix @ old_camera.R
    new_matrix = new_camera.intrinsic_matrix @ new_camera.R
    homography = (new_matrix @ np.linalg.inv(old_matrix)).astype(np.float32)
    pointsT = homography[:, :2] @ points.T + homography[:, 2:]
    pointsT = pointsT[:2] / pointsT[2:]
    return pointsT.T


def reproject_image_fast_if_visible(image, old_camera, new_camera, output_imshape):
    """Like reproject_image, but assumes no distortions."""

    homography = (
            old_camera.intrinsic_matrix @ old_camera.R @ np.linalg.inv(new_camera.R) @
            np.linalg.inv(new_camera.intrinsic_matrix)).astype(np.float32)

    coords = get_grid_coords(tuple(list(output_imshape)))
    coords = homography @ coords
    coords = coords[:2] / coords[2:]
    coords = coords.reshape(2, output_imshape[0], output_imshape[1])

    x, y = coords
    is_visible = np.logical_and(
        np.logical_and(np.all(0 <= x), np.all(x < image.shape[1])),
        np.logical_and(np.all(0 <= y), np.all(y < image.shape[0])))

    if not is_visible:
        return None

    # print('remapping')
    remapped = cv2.remap(image, *coords, cv2.INTER_LINEAR)
    # print('returning')
    if image.ndim == 2:
        return np.expand_dims(remapped, -1)
    return remapped


@functools.lru_cache()
def get_image_loader():
    return ImageLoader()


class ImageLoader():
    def __init__(self):
        import tensorflow as tf
        with tf.device('/CPU:0'):
            self.encoded_jpeg_t = tf.placeholder(shape=(), dtype=tf.string)
            self.box_t = tf.placeholder(shape=(4,), dtype=tf.int32)
            self.im1_t = tf.io.decode_and_crop_jpeg(
                self.encoded_jpeg_t, self.box_t, channels=3, dct_method='INTEGER_FAST',
                fancy_upscaling=False, ratio=1)
            self.im2_t = tf.io.decode_and_crop_jpeg(
                self.encoded_jpeg_t, self.box_t, channels=3, dct_method='INTEGER_FAST',
                fancy_upscaling=False, ratio=2)
            self.im4_t = tf.io.decode_and_crop_jpeg(
                self.encoded_jpeg_t, self.box_t, channels=3, dct_method='INTEGER_FAST',
                fancy_upscaling=False, ratio=4)
            self.im8_t = tf.io.decode_and_crop_jpeg(
                self.encoded_jpeg_t, self.box_t, channels=3, dct_method='INTEGER_FAST',
                fancy_upscaling=False, ratio=8)
        self.sess = tf.Session()

    def load(self, encoded_jpeg, box):
        box = np.array(box)[[1, 0, 3, 2]]
        return self.sess.run(
            self.encoded_jpeg_t, feed_dict={self.encoded_jpeg_t: encoded_jpeg, self.box_t: box})


def reproject_image_fast_partial(image_path, old_camera, new_camera, output_imshape):
    import jpeg4py
    encoded_jpeg = bytes(jpeg4py.JPEG(image_path).source)
    box = None
    get_image_loader().load(encoded_jpeg, box)
    return
