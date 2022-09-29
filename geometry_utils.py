import numpy as np

from constants import CA_C_DIST, N_CA_DIST, N_CA_C_ANGLE


def rotation_matrix(angle, axis):
    """
    Args:
        angle: (n,)
        axis: 0=x, 1=y, 2=z
    Returns:
        (n, 3, 3)
    """
    n = len(angle)
    R = np.eye(3)[None, :, :].repeat(n, axis=0)

    axis = 2 - axis
    start = axis // 2
    step = axis % 2 + 1
    s = slice(start, start + step + 1, step)

    R[:, s, s] = np.array(
        [[np.cos(angle), (-1) ** (axis + 1) * np.sin(angle)],
         [(-1) ** axis * np.sin(angle), np.cos(angle)]]
    ).transpose(2, 0, 1)
    return R


def get_bb_transform(n_xyz, ca_xyz, c_xyz):
    """
    Compute translation and rotation of the canoncical backbone frame (triangle N-Ca-C) from a position with
    Ca at the origin, N on the x-axis and C in the xy-plane to the global position of the backbone frame

    Args:
        n_xyz: (n, 3)
        ca_xyz: (n, 3)
        c_xyz: (n, 3)

    Returns:
        quaternion represented as array of shape (n, 4)
        translation vector which is an array of shape (n, 3)
    """

    translation = ca_xyz
    n_xyz = n_xyz - translation
    c_xyz = c_xyz - translation

    # Find rotation matrix that aligns the coordinate systems
    #    rotate around y-axis to move N into the xy-plane
    theta_y = np.arctan2(n_xyz[:, 2], -n_xyz[:, 0])
    Ry = rotation_matrix(theta_y, 1)
    n_xyz = np.einsum('noi,ni->no', Ry.transpose(0, 2, 1), n_xyz)

    #    rotate around z-axis to move N onto the x-axis
    theta_z = np.arctan2(n_xyz[:, 1], n_xyz[:, 0])
    Rz = rotation_matrix(theta_z, 2)
    # n_xyz = np.einsum('noi,ni->no', Rz.transpose(0, 2, 1), n_xyz)

    #    rotate around x-axis to move C into the xy-plane
    c_xyz = np.einsum('noj,nji,ni->no', Rz.transpose(0, 2, 1),
                      Ry.transpose(0, 2, 1), c_xyz)
    theta_x = np.arctan2(c_xyz[:, 2], c_xyz[:, 1])
    Rx = rotation_matrix(theta_x, 0)

    # Final rotation matrix
    R = np.einsum('nok,nkj,nji->noi', Ry, Rz, Rx)

    # Convert to quaternion
    # q = w + i*u_x + j*u_y + k * u_z
    quaternion = rotation_matrix_to_quaternion(R)

    return quaternion, translation


def get_bb_coords_from_transform(ca_coords, quaternion):
    """
    Args:
        ca_coords: (n, 3)
        quaternion: (n, 4)
    Returns:
        backbone coords (n*3, 3), order is [N, CA, C]
        backbone atom types as a list of length n*3
    """
    R = quaternion_to_rotation_matrix(quaternion)
    bb_coords = np.tile(np.array(
        [[N_CA_DIST, 0, 0],
         [0, 0, 0],
         [CA_C_DIST * np.cos(N_CA_C_ANGLE), CA_C_DIST * np.sin(N_CA_C_ANGLE), 0]]),
        [len(ca_coords), 1])
    bb_coords = np.einsum('noi,ni->no', R.repeat(3, axis=0), bb_coords) + ca_coords.repeat(3, axis=0)
    bb_atom_types = [t for _ in range(len(ca_coords)) for t in ['N', 'C', 'C']]

    return bb_coords, bb_atom_types


def quaternion_to_rotation_matrix(q):
    """
    x_rot = R x

    Args:
        q: (n, 4)
    Returns:
        R: (n, 3, 3)
    """
    # Normalize
    q = q / (q ** 2).sum(1, keepdims=True) ** 0.5

    # https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.stack([
        np.stack([1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w,
                  2 * x * z + 2 * y * w], axis=1),
        np.stack([2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2,
                  2 * y * z - 2 * x * w], axis=1),
        np.stack([2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w,
                  1 - 2 * x ** 2 - 2 * y ** 2], axis=1)
    ], axis=1)

    return R


def rotation_matrix_to_quaternion(R):
    """
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Args:
        R: (n, 3, 3)
    Returns:
        q: (n, 4)
    """

    t = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    r = np.sqrt(1 + t)
    w = 0.5 * r
    x = np.sign(R[:, 2, 1] - R[:, 1, 2]) * np.abs(
        0.5 * np.sqrt(1 + R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2]))
    y = np.sign(R[:, 0, 2] - R[:, 2, 0]) * np.abs(
        0.5 * np.sqrt(1 - R[:, 0, 0] + R[:, 1, 1] - R[:, 2, 2]))
    z = np.sign(R[:, 1, 0] - R[:, 0, 1]) * np.abs(
        0.5 * np.sqrt(1 - R[:, 0, 0] - R[:, 1, 1] + R[:, 2, 2]))

    return np.stack((w, x, y, z), axis=1)
