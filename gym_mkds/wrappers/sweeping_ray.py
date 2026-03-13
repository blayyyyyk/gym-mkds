import gymnasium as gym
import numpy as np
from desmume.emulator_mkds import MarioKart
from desmume.vector import generate_plane_vectors
import trimesh
from typing import Optional
import math


class SweepingRay(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, n_rays: int, min_val: int, max_val: int):
        super().__init__(env)
        self.n_rays = n_rays
        self.min_val = min_val
        self.max_val = max_val
        if isinstance(env.observation_space, gym.spaces.Dict):
            self.observation_space = gym.spaces.Dict({
                "wall_distances": gym.spaces.Box(low=0, high=math.inf, shape=(n_rays,), dtype=np.float32)
            })
        else:
            self.observation_space = gym.spaces.Box(low=0, high=math.inf, shape=(n_rays,), dtype=np.float32)
    
    def observation(self, observation):
        emu: MarioKart = self.get_wrapper_attr('emu')
        if not emu.memory.race_ready:
            return {
                "wall_distances": np.zeros((self.n_rays,), dtype=np.float32)
            }

        # boundary extraction
        boundary_lines = get_track_lines(emu)
        B, T, C = boundary_lines.shape
        boundary_lines = boundary_lines.reshape(B*T, C)

        # surface info
        position = emu.memory.driver.position
        mtx = emu.memory.driver.mainMtx[:3, :].T
        normal = mtx[1].cpu().numpy()

        # ray generation
        ray_origin, ray_direction = generate_plane_vectors(self.n_rays, 180, mtx, position)
        ray_direction= ray_direction.cpu().numpy()

        position = position.cpu().numpy()

        # projection to local surface
        boundary_lines_uv = project_to_plane(boundary_lines, normal, position)
        boundary_lines_uv = boundary_lines_uv.reshape(B, T, C-1)
        ray_direction_uv = project_to_plane(ray_direction + position[None, :], normal, position)

        # collect ray intersections
        ray_origin_uv = np.zeros_like(ray_direction_uv)
        t = raycast_2d(boundary_lines_uv, ray_origin_uv, ray_direction_uv).clip(self.min_val, self.max_val)
        if isinstance(self.observation_space, gym.spaces.Dict) and isinstance(observation, dict):
            return {
                "wall_distances": t
            }
            
        return t


def get_track_lines(emu: MarioKart) -> np.ndarray:
    col_data = emu.memory.collision_data
    col_type = col_data["prism_attribute"]["collision_type"]
    road_mask = (col_type == 0)

    v1 = col_data["v1"].cpu().numpy()
    v2 = col_data["v2"].cpu().numpy()
    v3 = col_data["v3"].cpu().numpy()
    triangles = np.stack([v1, v2, v3], axis=1)

    threshold = 10
    triangles = np.round(triangles / threshold) * threshold
    boundary_lines = find_boundary_lines(triangles, road_mask)
    return boundary_lines

def find_boundary_lines(triangles: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    """
    Finds shared lines between road and non-road triangles.
    """
    # build mesh
    raw_vertices = triangles.reshape(-1, 3)
    raw_faces = np.arange(len(raw_vertices)).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=raw_vertices, faces=raw_faces, process=True, maintain_order=True)

    # face adjacency lists
    f1 = mesh.face_adjacency[:, 0]
    f2 = mesh.face_adjacency[:, 1]

    # one face is road and neighboring face is off-road
    boundary_mask = road_mask[f1] != road_mask[f2]

    # vertex indices for those specific boundary edges
    boundary_edge_indices = mesh.face_adjacency_edges[boundary_mask]

    # map the indices back to 3D coordinates
    boundary_lines = mesh.vertices[boundary_edge_indices] # (E, 2, 3)

    return boundary_lines


def project_to_plane(points: np.ndarray, normal: np.ndarray, plane_point: Optional[np.ndarray] = None):
    """
    Projects an (N, 3) array of points onto a plane.

    Args:
        points: np.ndarray of shape (N, 3).
        normal: np.ndarray of shape (3,) representing the plane's normal vector.
        plane_point: np.ndarray of shape (3,) for a point on the plane. Defaults to origin.

    Returns:
        projected_3d: (N, 3) array of the points flattened onto the plane.
        local_2d: (N, 2) array of the points in the plane's local coordinate system.
    """
    if plane_point is None:
        plane_point = np.zeros(3)

    # normalize
    n_hat = normal / np.linalg.norm(normal)
    v = points - plane_point # compute delta
    dists = np.dot(v, n_hat)
    projected_3d = points - (dists[:, np.newaxis] * n_hat)
    axis_idx = np.argmin(np.abs(n_hat))
    arbitrary_vec = np.zeros(3)
    arbitrary_vec[axis_idx] = 1.0

    # create orthoganol basis
    u_vec = np.cross(n_hat, arbitrary_vec)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(n_hat, u_vec) # this is assumed orthoganol...
    
    # project onto plane, uv coords
    vectors_on_plane = projected_3d - plane_point
    u_coords = np.dot(vectors_on_plane, u_vec)
    v_coords = np.dot(vectors_on_plane, v_vec)
    local_2d = np.column_stack((u_coords, v_coords))
    return local_2d


def unproject_from_plane(local_2d: np.ndarray, normal: np.ndarray, plane_point: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Converts an (N, 2) array of local plane coordinates back to global (N, 3) coordinates.

    Args:
        local_2d: np.ndarray of shape (N, 2) representing (U, V) coordinates.
        normal: np.ndarray of shape (3,) representing the plane's normal vector.
        plane_point: np.ndarray of shape (3,) for a point on the plane. Defaults to origin.

    Returns:
        points_3d: (N, 3) array of the points in global 3D space.
    """
    if plane_point is None:
        plane_point = np.zeros(3)

    # normalize
    n_hat = normal / np.linalg.norm(normal)

    # recompute basis vectors
    axis_idx = np.argmin(np.abs(n_hat))
    arbitrary_vec = np.zeros(3)
    arbitrary_vec[axis_idx] = 1.0
    u_vec = np.cross(n_hat, arbitrary_vec)
    u_vec /= np.linalg.norm(u_vec)
    v_vec = np.cross(n_hat, u_vec)

    # map uv coords to global coord space
    u_coords = local_2d[:, 0, np.newaxis]
    v_coords = local_2d[:, 1, np.newaxis]

    # scale basis vectors
    points_3d = plane_point + (u_coords * u_vec) + (v_coords * v_vec)
    return points_3d


def raycast_2d(segments: np.ndarray, ray_origins: np.ndarray, ray_dirs: np.ndarray) -> np.ndarray:
    """
    Casts N rays against M line segments and returns the minimum t distance for each ray.

    Args:
        segments: (M, 2, 2) array of line segments [Endpoint_A, Endpoint_B].
        ray_origins: (N, 2) or (2,) array of ray start points.
        ray_dirs: (N, 2) array of ray direction vectors.

    Returns:
        min_t: (N,) array of distances to the closest segment. Misses are marked as np.inf.
    """
    # split segment endpoints
    A = segments[:, 0, :] # (M, 2)
    B = segments[:, 1, :]
    S = B - A

    # conform ray shapes
    ray_dirs = np.atleast_2d(ray_dirs)
    ray_origins = np.atleast_2d(ray_origins)
    if len(ray_origins) == 1 and len(ray_dirs) > 1:
        ray_origins = np.broadcast_to(ray_origins, ray_dirs.shape)

    P = ray_origins
    R = ray_dirs
    P_exp = P[:, np.newaxis, :] # (N, 1, 2)
    R_exp = R[:, np.newaxis, :] # (N, 1, 2)
    A_exp = A[np.newaxis, :, :] # (1, M, 2)
    S_exp = S[np.newaxis, :, :] # (1, M, 2)

    delta = A_exp - P_exp # (N, M, 2)
    def cross_2d(v, w):
        return v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]

    den = cross_2d(R_exp, S_exp)     # (N, M)
    num_t = cross_2d(delta, S_exp)   # (N, M)
    num_u = cross_2d(delta, R_exp)   # (N, M)

    # solve for t and u (we only need t tho)
    old_settings = np.seterr(divide='ignore', invalid='ignore') # ignore stupid error
    t = num_t / den
    u = num_u / den
    np.seterr(**old_settings)

    # intersection rules
    epsilon = 1e-8
    valid_hits = (np.abs(den) > epsilon) & (t >= 0) & (u >= 0) & (u <= 1)

    # invalid hit mask
    t_valid = np.where(valid_hits, t, np.inf)

    # return minimum distance ray
    min_t = np.min(t_valid, axis=1)  # (N,)
    return min_t

