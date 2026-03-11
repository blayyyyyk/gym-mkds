from functools import reduce
from typing import Any, Callable, Generic, Optional, TypedDict, TypeVar, cast

import cairo
import gymnasium as gym
import numpy as np
import torch
import trimesh
from desmume.emulator_mkds import MarioKart
from scipy.spatial.distance import cdist

COLOR_MAP = [
    # --- DRIVEABLE SURFACES (Grays/Whites) ---
    [128, 128, 128],  # 0:  Road (Standard Gray)
    [200, 230, 255],  # 1:  Slippery Road (Icy Blue-White)
    # --- OFFROAD (Greens/Browns) ---
    [34, 139, 34],  # 2:  Weak Offroad (Forest Green - Grass)
    [139, 69, 19],  # 3:  Offroad (Saddle Brown - Mud/Dirt)
    # --- TECHNICAL (Purples) ---
    [238, 130, 238],  # 4:  Sound Trigger (Violet)
    # --- HEAVY OFFROAD ---
    [80, 50, 20],  # 5:  Heavy Offroad (Dark Brown - Deep Mud)
    # --- SLIPPERY VARIANTS ---
    [175, 238, 238],  # 6:  Slippery Road 2 (Pale Turquoise)
    # --- BOOSTS (Oranges) ---
    [255, 140, 0],  # 7:  Boost Panel (Dark Orange)
    # --- WALLS (Reds) ---
    [255, 0, 0],  # 8:  Wall (Pure Red)
    [255, 105, 180],  # 9:  Invisible Wall (Hot Pink - distinct from normal wall)
    # --- BOUNDARIES (Blacks/Darks) ---
    [0, 0, 0],  # 10: Out of Bounds (Black)
    [25, 25, 112],  # 11: Fall Boundary (Midnight Blue - Abyss)
    # --- JUMPS (Yellows) ---
    [255, 255, 0],  # 12: Jump Pad (Yellow)
    # --- AI/DRIVER LOGIC (Tinted Grays) ---
    [169, 169, 169],  # 13: Road (no drivers) (Dark Gray)
    [139, 0, 0],  # 14: Wall (no drivers) (Dark Red)
    # --- MECHANICS (Metals/Indigos) ---
    [75, 0, 130],  # 15: Cannon Activator (Indigo)
    [205, 92, 92],  # 16: Edge Wall (Indian Red)
    # --- WATER ---
    [0, 0, 255],  # 17: Falls Water (Pure Blue)
    # --- BOOST VARIANT ---
    [255, 69, 0],  # 18: Boost Pad w/ Min Speed (Red-Orange)
    # --- SPECIAL ROADS ---
    [192, 192, 192],  # 19: Loop Road (Silver)
    [255, 215, 0],  # 20: Special Road (Gold - e.g., Rainbow Road segments)
    # --- WALL VARIANT ---
    [128, 0, 0],  # 21: Wall 3 (Maroon)
    # --- RECALC ---
    [0, 255, 0],  # 22: Force Recalc (Lime Green - Debug visual)
]

def draw_points(
    ctx: cairo.Context,
    pts: np.ndarray,
    colors: np.ndarray,
    radius_scale: float | np.ndarray,
):
    if isinstance(radius_scale, float):
        radius_scale = radius_scale * np.array(1)

    if pts.ndim == 1:
        pts = pts[None, :]

    if colors.ndim == 1:
        colors = colors[None, :]

    assert colors.shape[0] == 1 or colors.shape[0] == pts.shape[0]
    if colors.shape[0] == 1:
        colors = colors.repeat(pts.shape[0], axis=0)

    for (x, y, z), (r, g, b) in zip(pts, colors):
        ctx.set_source_rgb(r, g, b)
        ctx.arc(x, y, radius_scale * z, 0, 2 * np.pi)
        ctx.fill()


def draw_lines(
    ctx: cairo.Context,
    pts1: np.ndarray,
    pts2: np.ndarray,
    colors: np.ndarray,
    stroke_width_scale=1.0,
):
    if pts1.ndim == 1:
        pts1 = pts1[None, :]

    if pts2.ndim == 1:
        pts2 = pts2[None, :]

    assert (
        pts2.shape[0] == pts1.shape[0]
    ), "All point arrays must have the same batch size"

    if colors.ndim == 1:
        colors = colors[None, :]

    assert colors.shape[0] == 1 or colors.shape[0] == pts1.shape[0]
    if colors.shape[0] == 1:
        colors = colors.repeat(pts1.shape[0], axis=0)

    for p1, p2, (r, g, b) in zip(pts1, pts2, colors):
        ctx.set_source_rgb(r, g, b)
        ctx.set_line_width(stroke_width_scale)
        ctx.move_to(*p1[:2])
        ctx.line_to(*p2[:2])
        ctx.stroke()


def draw_triangles(
    ctx: cairo.Context,
    pts1: np.ndarray,
    pts2: np.ndarray,
    pts3: np.ndarray,
    colors: np.ndarray,
):
    n = pts1.shape[0]
    assert (
        pts2.shape[0] == n and pts3.shape[0] == n
    ), "All point arrays must have the same batch size"

    colors = np.asarray(colors)
    if colors.ndim == 1:
        colors = colors[None, :]
    assert colors.shape[1] == 3, "colors must have 3 channels (RGB)"
    if colors.shape[0] == 1:
        colors = np.repeat(colors, n, axis=0)
    else:
        assert colors.shape[0] == n, "colors must be [1,3] or [N,3]"

    l1 = np.concatenate([pts2, pts3, pts1], axis=0)  # p2->p3, p3->p1, p1->p2
    l2 = np.concatenate([pts3, pts1, pts2], axis=0)
    c3 = np.tile(colors, (3, 1))

    draw_lines(ctx, l1, l2, c3)  # assumes draw_lines accepts NumPy arrays



class OverlayWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(OverlayWrapper, self).__init__(env)
        self.depth_mask = True
        
    def __call__(self, env: gym.Env) -> gym.Env:
        return self.__class__(env)

    def _project(self, *pts: np.ndarray, colors: np.ndarray, depth_mask=True) -> tuple[np.ndarray, ...]:
        B, C = pts[0].shape
        pts_cat = np.concat(pts, axis=0)
        emu: MarioKart = self.get_wrapper_attr('emu')
        proj = emu.memory.project_to_screen(torch.tensor(pts_cat, dtype=torch.float32), normalize_depth=True)
        proj_mask = proj["mask"].view(len(pts), B).all(dim=0)
        proj_pts = proj["screen"].view(len(pts), B, C)
        proj_pts = proj_pts[:, proj_mask, :] if depth_mask else proj_pts
        proj_colors = colors[proj_mask] if depth_mask else colors
        return tuple([x.squeeze(0).cpu().numpy() for x in proj_pts.chunk(len(pts), dim=0)]) + (proj_colors,)
        
    def _compute(self) -> tuple[np.ndarray, ...]:
        ...

    def render(self):
        if self.render_mode == "rgb_array":
            raw_rgb = cast(np.ndarray, super().render())
            emu: MarioKart = self.get_wrapper_attr('emu')
            if not emu.memory.race_ready:
                return raw_rgb

            h, w, _ = raw_rgb.shape
            arr = np.zeros((h, w, 4), dtype=np.uint8)
            arr[:, :, :3] = raw_rgb

            surface = cairo.ImageSurface.create_for_data(arr, cairo.FORMAT_RGB24, w, h)
            ctx = cairo.Context(surface)

            out = self._compute()
            if len(out) == 2:
                pts, colors = out
                pts, colors = self._project(pts, colors=colors, depth_mask=self.depth_mask)
                draw_points(ctx, pts[0], colors, radius_scale=5.0)
            elif len(out) == 3:
                pts1, pts2, colors = out
                pts1, pts2, colors = self._project(pts1, pts2, colors=colors, depth_mask=self.depth_mask)
                draw_lines(ctx, pts1, pts2, colors)
            elif len(out) == 4:
                pts1, pts2, pts3, colors = out
                pts1, pts2, pts3, colors = self._project(pts1, pts2, pts3, colors=colors, depth_mask=self.depth_mask)
                draw_triangles(ctx, pts1, pts2, pts3, colors)
            else:
                raise ValueError("Overlay wrapper draw function must return tuple of size 2, 3, or 4")

            surface.flush()
            return arr[:, :, :3]
            
class CollisionPrisms(OverlayWrapper):
    def _compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        emu = self.get_wrapper_attr('emu')
        v1 = emu.memory.collision_data["v1"].to(emu.device)
        v2 = emu.memory.collision_data["v2"].to(emu.device)
        v3 = emu.memory.collision_data["v3"].to(emu.device)
        tri = torch.stack([v1, v2, v3], dim=0).cpu().numpy()
    
        # material color
        color_map = np.array(COLOR_MAP, dtype=np.uint8)
        collision_type = emu.memory.collision_data["prism_attribute"]["collision_type"]
        floor_mask = (
            emu.memory.collision_data["prism_attribute"]["is_floor"] == 1
        )
        wall_mask = emu.memory.collision_data["prism_attribute"]["is_wall"] != 1
        collision_type = collision_type[floor_mask & wall_mask]
        colors = color_map[collision_type]
        tri = tri[:, floor_mask & wall_mask]
        v1, v2, v3 = np.unstack(tri)
        return v1, v2, v3, colors
        
class RaySweep(OverlayWrapper):
    # disable depth mask
    def __init__(self, env: gym.Env):
        super(RaySweep, self).__init__(env)
        self.depth_mask = False
    
    def _compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        emu: MarioKart = self.get_wrapper_attr('emu')
        P = emu.memory.driver.position.to(emu.device)
        P = P.unsqueeze(0)
        max_dist = emu.max_dist
        n_rays = emu.n_rays
        info = emu.memory.obstacle_info(
            n_rays=n_rays, max_dist=max_dist, device=emu.device
        )
        R, D = info["position"], info["distance"]
        P = P.expand_as(R)
        colors = torch.tensor([1.0, 0.0, 0.0]).to(emu.device)
        colors = colors.expand_as(P)
        weight = (D - D.mean()) / D.std()
        weight = weight.clamp(0, 1.0)
        colors = colors.clone()
        colors[:, 0] -= weight
        colors[:, 1] += weight
        colors = colors.cpu().numpy()
        return R.cpu().numpy(), P.cpu().numpy(), colors
        
        
class TrackBoundary(OverlayWrapper):
    def _compute(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        emu: MarioKart = self.get_wrapper_attr('emu')
    
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
        p0, p1 = np.split(boundary_lines, 2, axis=1)
        p0 = p0.reshape(-1, 3) # squeeze
        p1 = p1.reshape(-1, 3) # squeeze
    
        colors = np.array([0.0, 0.1, 0.9])[None, :].repeat(p0.shape[0], axis=0)
        return p0, p1, colors
        

def compose_overlays(env: gym.Env, *overlay_classes: OverlayWrapper) -> gym.Env:
    return reduce(lambda e, cls: cls(e), overlay_classes, env)

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
