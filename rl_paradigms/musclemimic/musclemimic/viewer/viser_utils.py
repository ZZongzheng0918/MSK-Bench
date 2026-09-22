from __future__ import annotations

import mujoco
import numpy as np
import trimesh
import trimesh.creation
import trimesh.visual


def _create_primitive_mesh(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    """Create a trimesh for a MuJoCo primitive geom.

    Supports: SPHERE, BOX, CYLINDER, CAPSULE, ELLIPSOID. Returns None if not supported.
    """
    gtype = int(mj_model.geom_type[geom_id])
    size = np.array(mj_model.geom_size[geom_id], dtype=float)

    # MuJoCo enum values are available as mujoco.mjtGeom.*
    if gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        r = float(size[0])
        return trimesh.creation.icosphere(subdivisions=2, radius=r)
    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        # MuJoCo stores half-extents
        ext = 2.0 * size[:3]
        return trimesh.creation.box(extents=ext)
    if gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        r = float(size[0])
        h = float(2.0 * size[1])
        return trimesh.creation.cylinder(radius=r, height=h, sections=24)
    if gtype == mujoco.mjtGeom.mjGEOM_CAPSULE:
        r = float(size[0])
        h = float(2.0 * size[1])
        # trimesh uses height excluding hemispheres
        return trimesh.creation.capsule(radius=r, height=h, count=[16, 8])
    if gtype == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        # size contains the radii for the principal axes
        # trimesh.ellipsoid expects radii
        radii = size[:3]
        return trimesh.creation.ellipsoid(radii=radii, subdivisions=2)

    # For PLANE and others in MVP: skip
    return None


def _create_mesh_from_model(mj_model: mujoco.MjModel, geom_id: int) -> trimesh.Trimesh | None:
    """Create a trimesh from a MuJoCo mesh geom (vertices + faces only)."""
    mesh_id = int(mj_model.geom_dataid[geom_id])
    if mesh_id < 0 or mesh_id >= mj_model.nmesh:
        return None

    vert_start = int(mj_model.mesh_vertadr[mesh_id])
    vert_count = int(mj_model.mesh_vertnum[mesh_id])
    face_start = int(mj_model.mesh_faceadr[mesh_id])
    face_count = int(mj_model.mesh_facenum[mesh_id])

    vertices = np.array(mj_model.mesh_vert[vert_start : vert_start + vert_count], dtype=float)
    faces = np.array(mj_model.mesh_face[face_start : face_start + face_count], dtype=np.int32)

    if vertices.size == 0 or faces.size == 0:
        return None

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh


def _geom_color_rgba(mj_model: mujoco.MjModel, geom_id: int) -> np.ndarray:
    """Get RGBA color for a geom: prefer geom_rgba, else default bluish visual/collision tint."""
    rgba = np.array(mj_model.geom_rgba[geom_id], dtype=float)
    if rgba[3] > 0:  # alpha set
        return rgba

    # fallback: pick a default based on collision/visual
    is_collision = (mj_model.geom_contype[geom_id] != 0) or (mj_model.geom_conaffinity[geom_id] != 0)
    if is_collision:
        return np.array([0.8, 0.4, 0.4, 0.6], dtype=float)
    return np.array([0.12, 0.5, 0.9, 1.0], dtype=float)


def build_body_meshes(mj_model: mujoco.MjModel, include_collision: bool = False) -> dict[int, trimesh.Trimesh]:
    """Merge geoms per body into a single mesh in the body frame.

    Args:
        mj_model: MuJoCo model
        include_collision: If True, include collision geoms; else only visual geoms (group 0)

    Returns:
        dict: body_id -> merged trimesh in body local frame
    """
    import viser.transforms as vtf

    body_geoms: dict[int, list[int]] = {}

    for gid in range(mj_model.ngeom):
        # Skip collision geoms unless requested
        is_collision = (mj_model.geom_contype[gid] != 0) or (mj_model.geom_conaffinity[gid] != 0)
        if include_collision != is_collision:
            continue

        # Only show skeleton (group 0) - skip muscle visualization geoms (group 3)
        geom_group = int(mj_model.geom_group[gid])
        if not include_collision and geom_group != 0:
            continue

        body_id = int(mj_model.geom_bodyid[gid])
        # skip world body 0 geoms
        if body_id == 0:
            continue
        body_geoms.setdefault(body_id, []).append(gid)

    out: dict[int, trimesh.Trimesh] = {}

    for body_id, geom_ids in body_geoms.items():
        parts: list[trimesh.Trimesh] = []
        for gid in geom_ids:
            gtype = int(mj_model.geom_type[gid])
            if gtype == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = _create_mesh_from_model(mj_model, gid)
            else:
                mesh = _create_primitive_mesh(mj_model, gid)
            if mesh is None:
                continue

            # color
            rgba = _geom_color_rgba(mj_model, gid)
            # Use per-vertex colors (works for all mesh types)
            mesh.visual = trimesh.visual.ColorVisuals(
                vertex_colors=np.tile((rgba * 255.0).astype(np.uint8), (len(mesh.vertices), 1))
            )

            # Transform mesh into body frame
            pos = np.array(mj_model.geom_pos[gid], dtype=float)
            quat = np.array(mj_model.geom_quat[gid], dtype=float)  # wxyz
            T = np.eye(4)
            T[:3, :3] = vtf.SO3(quat).as_matrix()
            T[:3, 3] = pos
            mesh.apply_transform(T)
            parts.append(mesh)

        if not parts:
            continue
        merged = parts[0] if len(parts) == 1 else trimesh.util.concatenate(parts)
        out[body_id] = merged

    return out
