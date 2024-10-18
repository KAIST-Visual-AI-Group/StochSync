import os
import torch
import numpy as np
import pymeshlab
import xatlas
from .matrix_utils import rodrigues

def read_obj(file_path):
    vertices = []
    faces = []
    tex_coords = []
    colors = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.split("#")[0].strip()
            parts = line.split()
            if not parts:
                continue
            prefix = parts[0]
            values = parts[1:]

            if prefix == "v":  # Vertex position and optional color
                vertex = list(
                    map(float, values[:3])
                )  # First 3 values are the vertex position
                vertices.append(vertex)
                if len(values) >= 6:
                    color = list(map(float, values[3:6]))  # Optional color
                    colors.append(color)

            elif prefix == "vt":  # Texture coordinates
                tex_coord = list(map(float, values))
                tex_coords.append(tex_coord)

            elif prefix == "f":  # Face indices
                face = [[v.strip() for v in p.split("/")] for p in values]
                face = [list(filter(len, p))[:2] for p in face]
                assert all(
                    len(set(p)) == 1 for p in face
                ), f"Only supports faces with same indices for all vertex properties, got: {face}"
                faces.append([p[0] for p in face])

    # Convert lists to numpy arrays
    vertices = np.array(vertices, dtype=float)
    colors = np.array(colors, dtype=float) if colors else None
    tex_coords = np.array(tex_coords, dtype=float) if tex_coords else None
    faces = np.array(faces, dtype=int)  # Faces have variable-length data

    # Store in dictionary
    obj_data = {
        "v": vertices,
        "f": faces,
    }
    if tex_coords is not None:
        obj_data["vt"] = tex_coords
    if colors is not None:
        obj_data["c"] = colors

    return obj_data


def write_obj(file_path, v, f, vt=None, vn=None, c=None):
    with open(file_path, "w") as file:
        # Write vertices and optional colors
        for i, vertex in enumerate(v):
            vertex_str = " ".join(map(str, vertex))
            if c is not None:
                color_str = " ".join(map(str, c[i]))
                file.write(f"v {vertex_str} {color_str}\n")
            else:
                file.write(f"v {vertex_str}\n")

        # Write texture coordinates
        if vt is not None:
            for tex_coord in vt:
                tex_coord_str = " ".join(map(str, tex_coord))
                file.write(f"vt {tex_coord_str}\n")

        # Write normals
        if vn is not None:
            for normal in vn:
                normal_str = " ".join(map(str, normal))
                file.write(f"vn {normal_str}\n")

        # Write faces (F, 3)
        for face in f:
            if vt is not None and vn is not None:
                face_str = " ".join([f"{v}/{v}/{v}" for v in face])
            elif vt is None and vn is not None:
                face_str = " ".join([f"{v}//{v}" for v in face])
            elif vt is not None and vn is None:
                face_str = " ".join([f"{v}/{v}" for v in face])
            else:
                face_str = " ".join([f"{v}" for v in face])
            file.write(f"f {face_str}\n")

def convert_to_obj(src, dest):
    flag = False
    ms = pymeshlab.MeshSet()
    if src.endswith(".glb") or src.endswith(".gltf"):
        ms.load_new_mesh(src, load_in_a_single_layer=False)
        if len(ms) > 1:
            flag = True
            del ms
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(src, load_in_a_single_layer=True)
    else:
        ms.load_new_mesh(src)
    
    try:
        ms.compute_texcoord_transfer_wedge_to_vertex()
    except:
        pass
    ms.save_current_mesh(dest, save_textures=False, save_wedge_texcoord=False)
    return flag

def extract_texture(src, dest):
    ms = pymeshlab.MeshSet()
    assert src.endswith(".glb") or src.endswith(".gltf")
    ms.load_new_mesh(src, load_in_a_single_layer=False)
    assert len(ms) == 1
    assert ms.current_mesh().texture_number() == 1
    ms.current_mesh().texture(0).save(dest)

# simplify using pymeshlab
def simplify_mesh(v, f, target_face_num=30000):
    m = pymeshlab.Mesh(v, f - 1)  # pymeshlab uses 0-based indexing

    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "mesh")

    # simplify mesh
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()  # verts not refed by any faces
    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0
    ms.meshing_remove_connected_component_by_diameter()
    
    face_num = len(ms.current_mesh().face_matrix())
    if face_num > target_face_num:
        ms.meshing_decimation_quadric_edge_collapse(
            targetfacenum=target_face_num, autoclean=True
        )

    m = ms.current_mesh()
    return m.vertex_matrix(), m.face_matrix() + 1

def unwrap_mesh(v, f):
    """
    Unwraps a 3D mesh to generate UV coordinates.

    Parameters:
    -----------
    v : np.ndarray
        Vertex positions (N, 3).
    f : np.ndarray
        Face indices (F, 3).

    Returns:
    --------
    remapped_vertices : np.ndarray
        Remapped vertex positions (M, 3).
    indices : np.ndarray
        New face indices (F, 3).
    uvs : np.ndarray
        UV coordinates (M, 2).
    """
    vmapping, indices, uvs = xatlas.parametrize(v, f - 1)
    return v[vmapping], (indices + 1), uvs

def rotate_mesh(v, angle, axis):
    """
    Rotates a 3D mesh around an axis.

    Parameters:
    -----------
    v : np.ndarray
        Vertex positions (N, 3).
    angle : float
        Rotation angle in degrees.
    axis : str
        Rotation axis, one of "x", "y", or "z".

    Returns:
    --------
    rotated_v : np.ndarray
        Rotated vertex positions (N, 3).
    """
    if axis == "x":
        R = rodrigues(torch.tensor([1.0, 0.0, 0.0]), angle * np.pi / 180).numpy()
    elif axis == "y":
        R = rodrigues(torch.tensor([0.0, 1.0, 0.0]), angle * np.pi / 180).numpy()
    elif axis == "z":
        R = rodrigues(torch.tensor([0.0, 0.0, 1.0]), angle * np.pi / 180).numpy()
        
    return v @ R.T