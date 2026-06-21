"""
STL Forge — pipeline helpers (mesh cleanup + printability).

Kept separate from handler.py so it can be tested in isolation and reused by
the future Gradio/Streamlit client without dragging in runpod.
"""

import io
import numpy as np
import trimesh


def make_printable(mesh: trimesh.Trimesh, target_faces: int = 100_000) -> trimesh.Trimesh:
    """Clean, repair, decimate, centre — the printability pipeline.

    See [[STL Forge - Project Plan]] §3 for rationale.
    Returns a trimesh.Trimesh that's slicer-friendly.
    """
    # 1. basic cleanup
    mesh.process(validate=True)
    mesh.merge_vertices()
    mesh.remove_duplicate_faces()
    mesh.remove_degenerate_faces()
    mesh.remove_unreferenced_vertices()

    # 2. fix holes (best-effort)
    try:
        trimesh.repair.fill_holes(mesh)
    except Exception:
        pass

    # 3. fix winding + normals + inversion
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)

    # 4. keep the largest connected component
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        mesh = max(components, key=lambda m: len(m.faces))

    # 5. decimate
    if len(mesh.faces) > target_faces:
        try:
            mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
        except Exception:
            pass

    # 6. centre on origin
    mesh.rezero()
    return mesh


def export_stl_bytes(mesh: trimesh.Trimesh) -> bytes:
    """Binary STL export, little-endian."""
    buf = io.BytesIO()
    mesh.export(buf, file_type="stl")
    return buf.getvalue()


def printability_report(mesh: trimesh.Trimesh) -> dict:
    """Slicer-relevant facts about a mesh. Cheap to compute, useful in responses."""
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "watertight": bool(mesh.is_watertight),
        "is_volume": bool(mesh.is_volume),
        "bbox_extents": mesh.extents.tolist() if mesh.extents is not None else None,
        "euler_number": int(mesh.euler_number) if mesh.euler_number is not None else None,
    }
