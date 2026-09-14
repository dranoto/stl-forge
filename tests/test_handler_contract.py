import ast
import json
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
HANDLER_PATH = REPO_ROOT / "handler.py"


def load_handler_functions(*names):
    """Load selected handler functions without importing GPU-only dependencies."""
    source = HANDLER_PATH.read_text(encoding="utf-8")
    module = ast.parse(source, filename=str(HANDLER_PATH))
    selected: list[ast.stmt] = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    class NumpyStub:
        @staticmethod
        def asarray(value):
            return value

    namespace = {
        "os": os,
        "io": __import__("io"),
        "sys": sys,
        "base64": __import__("base64"),
        "tempfile": __import__("tempfile"),
        "time": __import__("time"),
        "traceback": __import__("traceback"),
        "requests": types.SimpleNamespace(),
        "np": NumpyStub,
        "trimesh": types.SimpleNamespace(),
        "Image": __import__("PIL.Image", fromlist=["Image"]),
        "DEFAULT_TARGET_FACES": 100_000,
        "DEFAULT_MC_RESOLUTION": 256,
        "FORCE_INLINE_KEY": "force_inline",
        "STL_B64_INLINE_MAX_BYTES": 7 * 1024 * 1024,
        "HF_MODEL": "test/model",
        "_load_time": 0.0,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(HANDLER_PATH), "exec"), namespace)
    return namespace


class FakeMeshResult:
    vertices = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    faces = [[0, 1, 2]]


class RecordingPipeline:
    def __init__(self):
        self.kwargs = None

    def __call__(self, *, image, **kwargs):
        self.kwargs = kwargs
        return [FakeMeshResult()]


class HandlerContractTests(unittest.TestCase):
    def test_requested_octree_resolution_is_passed_to_pipeline(self):
        namespace = load_handler_functions("handler")
        handler = namespace["handler"]
        pipeline = RecordingPipeline()
        class ExportableMesh:
            vertices = FakeMeshResult.vertices
            faces = FakeMeshResult.faces
            is_watertight = True
            extents = types.SimpleNamespace(tolist=lambda: [1.0, 1.0, 0.0])

            def export(self, file_obj, file_type):
                self.exported_as = file_type
                file_obj.write(b"solid test")

        namespace["trimesh"].Trimesh = lambda vertices, faces: ExportableMesh()
        namespace.update(
            load_image=lambda value: namespace["Image"].new("RGB", (2, 2), "white"),
            flatten_alpha=lambda image: image,
            load_pipeline=lambda: pipeline,
            make_printable=lambda mesh, target_faces: mesh,
            upload_to_r2=lambda data, job_id: None,
        )

        cleanup_module = types.ModuleType("runpod.serverless.utils.rp_cleanup")
        setattr(cleanup_module, "clean", lambda: None)

        with patch.dict(sys.modules, {"runpod.serverless.utils.rp_cleanup": cleanup_module}):
            output = handler(
                {
                    "id": "test-job",
                    "input": {
                        "image": "unused",
                        "mc_resolution": 96,
                        "num_inference_steps": 1,
                        "target_faces": 50_000,
                    },
                }
            )

        self.assertNotIn("error", output)
        self.assertEqual(
            pipeline.kwargs,
            {"num_inference_steps": 1, "octree_resolution": 96},
        )

    def test_hub_json_exposes_all_optional_storage_fields(self):
        hub = json.loads((REPO_ROOT / ".runpod" / "hub.json").read_text(encoding="utf-8"))
        env = {item["key"]: item for item in hub["config"]["env"]}
        expected = {
            "BUCKET_ENDPOINT_URL",
            "BUCKET_ACCESS_KEY_ID",
            "BUCKET_SECRET_ACCESS_KEY",
            "BUCKET_NAME",
        }
        self.assertTrue(expected.issubset(env))
        for key in expected:
            field = env[key]["input"]
            self.assertEqual(field["type"], "string")
            self.assertFalse(field["required"])
            self.assertTrue(field["advanced"])
            self.assertEqual(field["default"], "")

    def test_hub_smoke_test_requests_reduced_resolution(self):
        tests = json.loads((REPO_ROOT / ".runpod" / "tests.json").read_text(encoding="utf-8"))
        smoke_input = tests["tests"][0]["input"]
        self.assertEqual(smoke_input["mc_resolution"], 96)
        self.assertEqual(smoke_input["num_inference_steps"], 1)


if __name__ == "__main__":
    unittest.main()
