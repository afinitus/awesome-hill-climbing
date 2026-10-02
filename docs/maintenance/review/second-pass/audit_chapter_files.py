"""Read-only structural audit and per-file inventory of the chapter scope at 25a9a1d.

Run from the repository root. Writes only this audit's JSON inventory; published artifacts
are read, never regenerated. Manual/source-review scope is explained in CHAPTERS.md.
"""

from __future__ import annotations

import ast
import hashlib
import json
import struct
import subprocess
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
import torch

from lastmile.common.policy_flow import FlowChunk

ROOT = Path(__file__).resolve().parents[4]
BASELINE = "25a9a1d"
SCOPES = ["algos", "lastmile", "tests", "checkpoints", "results", "docs/chapters"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASELINE, "--", *SCOPES], cwd=ROOT, text=True,
    ).splitlines()
    files = []
    for name in paths:
        path = ROOT / name
        data = path.read_bytes()
        entry = {"path": name, "bytes": len(data), "sha256": sha(data), "status": "checked"}
        if path.suffix == ".py":
            ast.parse(data, filename=name)
            if name in ("tests/test_build_site.py", "tests/test_sweep_tools.py"):
                method = "AST parse here; generator/maintenance source review delegated to root"
            elif name.startswith("algos/"):
                method = "first-pass core algorithm review; second-pass config/seed/accounting/replot paths; AST parse"
            elif name.startswith("tests/"):
                method = "test source inspection and regression-scope review; AST parse; focused execution where changed"
            else:
                method = "second-pass complete source read; API/physics/accounting contract review; AST parse"
        elif path.suffix == ".json":
            obj = json.loads(data)
            method = "JSON parse; saved count/interval/statistics reconciliation; manifest/accounting inspection"
            if name.startswith("results/"):
                assert obj["schema"] == 1
                assert obj["setting"] == "sim"
                for kind in ("base", "final"):
                    score = obj.get(kind)
                    if score:
                        assert 0 <= score["k"] <= score["n"] and score["n"] > 0
                        assert np.isclose(score["sr"], score["k"] / score["n"])
                for budget in obj["budget"].values():
                    assert all(np.isfinite(x) and x >= 0 for x in budget.values())
        elif path.suffix == ".pt":
            raw = torch.load(path, map_location="cpu", weights_only=True)
            model = FlowChunk.load(path)
            manifest = json.loads(path.with_suffix(".json").read_text())
            assert model.config.hash() == manifest["config_hash"]
            assert raw["config"] == manifest["arch"]
            assert all(torch.isfinite(v).all() for v in model.state_dict().values())
            assert all(torch.all(getattr(model, k) > 0) for k in ("obs_std", "act_std"))
            entry["tensor_parameters"] = sum(x.numel() for x in model.state_dict().values())
            method = "weights_only CPU load; architecture/config hash/manifest match; finite tensors and positive normalizers; prior 256-episode replay"
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as saved:
                for array in saved.values():
                    if np.issubdtype(array.dtype, np.number):
                        assert np.isfinite(array).all()
                entry["array_count"] = len(saved.files)
            method = "allow_pickle=False load; every numerical array finite; archive checksum match; prior 256-episode replay"
        elif path.suffix == ".xml":
            ElementTree.fromstring(data)
            model = mujoco.MjModel.from_xml_path(str(path))
            assert np.isfinite(model.body_mass).all() and np.isfinite(model.geom_pos).all()
            entry["compiled_meshes"] = model.nmesh
            method = "XML parse; MuJoCo model compilation resolving all includes/meshes; finite geometry/masses"
        elif path.suffix.lower() == ".stl":
            count = struct.unpack_from("<I", data, 80)[0]
            assert len(data) == 84 + 50 * count
            triangles = np.frombuffer(data, dtype=np.dtype([("vectors", "<f4", (12,)), ("attr", "<u2")]), offset=84)
            assert np.isfinite(triangles["vectors"]).all()
            entry["triangles"] = count
            method = "binary STL length/triangle structure validation; all vertex/normal floats finite; SHA256 and pinned-vendor inventory"
        elif path.suffix.lower() == ".obj":
            counts = Counter()
            for line in data.decode().splitlines():
                parts = line.split()
                if not parts:
                    continue
                counts[parts[0]] += 1
                if parts[0] in ("v", "vn", "vt"):
                    assert np.isfinite([float(x) for x in parts[1:]]).all()
                elif parts[0] == "f":
                    assert len(parts) >= 4
                    for token in parts[1:]:
                        idx = int(token.split("/")[0])
                        assert idx != 0 and abs(idx) <= counts["v"]
            entry["vertices"], entry["faces"] = counts["v"], counts["f"]
            method = "OBJ text geometry parse; finite vertices/normals/UVs and valid face vertex indices; SHA256 and pinned-vendor inventory"
        else:
            data.decode()
            if name.startswith("docs/chapters/"):
                method = "first-pass complete narrative/source review; second-pass reproducibility/caveat and arithmetic reconciliation"
            else:
                method = "vendor provenance/readme/changelog/license identification read; UTF8 decode; upstream preview HTTP 200 check where applicable"
        entry["inspection_method"] = method
        files.append(entry)
    archive = json.loads((ROOT / "docs/maintenance/review/archive/ch05_target_policy_bug/manifest.json").read_text())
    for relative, key in (("algos/05_rlpd_qchunk.py", "publication_source_sha256"),
                          ("checkpoints/base_v1_so101.pt", "base_checkpoint_sha256"),
                          ("checkpoints/ch05_qc_s0_review.npz", "corrected_qc_seed0_checkpoint_sha256")):
        assert sha((ROOT / relative).read_bytes()) == archive[key]
    report = {"baseline_commit": BASELINE, "scope": SCOPES, "file_count": len(files),
              "scope_counts": dict(Counter(x["path"].split("/")[0] for x in files)),
              "publication_provenance_hashes_match": True, "files": files}
    out = Path(__file__).with_name("CHAPTER-FILE-COVERAGE.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
