import io
import json
import zipfile

import pytest

from app.library import Library, clean_label
from tests.synth import j_path, sign_sequence, static_take


def test_clean_label():
    assert clean_label("  obrigado ") == "OBRIGADO"
    assert clean_label("bom   dia!") == "BOM DIA"
    assert clean_label("ç") == "Ç"
    assert clean_label("../../x") == "X"
    with pytest.raises(ValueError):
        clean_label("  ")


def test_add_reload_delete(tmp_path):
    lib = Library(tmp_path)
    ids = lib.add_samples("a", "static", [static_take("A") for _ in range(3)])
    assert len(ids) == 3 and lib.kind_of("A") == "static"
    lib.add_samples("J", "dynamic", [sign_sequence("I", hold=0, path=j_path, path_time=0.7)])
    assert {e.label for e in lib.classifier.examples} == {"A", "J"}

    again = Library(tmp_path)                      # reload from disk
    assert again.summary()[0]["label"] == "A" and again.summary()[0]["count"] == 3
    assert len(again.classifier.examples) == 4
    assert again.sample_json(ids[0])["frames"]

    assert again.delete_sample(ids[0])
    assert again.summary()[0]["count"] == 2
    assert again.delete_sign("J") and again.kind_of("J") is None
    assert not (tmp_path / "samples" / f"{ids[0]}.npz").exists()


def test_kind_conflict(tmp_path):
    lib = Library(tmp_path)
    lib.add_samples("A", "static", [static_take("A")])
    with pytest.raises(ValueError):
        lib.add_samples("A", "dynamic", [static_take("A")])


def test_export_import_roundtrip(tmp_path):
    src = Library(tmp_path / "home")
    src.add_samples("A", "static", [static_take("A") for _ in range(2)])
    src.add_samples("B", "static", [static_take("B") for _ in range(2)])
    data = src.export_zip()

    dst = Library(tmp_path / "demo")
    dst.add_samples("A", "static", [static_take("A")])
    res = dst.import_zip(data)
    assert res == {"added": 4, "skipped": 0, "conflicts": []}
    assert {s["label"]: s["count"] for s in dst.summary()} == {"A": 3, "B": 2}
    assert dst.import_zip(data)["added"] == 0      # importing twice adds nothing


def test_import_rejects_bad_files(tmp_path):
    lib = Library(tmp_path)
    with pytest.raises(ValueError):
        lib.import_zip(b"not a zip")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("signs.json", json.dumps({"signs": {"X": {"kind": "static", "samples": ["../../evil"]}}}))
        z.writestr("samples/../../evil.npz", b"x")
    assert lib.import_zip(buf.getvalue())["added"] == 0
    assert not (tmp_path.parent / "evil.npz").exists()
