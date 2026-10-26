import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import ColorInterp
from shapely.geometry import box

import embed_tif
import export_data


class PipelineTests(unittest.TestCase):
    def test_chip_edges_and_invalid_sizes(self):
        self.assertEqual(embed_tif.chip_starts(64, 64), [0])
        self.assertEqual(embed_tif.chip_starts(128, 64), [0, 64])
        self.assertEqual(embed_tif.chip_starts(130, 64), [0, 64, 66])
        for length, size in [(63, 64), (64, 0), (64, -8)]:
            with self.subTest(length=length, size=size), self.assertRaises(ValueError):
                embed_tif.chip_starts(length, size)

    def test_export_import_does_not_load_inference_libraries(self):
        subprocess.run(
            [sys.executable, '-c', 'import sys, export_data; assert "torch" not in sys.modules; assert "claymodel" not in sys.modules'],
            check=True,
        )

    def test_invalid_cli_sizes_fail_before_inference(self):
        for option, value in [("--chip", "0"), ("--chip", "-8"), ("--chip", "7"), ("--batch", "0")]:
            with self.subTest(option=option, value=value):
                result = subprocess.run(
                    [sys.executable, "embed_tif.py", "missing.tif", "unused.parquet", option, value],
                    capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(option, result.stderr)
                self.assertNotIn("Traceback", result.stderr)

    def test_duplicate_embeddings_have_valid_dbscan_presets(self):
        emb = np.repeat(np.eye(2, 12, dtype=np.float32), 20, axis=0)
        presets, labels = export_data.dbscan_presets(emb)
        self.assertEqual(len(presets), 5)
        for preset, lab in zip(presets, labels):
            self.assertEqual(preset["clusters"], 2)
            self.assertEqual(preset["noise"], 0)
            self.assertTrue((lab >= 0).all())

    def test_export_validates_before_replacing_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "view"
            out.mkdir()
            assets = ["meta.json", "chips.geojson", "embeddings.f32", "image.jpg"]
            for asset in assets:
                (out / asset).write_bytes(b"original")
            emb = np.random.default_rng(0).normal(size=(12, 12)).astype(np.float32)
            for invalid in [emb[:9], np.full_like(emb, np.nan), np.ones_like(emb), emb]:
                with self.subTest(count=len(invalid), finite=np.isfinite(invalid).all()):
                    gdf = self.frame(invalid)
                    with patch.object(gpd, "read_parquet", return_value=gdf), patch.object(
                        sys, "argv", ["export_data.py", str(root / "missing.tif"), "unused.parquet", "--out", str(out)]
                    ), self.assertRaises((ValueError, rasterio.errors.RasterioIOError)):
                        export_data.main()
                    for asset in assets:
                        self.assertEqual((out / asset).read_bytes(), b"original")

    def test_rank_one_export_produces_finite_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tif, out = root / "rgb.tif", root / "view"
            with rasterio.open(
                tif, "w", driver="GTiff", width=16, height=40, count=3,
                dtype="uint8", crs="EPSG:3857", transform=from_origin(5000, 5000, 1, 1),
            ) as src:
                src.write(np.full((3, 40, 16), 100, dtype=np.uint8))
            emb = np.tile([1, -1], 10).astype(np.float32)[:, None] * np.arange(1, 13, dtype=np.float32)
            with patch.object(gpd, "read_parquet", return_value=self.frame(emb)), patch.object(
                sys, "argv", ["export_data.py", str(tif), "unused.parquet", "--out", str(out)]
            ), contextlib.redirect_stdout(io.StringIO()):
                export_data.main()
            meta = json.loads((out / "meta.json").read_text())
            vectors = np.fromfile(out / "embeddings.f32", dtype="<f4").reshape(meta["count"], meta["dim"])
            self.assertTrue(np.isfinite(vectors).all())
            np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-6)
            self.assertTrue(np.isfinite(meta["pca_variance"]).all())
            features = json.loads((out / "chips.geojson").read_text())["features"]
            self.assertEqual([f["properties"]["id"] for f in features], list(range(len(emb))))
            self.assertTrue(all(len(f["properties"]["pca"]) == 7 for f in features))
            for asset in out.iterdir():
                asset.write_bytes(b"original")
            with patch.object(gpd, "read_parquet", return_value=self.frame(emb).drop(columns="chip_px")), patch.object(
                sys, "argv", ["export_data.py", str(tif), "unused.parquet", "--out", str(out)]
            ), self.assertRaises(AttributeError):
                export_data.main()
            self.assertTrue(all(asset.read_bytes() == b"original" for asset in out.iterdir()))

    def test_unsupported_rasters_fail_before_creating_a_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frame = self.frame(np.random.default_rng(0).normal(size=(12, 12)))
            for case in ["rotated", "no-crs", "not-rgb"]:
                with self.subTest(case=case):
                    tif, out = root / f"{case}.tif", root / case
                    transform = from_origin(5000, 5000, 1, 1)
                    if case == "rotated":
                        transform = transform @ rasterio.Affine.rotation(10)
                    with rasterio.open(
                        tif, "w", driver="GTiff", width=16, height=40, count=3, dtype="uint8",
                        crs=None if case == "no-crs" else "EPSG:3857", transform=transform,
                    ) as src:
                        src.write(np.full((3, 40, 16), 100, dtype=np.uint8))
                        if case == "not-rgb":
                            src.colorinterp = (ColorInterp.undefined,) * 3
                    with patch.object(gpd, "read_parquet", return_value=frame), patch.object(
                        sys, "argv", ["export_data.py", str(tif), "unused.parquet", "--out", str(out)]
                    ), self.assertRaises(ValueError):
                        export_data.main()
                    self.assertFalse(out.exists())

    @staticmethod
    def frame(emb):
        return gpd.GeoDataFrame(
            {"row": np.arange(len(emb)), "col": 0, "chip_px": 8, "embedding": list(emb)},
            geometry=[box(5000 + i, 5000, 5001 + i, 5001) for i in range(len(emb))], crs="EPSG:3857",
        )


if __name__ == "__main__":
    unittest.main()
