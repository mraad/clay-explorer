"""Clay v1.5 chip embeddings for an 8-bit RGB GeoTIFF (wall-to-wall recipe).

Uses the claymodel package, configs/ and checkpoints/ from the model dir: the folder holding this
script when it contains configs/ (the packaged bundle), else the sibling ../model repo:
    python embed_tif.py IN.tif OUT.parquet [--chip 256] [--model-dir DIR]
"""

import argparse
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.windows import Window, bounds
from pyproj import Geod, Transformer
from shapely import box as shp_box

PLATFORM = "linz"  # closest registered sensor: 8-bit RGB aerial, ~0.5 m


def chip_starts(length, size):
    # Tile with stride size; last chip snaps to the edge (overlaps) instead of padding.
    if size <= 0 or length < size:
        raise ValueError("chip size must be positive and fit inside both raster dimensions")
    starts = list(range(0, length - size + 1, size))
    if starts[-1] != length - size:
        starts.append(length - size)
    return starts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif")
    ap.add_argument("out")
    here = Path(__file__).resolve().parent
    ap.add_argument("--model-dir", type=Path, default=here if (here / "configs").is_dir() else here / "../model")
    ap.add_argument("--ckpt", help="default: MODEL_DIR/checkpoints/clay-v1.5.ckpt")
    ap.add_argument("--chip", type=int, default=256, help="chip size in pixels")
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    size = args.chip
    if size <= 0 or size % 8:
        ap.error("--chip must be a positive multiple of the 8 px patch size")
    if args.batch <= 0:
        ap.error("--batch must be positive")

    import torch
    import yaml
    from box import Box
    from torchvision.transforms import v2

    from claymodel.finetune.embedder.factory import Embedder

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    ckpt = args.ckpt or args.model_dir / "checkpoints/clay-v1.5.ckpt"
    with (args.model_dir / "configs/metadata.yaml").open() as metadata:
        meta = Box(yaml.safe_load(metadata))[PLATFORM]
    bands = meta.band_order  # red, green, blue == tif bands 1..3
    transform = v2.Normalize(
        mean=[meta.bands.mean[b] for b in bands], std=[meta.bands.std[b] for b in bands]
    )
    waves = torch.tensor([meta.bands.wavelength[b] for b in bands], device=device)

    with rasterio.open(args.tif) as src:
        rgb_interp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
        if src.dtypes[:3] != ("uint8",) * 3 or src.colorinterp[:3] != rgb_interp:
            raise ValueError("bands 1–3 must be uint8 and tagged red, green, blue; extra bands are ignored")
        if src.crs is None:
            raise ValueError("input raster must have a CRS")
        ys, xs = chip_starts(src.height, size), chip_starts(src.width, size)
        rgb = src.read([1, 2, 3])
        # RGB masks only: an all-nodata 4th band would zero out dataset_mask().
        valid = np.all([src.read_masks(i) > 0 for i in (1, 2, 3)], axis=0)
        crs, aff = src.crs, src.transform
        H, W = src.height, src.width
        # Ground distance of one pixel step at the centre: correct for any CRS.
        to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lon0, lat0 = to_wgs.transform(*(aff * (W / 2, H / 2)))
        lon1, lat1 = to_wgs.transform(*(aff * (W / 2 + 1, H / 2)))
        gsd = Geod(ellps="WGS84").inv(lon0, lat0, lon1, lat1)[2]

    print(f"{args.tif}: {W}x{H}, crs={crs}, ground gsd={gsd:.3f} m, device={device}")

    rows, chips = [], []
    for y in ys:
        for x in xs:
            chips.append(rgb[:, y : y + size, x : x + size])
            rows.append(
                {
                    "row": y,
                    "col": x,
                    "chip_px": size,
                    "valid_frac": float(valid[y : y + size, x : x + size].mean()),
                    "geometry": shp_box(*bounds(Window(x, y, size, size), aff)),
                }
            )

    embedder = Embedder(img_size=size, ckpt_path=ckpt, device=device)

    t0 = time.perf_counter()
    out = []
    with torch.no_grad():
        for i in range(0, len(chips), args.batch):
            # Per batch: float32 copies of all chips at once cost GBs at 256 px.
            batch = transform(torch.from_numpy(np.stack(chips[i : i + args.batch]).astype(np.float32))).to(device)
            n = len(batch)
            # Acquisition date unknown: zero time and latlon together, as training
            # does for 20% of chips (EODataset), so the input stays in-distribution.
            datacube = {
                "pixels": batch,
                "time": torch.zeros(n, 4, device=device),
                "latlon": torch.zeros(n, 4, device=device),
                "gsd": torch.tensor(gsd, device=device),
                "waves": waves,
            }
            out.append(embedder(datacube).float().cpu().numpy())
    emb = np.concatenate(out)
    elapsed = time.perf_counter() - t0
    if emb.shape != (len(rows), 1024) or not np.isfinite(emb).all():
        raise ValueError(f"expected finite embeddings with shape {(len(rows), 1024)}, got {emb.shape}")

    gdf = gpd.GeoDataFrame(rows, crs=crs)
    gdf["embedding"] = list(emb)
    gdf.to_parquet(args.out)
    print(f"{len(gdf)} chips -> {args.out} ({elapsed:.2f}s on {device})")


if __name__ == "__main__":
    main()
