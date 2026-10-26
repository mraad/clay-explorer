"""Turn embed_tif.py output into the static files the web app reads.

    ../model/.venv/bin/python export_data.py IN.tif EMBEDDINGS.parquet [--out data]

Writes data/{chips.geojson, embeddings.f32, image.jpg, meta.json}: PCA colours,
k-means for k = 2..8 and DBSCAN at several eps presets.
"""

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from PIL import Image
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from embed_tif import PLATFORM

MAX_CLUSTERS = 8  # 8-slot categorical palette in app.js; smaller DBSCAN clusters fold into the last one
K_RANGE = range(2, MAX_CLUSTERS + 1)
# DBSCAN: density is flat in 1024-d (one blob or all noise), so cluster in a
# 10-d PCA projection. eps presets are percentiles of each chip's cosine
# distance to its MIN_SAMPLES-th neighbour (the k-distance curve).
DB_PCA_DIMS = 10
DB_MIN_SAMPLES = 10
DB_QUANTILES = (10, 15, 20, 25, 30)


def dbscan_presets(emb):
    x = PCA(n_components=DB_PCA_DIMS, random_state=0).fit_transform(emb)
    normalize(x, copy=False)
    nn = NearestNeighbors(n_neighbors=DB_MIN_SAMPLES, metric="cosine").fit(x)
    kdist = nn.kneighbors(x)[0][:, -1]
    presets, labels = [], []
    for q in DB_QUANTILES:
        eps = max(float(np.percentile(kdist, q)), np.finfo(float).eps)
        lab = DBSCAN(eps=eps, min_samples=DB_MIN_SAMPLES, metric="cosine").fit_predict(x)
        # Relabel by size (largest = 0) so colours follow importance; -1 stays noise.
        sizes = np.bincount(lab[lab >= 0])
        rank = np.argsort(np.argsort(-sizes, kind="stable"))
        lab = np.where(lab >= 0, np.minimum(rank[lab.clip(0)], MAX_CLUSTERS - 1), -1) if sizes.size else lab
        labels.append(lab)
        presets.append(
            {
                "percentile": q,
                "eps": round(eps, 4),
                "clusters": int(lab.max() + 1),
                "noise": round(float((lab < 0).mean()), 4),
                "other": bool(sizes.size > MAX_CLUSTERS),
            }
        )
    return presets, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tif", type=Path)
    ap.add_argument("parquet", type=Path)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "data")
    ap.add_argument("--title", help="view title (default: the tif name)")
    ap.add_argument("--model", default="Clay v1.5", help="encoder name shown in the app, e.g. for a fine-tuned --ckpt")
    args = ap.parse_args()
    gdf = gpd.read_parquet(args.parquet)
    if len(gdf) < max(DB_PCA_DIMS, DB_MIN_SAMPLES):
        raise ValueError("export requires at least 10 chips for PCA and clustering; use a smaller --chip")
    emb = np.stack(gdf.embedding).astype(np.float32)
    if emb.ndim != 2 or emb.shape[1] < DB_PCA_DIMS or not np.isfinite(emb).all():
        raise ValueError("embeddings must be finite vectors with at least 10 dimensions")
    # Centre first: all chips share a large common component that pins raw cosine near 1.
    emb -= emb.mean(axis=0)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("scene-centred embeddings must have finite, nonzero lengths for cosine similarity")
    emb /= norms  # cosine sim == dot product

    # PCA -> false colour: robust-scale each of the first 3 components to [0, 1].
    pca = PCA(n_components=3, random_state=0)
    pcs = pca.fit_transform(emb)
    lo, hi = np.percentile(pcs, [2, 98], axis=0)
    rgb = np.clip(np.divide(pcs - lo, hi - lo, out=np.zeros_like(pcs), where=hi > lo), 0, 1)
    rgb = (rgb * 255).round().astype(int)

    props = gdf[["row", "col"]].copy()
    props["id"] = np.arange(len(gdf))
    props["pca"] = ["#{:02x}{:02x}{:02x}".format(*c) for c in rgb]
    for k in K_RANGE:
        props[f"k{k}"] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(emb)
    db_presets, db_labels = dbscan_presets(emb)
    for i, labels in enumerate(db_labels):
        props[f"db{i}"] = labels

    with rasterio.open(args.tif) as src:
        if src.dtypes[:3] != ("uint8",) * 3 or src.colorinterp[:3] != (ColorInterp.red, ColorInterp.green, ColorInterp.blue):
            raise ValueError("bands 1–3 must be uint8 and tagged red, green, blue")
        if src.transform.b or src.transform.d or src.transform.a <= 0 or src.transform.e >= 0:
            raise ValueError("export requires a north-up raster; reproject rotated imagery first")
        epsg = src.crs.to_epsg() if src.crs else None
        if epsg is None:
            raise ValueError("export requires a raster CRS with an EPSG code")
        img = np.moveaxis(src.read([1, 2, 3]), 0, -1)
        b = src.bounds

    meta = {
        "title": args.title or args.tif.stem,
        "source": args.tif.name,
        "model": f"{args.model} ({PLATFORM} sensor profile, class token)",
        "count": len(gdf),
        "dim": emb.shape[1],
        "chip_px": int(gdf.chip_px.iloc[0]),
        "image": {"extent": [b.left, b.bottom, b.right, b.top], "wkid": epsg},
        "pca_variance": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "k_range": [K_RANGE.start, K_RANGE.stop - 1],
        "dbscan": {"pca_dims": DB_PCA_DIMS, "min_samples": DB_MIN_SAMPLES, "presets": db_presets},
    }
    text = json.dumps(meta, indent=2, allow_nan=False)
    args.out.mkdir(parents=True, exist_ok=True)
    gpd.GeoDataFrame(props, geometry=gdf.geometry, crs=gdf.crs).to_crs(4326).to_file(
        args.out / "chips.geojson", driver="GeoJSON"
    )
    emb.astype("<f4").tofile(args.out / "embeddings.f32")
    Image.fromarray(img).save(args.out / "image.jpg", quality=88)
    (args.out / "meta.json").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
