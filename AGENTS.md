# Repository Guidelines

## Project Structure & Module Organization

- `index.html`, `app.js`, and `style.css` implement a VanillaJS map app using ArcGIS Maps SDK 5.1 and Calcite from CDN; there is no frontend build step or API key requirement.
- `embed_tif.py` generates GeoParquet embeddings; `export_data.py` produces browser assets.
- `data/` holds the default export; `views/<name>/` hold alternate exports. Both are generated and gitignored.
- Local-only files are gitignored. When `private/README.md` exists, it documents them; read it before work outside the public app and pipeline.
- `tests/` contains Python regression checks and Node frontend checks using a stubbed ArcGIS SDK.
- `README.md` documents pipeline/model choices; `tasks/lessons.md` records corrections. `CLAUDE.md` imports this shared guide.

## Build, Test, and Development Commands

Run from the repository root:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
../model/.venv/bin/python embed_tif.py area.tif area.parquet --chip 64
../model/.venv/bin/python export_data.py area.tif area.parquet --out views/example
```

These serve, embed, and export. Alternate views use `/?data=views/example`; preserve relative-path validation. Python uses the read-only sibling `../model` fork (`fix/errors`); the scripts use local model files when `configs/` sits next to them. `--ckpt` accepts fine-tuned encoders; missing encoder weights raise `ValueError`.

## Coding Style & Naming Conventions

Match existing formatting: four-space Python indentation; two-space JavaScript, HTML, and CSS indentation. Use Python `snake_case`, JavaScript `camelCase`, and uppercase constants. No formatter or linter is configured.

## Architecture & Data Contracts

- `--chip` becomes parquet `chip_px`, read by the exporter. Inference-only imports stay inside `embed_tif.main()` so exporting does not load torch or claymodel. Extra raster bands are ignored; only uint8 RGB bands 1–3 are used.
- Change exporter and frontend together: `meta.json` supplies count/dimension, image extent/WKID, PCA variance, `k_range`, and DBSCAN dimensions/minimum samples/presets (`eps`, `percentile`, `clusters`, `noise`, `other`). GeoJSON properties are `id`, `row`, `col`, `pca`, `k2`–`k8`, `db0`–`db4`; noise is `-1`.
- `embeddings.f32` is little-endian float32 `[count, dim]`, scene-centred, L2-normalized, and ordered identically to features.
- Preserve the single `state`/`render()` flow, rebuilt single-selection Calcite lists and `calciteListItemSelect` events. Imagery uses MediaLayer; `Map` is ArcGIS's class.
- Similarity defaults to `inputs: "single"`: each click replaces A (`selected`). In `inputs: "pair"`, the first click sets A and the second sets B (`second`); wait for both before scoring. Later clicks replace B. Switching input modes clears both. Normalize the add/subtract/element-wise multiply result, exclude the inputs, handle zero-length results, and keep cache keys in sync with both inputs and the operation.
- Keep the eight-slot cluster palette and click-to-isolate accessibility. Synchronize `SIMILAR_RAMP` and CSS `.ramp`: `#b2182b` → `#8f8e88` → `#7fcf7a`; similarity uses other-chip min/max scaling and 0.85 alpha.
- Popup help uses `HELP` in `app.js`, `data-help` buttons, and one Calcite popover. Show one help button for the hovered or keyboard-focused control row. Hide the previous button immediately. Keep help available on touch screens. Keep explanations direct. Use one action per numbered step. Use the same term for the same item. Test each topic and its connection to the selected control.

## Testing Guidelines

Run `../model/.venv/bin/python -m unittest discover -s tests -v` and `node --test tests/app.test.mjs`. No coverage target exists. Export requires at least 10 finite, nondegenerate embeddings and north-up imagery with an EPSG code; validate before replacing assets.

Check Spectrum, Similar, both clustering methods, isolation, and each exported view in a browser. For headless checks, use playwright-core with Chrome and `--use-angle=swiftshader`; wait for `document.querySelector("arcgis-map").ready`, screenshot each mode, and assert no console errors.

## Commit & Pull Request Guidelines

Use short imperative commit subjects. PRs should explain the behavior change, link relevant issues, and report verification results and skipped checks.
