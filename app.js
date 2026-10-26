// Clay embedding explorer: chip-level embeddings over the source imagery.
// Data comes from export_data.py (<view>/meta.json, chips.geojson, embeddings.f32, image.jpg).

// View folder: ?data=views/example, default data/. Plain relative paths only (no "..", no scheme).
const view = new URLSearchParams(location.search).get("data") ?? "data";
if (!/^[A-Za-z0-9_-]+(\/[A-Za-z0-9_-]+)*$/.test(view)) throw new Error(`invalid data folder: ${view}`);

// Start the data downloads before awaiting the SDK modules so both load in parallel.
async function loadData(file, type) {
  const response = await fetch(`${view}/${file}`);
  if (!response.ok) throw new Error(`Could not load ${view}/${file}: HTTP ${response.status}`);
  return response[type]();
}

const dataLoaded = Promise.all([
  loadData("meta.json", "json"),
  loadData("chips.geojson", "json"),
  loadData("embeddings.f32", "arrayBuffer"),
]);

const modulesLoaded = Promise.all(
  [
    "@arcgis/core/Map.js",
    "@arcgis/core/layers/GraphicsLayer.js",
    "@arcgis/core/Graphic.js",
    "@arcgis/core/geometry/Polygon.js",
    "@arcgis/core/layers/MediaLayer.js",
    "@arcgis/core/layers/support/ImageElement.js",
    "@arcgis/core/layers/support/ExtentAndRotationGeoreference.js",
    "@arcgis/core/geometry/Extent.js",
  ].map((m) => $arcgis.import(m)),
);
const [[meta, chips, embBuffer], modules] = await Promise.all([dataLoaded, modulesLoaded]).catch((error) => {
  document.getElementById("subtitle").textContent = error.message;
  throw error;
});
const [Map, GraphicsLayer, Graphic, Polygon, MediaLayer, ImageElement, ExtentAndRotationGeoreference, Extent] = modules;

// Dark-mode categorical slots, fixed order (dataviz reference palette).
const CLUSTER_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const NOISE_COLOR = "#c3c2b7"; // neutral, only drawn when noise is isolated
// Diverging red (not similar) -> neutral -> green (similar). Lightness rises
// monotonically (dark red to light green) so red-green colour-blind viewers still read the order.
const SIMILAR_RAMP = [
  [178, 24, 43],
  [143, 142, 136],
  [127, 207, 122],
];
const SELECTED_OUTLINE = { color: [255, 255, 255, 1], width: 2.5 };
const SECOND_OUTLINE = { ...SELECTED_OUTLINE, style: "dash" };
const NO_OUTLINE = { width: 0 }; // dense chip grids read better without borders
const TOP_MATCHES = 6;

const $ = (id) => document.getElementById(id);

const emb = new Float32Array(embBuffer); // row-major [count, dim], centred and L2-normalized
const { count: n, dim } = meta;
if (emb.length !== n * dim || chips.features.length !== n) {
  throw new Error(`data mismatch: ${chips.features.length} chips, ${emb.length / dim} embeddings`);
}

const state = { mode: "pca", inputs: "single", selected: null, second: null, operation: "add", method: "kmeans", k: 5, eps: 2, isolated: null };
let similarity = null;

const HELP = {
  map: {
    title: "Map",
    text: "A chip is a rectangular area of the image. An embedding is a numerical description of a chip. The app uses embeddings to compare chips. The app does not change the source image.",
    steps: ["Drag the map to move the image.", "Select Zoom in to show a smaller area.", "Select Zoom out to show a larger area."],
  },
  pca: {
    title: "Spectrum",
    text: "Spectrum shows three patterns in the chip embeddings. The app finds these patterns with principal component analysis (PCA). Each pattern controls one colour channel: red, green, or blue. Similar colours indicate similar embeddings. The percentages show how much embedding variation each pattern represents. The colours do not identify a verified ground type.",
    steps: ["Select Spectrum.", "Compare the chip colours.", "Read the percentages below the description."],
  },
  similar: {
    title: "Similar",
    text: "This view compares chip embeddings. A high cosine similarity score indicates a similar embedding direction. The app colours each chip by its score. The list shows the closest matches. A similar embedding does not confirm the same ground type.",
    steps: ["Select Similar.", "Select an input mode.", "Select a chip for A.", "In A and B, select a second chip.", "Read the Closest chips list."],
  },
  single: {
    title: "A only",
    text: "This mode uses one chip as the search input. A solid outline identifies A. Each new selection replaces A. The app updates the results after each selection. The app excludes A from the Closest chips list.",
    steps: ["Select A only.", "Select a chip on the map.", "Read the Closest chips list."],
  },
  pair: {
    title: "A and B",
    text: "This mode uses two chips as the search inputs. The app waits for both selections. A solid outline identifies A. A dashed outline identifies B. Later selections replace B. A change of input mode clears both selections.",
    steps: ["Select A and B.", "Select a chip for A.", "Select a second chip for B.", "Select the required vector operation."],
  },
  add: {
    title: "Add A + B",
    text: "The app adds each value in B to the value at the same position in A. Both inputs have equal weight. The app sets the result length to one before it calculates similarity. The app excludes A and B from the Closest chips list. Addition does not guarantee a mixture of the visible ground types.",
    steps: ["Select A and B.", "Select a chip for A.", "Select a second chip for B.", "Select Add A + B."],
  },
  subtract: {
    title: "Subtract A − B",
    text: "The app subtracts each value in B from the value at the same position in A. Input order changes the result. Swap A/B changes A − B to B − A. Identical input embeddings produce a zero-length result. The app cannot calculate similarity for that result. Subtraction does not remove an object from the image.",
    steps: ["Select A and B.", "Select a chip for A.", "Select a second chip for B.", "Select A − B."],
  },
  multiply: {
    title: "Multiply A × B",
    text: "The app multiplies each value in A by the value at the same position in B. Values with the same sign produce a positive result. Values with different signs produce a negative result. The app sets the result length to one before it calculates similarity. Multiplication does not identify a shared object in the images.",
    steps: ["Select A and B.", "Select a chip for A.", "Select a second chip for B.", "Select A × B."],
  },
  swap: {
    title: "Swap A/B",
    text: "This control exchanges the two inputs. A becomes B. B becomes A. Subtraction changes from A − B to B − A. Addition and multiplication keep the same result. This control is available after both inputs are selected.",
    steps: ["Select Swap A/B.", "Read the updated Closest chips list."],
  },
  clear: {
    title: "Clear",
    text: "Clear removes both selections. The app removes the selection outlines. The app clears the Closest chips list. The current input mode and vector operation remain selected. The source image does not change.",
    steps: ["Select Clear.", "Select a chip for A."],
  },
  score: {
    title: "Cosine similarity",
    text: "The score compares directions in the embedding space. A score of 1 means the same direction. A score of 0 means perpendicular directions. A score of −1 means opposite directions. Green shows the highest scores in the current results. Red shows the lowest scores. The selected inputs do not set the colour range.",
    steps: ["Read the score beside a chip in the list.", "Compare the score with the colour scale."],
  },
  matches: {
    title: "Closest chips",
    text: "The list shows up to six chips with the highest similarity scores. The app excludes the selected inputs. In A only, a list selection replaces A. In A and B, a list selection replaces B. Row and column values identify the chip position in the source image. The value at the right is the cosine similarity score.",
    steps: ["Select a chip in the list.", "Read the updated Closest chips list."],
  },
  clusters: {
    title: "Clusters",
    text: "A cluster is a group of chips with similar embeddings. Each colour identifies one cluster. K-means assigns every chip to a cluster. DBSCAN can leave chips outside the clusters. These chips are noise. A cluster number does not identify a verified ground type.",
    steps: ["Select Clusters.", "Select K-means or DBSCAN.", "Read the cluster list."],
  },
  kmeans: {
    title: "K-means",
    text: "K-means divides the chip embeddings into a specified number of clusters. It assigns every chip to one cluster. Similar embeddings can occur on both sides of a cluster boundary. The selected number does not confirm the number of ground types in the image.",
    steps: ["Select K-means.", "Move the Number of clusters slider to the required value.", "Read the cluster list."],
  },
  dbscan: {
    title: "DBSCAN",
    text: "DBSCAN finds dense clusters of similar embeddings. The app compares ten values derived from each embedding. A dense core requires at least ten nearby chips. Chips outside the clusters are noise. Noise chips have no colour until you select Noise. DBSCAN does not assign every chip to a cluster.",
    steps: ["Select DBSCAN.", "Move the Neighbourhood radius (eps) slider to the required value.", "Read the cluster and noise counts."],
  },
  count: {
    title: "Number of clusters",
    text: "This slider sets the number of K-means clusters. A larger number divides the chips into more clusters. Every chip remains in one cluster. The app calculates the clusters before it displays the image. The slider selects one of these results.",
    steps: ["Select K-means.", "Move the Number of clusters slider to the required value."],
  },
  radius: {
    title: "Neighbourhood radius (eps)",
    text: "Each slider step selects a radius calculated for this image. A larger radius can join clusters. A larger radius can move noise chips into a cluster. The distance is measured between embeddings. It is not a distance on the map. Radius values can differ between images.",
    steps: ["Select DBSCAN.", "Move the Neighbourhood radius (eps) slider one step.", "Read the updated cluster and noise counts."],
  },
  isolate: {
    title: "Show one cluster",
    text: "A cluster selection shows only that cluster. The other chip colours are hidden. Noise shows the chips that DBSCAN left outside its clusters. Other can contain several small DBSCAN clusters. The source image remains visible.",
    steps: ["Select a cluster in the list.", "Select the same cluster again to show all clusters."],
  },
  opacity: {
    title: "Overlay opacity (%)",
    text: "Opacity controls the strength of the chip colours. At 0 percent, the chip colours are hidden. A larger value makes the chip colours stronger. Selection outlines also become less visible at lower values. Opacity does not change similarity scores or cluster assignments.",
    steps: ["Move the Overlay opacity (%) slider to the required value."],
  },
};

for (const button of document.querySelectorAll("[data-help]")) {
  button.addEventListener("click", () => {
    const topic = HELP[state[button.dataset.help] ?? button.dataset.help];
    const popup = $("help-popup");
    const text = document.createElement("p");
    text.textContent = topic.text;
    const steps = document.createElement("ol");
    for (const instruction of topic.steps) {
      const step = document.createElement("li");
      step.textContent = instruction;
      steps.append(step);
    }
    $("help-content").replaceChildren(text, steps);
    popup.heading = topic.title;
    popup.label = `${topic.title} help`;
    popup.referenceElement = button;
    popup.open = true;
  });
}

$("help-popup").addEventListener("calcitePopoverClose", () => {
  // Return keyboard focus when closing help leaves focus on the page.
  const popup = $("help-popup");
  if (!popup.open && document.activeElement === document.body) popup.referenceElement?.setFocus?.();
});

// Pointer focus can remain after the pointer leaves a control.
document.addEventListener("keydown", () => {
  document.documentElement.dataset.keyboardNavigation = "";
}, true);
for (const type of ["pointerdown", "pointermove"]) {
  document.addEventListener(type, () => {
    delete document.documentElement.dataset.keyboardNavigation;
  }, { capture: true, passive: true });
}

// --- Map ------------------------------------------------------------------

const [xmin, ymin, xmax, ymax] = meta.image.extent;
const imageExtent = new Extent({ xmin, ymin, xmax, ymax, spatialReference: { wkid: meta.image.wkid } });

const imagery = new MediaLayer({
  title: meta.source,
  source: new ImageElement({
    image: `${view}/image.jpg`,
    georeference: new ExtentAndRotationGeoreference({ extent: imageExtent }),
  }),
});

const graphics = chips.features.map(
  (f) =>
    new Graphic({
      geometry: new Polygon({ rings: f.geometry.coordinates, spatialReference: { wkid: 4326 } }),
      attributes: f.properties,
    }),
);
const overlay = new GraphicsLayer({ title: "Chips", graphics, opacity: 0.7 });

const viewElement = document.querySelector("arcgis-map");
viewElement.map = new Map({ layers: [imagery, overlay] });
viewElement.extent = imageExtent.clone().expand(1.15);
viewElement.background = { color: [26, 26, 25] };
viewElement.constraints = { rotationEnabled: false };
viewElement.popupDisabled = true;

viewElement.addEventListener("arcgisViewClick", async (event) => {
  const { results } = await viewElement.hitTest(event.detail, { include: [overlay] });
  const hit = results.find((r) => r.type === "graphic");
  if (hit) select(hit.graphic.attributes.id);
});

// --- Rendering ------------------------------------------------------------

// Keep both input outlines when switching modes: solid A, dashed B.
function paint(g, color) {
  const outline = g.attributes.id === state.selected ? SELECTED_OUTLINE : g.attributes.id === state.second ? SECOND_OUTLINE : NO_OUTLINE;
  g.symbol = { type: "simple-fill", color, outline };
}

function similarities(i, second = null, operation = "add") {
  if (similarity?.id === i && similarity.second === second && similarity.operation === operation) return similarity;
  const query = new Float32Array(dim);
  const a = emb.subarray(i * dim, (i + 1) * dim);
  const b = second === null ? null : emb.subarray(second * dim, (second + 1) * dim);
  for (let d = 0; d < dim; d++) {
    if (b === null) query[d] = a[d];
    else if (operation === "add") query[d] = a[d] + b[d];
    else if (operation === "subtract") query[d] = a[d] - b[d];
    else query[d] = a[d] * b[d];
  }
  const norm = Math.hypot(...query);
  if (!Number.isFinite(norm) || norm === 0) return null;
  for (let d = 0; d < dim; d++) query[d] /= norm;

  const sims = new Float32Array(n);
  for (let j = 0; j < n; j++) {
    const candidate = emb.subarray(j * dim, (j + 1) * dim);
    let dot = 0;
    for (let d = 0; d < dim; d++) dot += query[d] * candidate[d];
    sims[j] = Math.max(-1, Math.min(1, dot));
  }
  const ranked = [...sims.keys()].filter((j) => j !== i && j !== second).sort((a, b) => sims[b] - sims[a]);
  return (similarity = { id: i, second, operation, sims, ranked });
}

function render() {
  for (const section of document.querySelectorAll("section[data-mode]")) {
    section.hidden = section.dataset.mode !== state.mode;
  }
  if (state.mode === "pca") renderPca();
  if (state.mode === "similar") renderSimilar();
  if (state.mode === "clusters") renderClusters();
}

function renderPca() {
  for (const g of graphics) paint(g, g.attributes.pca);
}

function renderSimilar() {
  const pair = state.inputs === "pair";
  const has = state.selected !== null;
  const ready = has && (!pair || state.second !== null);
  $("similar-hint").hidden = has;
  $("similar-hint").textContent = pair
    ? "Select a chip for A. Select a second chip for B. Select a vector operation."
    : "Select a chip on the map. The app lists the closest matches.";
  $("vector-controls").hidden = !has;
  $("vector-a").textContent = has ? `A: Chip ${state.selected} (solid outline)` : "";
  $("vector-b").hidden = !pair;
  $("vector-b").textContent = state.second === null ? "B: Select a chip" : `B: Chip ${state.second} (dashed outline)`;
  $("operation").hidden = !pair;
  $("operation-row").hidden = !pair;
  $("swap-vectors").hidden = !pair;
  $("swap-row").hidden = !pair;
  $("swap-vectors").disabled = state.second === null;
  const result = ready ? similarities(state.selected, pair ? state.second : null, state.operation) : null;
  $("similar-body").hidden = result === null;
  $("vector-status").textContent = !has ? "" : !ready
    ? "Select B to see combined-vector matches."
    : result === null ? "This operation produces a zero-length vector. Choose different chips or another operation."
    : pair ? "Showing matches for the combined vector. A new selection replaces B." : "Showing matches for A. A new selection replaces A.";
  if (result === null) {
    for (const g of graphics) paint(g, [0, 0, 0, 0]);
    $("matches").replaceChildren();
    return;
  }
  const { sims, ranked } = result;
  const hi = sims[ranked[0]] ?? 1;
  const lo = sims[ranked.at(-1)] ?? -1;
  for (const g of graphics) {
    const { id } = g.attributes;
    const t = Math.max(0, Math.min(1, (sims[id] - lo) / (hi - lo || 1)));
    paint(g, [...ramp(t), 0.85]);
  }
  $("sim-min").textContent = lo.toFixed(2);
  $("sim-max").textContent = hi.toFixed(2);

  $("matches").replaceChildren(
    ...ranked.slice(0, TOP_MATCHES).map((j) => {
      const { row, col } = graphics[j].attributes;
      return listItem(j, `Chip ${j}`, `row ${row} · col ${col}`, sims[j].toFixed(2));
    }),
  );
}

function ramp(t) {
  const [a, b, u] = t < 0.5 ? [SIMILAR_RAMP[0], SIMILAR_RAMP[1], t * 2] : [SIMILAR_RAMP[1], SIMILAR_RAMP[2], t * 2 - 1];
  return a.map((v, i) => Math.round(v + (b[i] - v) * u));
}

function renderClusters() {
  const dbscan = state.method === "dbscan";
  for (const el of document.querySelectorAll("[data-method]")) el.hidden = el.dataset.method !== state.method;
  const preset = meta.dbscan.presets[state.eps];
  const key = dbscan ? `db${state.eps}` : `k${state.k}`;
  const counts = new Array(dbscan ? preset.clusters : state.k).fill(0);
  let noise = 0;
  for (const g of graphics) {
    const c = g.attributes[key]; // -1 = DBSCAN noise
    if (c < 0) noise++;
    else counts[c]++;
    const shown = state.isolated === null ? c >= 0 : state.isolated === c;
    paint(g, shown ? (CLUSTER_COLORS[c] ?? NOISE_COLOR) : [0, 0, 0, 0]);
  }
  const items = counts.map((count, c) => {
    const other = dbscan && preset.other && c === CLUSTER_COLORS.length - 1;
    return clusterItem(c, other ? "Other" : `Cluster ${c + 1}`, count, CLUSTER_COLORS[c]);
  });
  if (dbscan) {
    items.push(clusterItem(-1, "Noise", noise, NOISE_COLOR));
    $("eps-value").textContent =
      `eps ${preset.eps} (p${preset.percentile} of ${meta.dbscan.min_samples}-NN cosine distance, ` +
      `${meta.dbscan.pca_dims}-d PCA) · ${preset.clusters} clusters · ${(preset.noise * 100).toFixed(0)}% noise`;
  }
  $("clusters").replaceChildren(...items);
}

function clusterItem(value, label, count, color) {
  const item = listItem(value, label, "", `${count} chips`);
  item.selected = state.isolated === value;
  const swatch = document.createElement("span");
  swatch.slot = "content-start";
  swatch.className = "swatch";
  swatch.style.background = color;
  item.append(swatch);
  return item;
}

function listItem(value, label, description, end) {
  const item = document.createElement("calcite-list-item");
  Object.assign(item, { value: String(value), label, description });
  const endText = document.createElement("span");
  endText.slot = "content-end";
  endText.className = "muted";
  endText.textContent = end;
  item.append(endText);
  return item;
}

// --- Interaction ----------------------------------------------------------

function setMode(mode) {
  state.mode = mode;
  for (const item of $("mode").querySelectorAll("calcite-segmented-control-item")) {
    item.checked = item.value === mode;
  }
  render();
}

function select(id) {
  if (state.inputs === "single" || state.selected === null) state.selected = id;
  else state.second = id;
  setMode("similar");
}

$("mode").addEventListener("calciteSegmentedControlChange", (e) => setMode(e.target.value));

$("inputs").addEventListener("calciteSegmentedControlChange", (e) => {
  state.inputs = e.target.value;
  state.selected = null;
  state.second = null;
  render();
});

$("operation").addEventListener("calciteSegmentedControlChange", (e) => {
  state.operation = e.target.value;
  render();
});

$("swap-vectors").addEventListener("click", () => {
  if (state.second === null) return;
  [state.selected, state.second] = [state.second, state.selected];
  render();
});

$("clear-vectors").addEventListener("click", () => {
  state.selected = null;
  state.second = null;
  render();
});

// Cluster controls: element id == state key; any change clears the isolated cluster.
for (const [id, event] of [["method", "calciteSegmentedControlChange"], ["k", "calciteSliderInput"], ["eps", "calciteSliderInput"]]) {
  $(id).addEventListener(event, (e) => {
    state[id] = id === "method" ? e.target.value : Number(e.target.value);
    state.isolated = null;
    render();
  });
}

$("matches").addEventListener("calciteListItemSelect", (e) => select(Number(e.target.value)));

$("clusters").addEventListener("calciteListItemSelect", (e) => {
  state.isolated = e.target.selected ? Number(e.target.value) : null;
  render();
});

$("opacity").addEventListener("calciteSliderInput", (e) => {
  overlay.opacity = Number(e.target.value) / 100;
});

// --- Init -----------------------------------------------------------------

[$("k").min, $("k").max] = meta.k_range;
$("eps").max = meta.dbscan.presets.length - 1;
$("title").textContent = `${meta.title} · Clay embeddings`;
$("subtitle").textContent = `${n} chips of ${meta.chip_px} px · ${dim}-d · ${meta.model}`;
$("pca-variance").textContent =
  "Variance explained: " + meta.pca_variance.map((v, i) => `PC${i + 1} ${(v * 100).toFixed(0)}%`).join(" · ");
render();
