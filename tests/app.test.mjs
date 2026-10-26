import assert from "node:assert/strict";
import { existsSync, readdirSync } from "node:fs";
import { readFile } from "node:fs/promises";
import test from "node:test";
import vm from "node:vm";

const source = await readFile(new URL("../app.js", import.meta.url), "utf8");
const html = await readFile(new URL("../index.html", import.meta.url), "utf8");
const ids = new Set(Array.from(html.matchAll(/\bid="([^"]+)"/g), (match) => match[1]));
const helpTags = Array.from(html.matchAll(/<calcite-action\b[^>]*\bdata-help="([^"]+)"[^>]*>/g));

class Element {
  children = [];
  listeners = {};
  style = {};
  addEventListener(event, listener) { this.listeners[event] = listener; }
  replaceChildren(...children) { this.children = children; }
  append(child) { this.children.push(child); }
  querySelectorAll() { return []; }
  setFocus() { this.focused = true; }
}

class ArcGISObject {
  constructor(options) { Object.assign(this, options); }
  clone() { return new ArcGISObject(this); }
  expand() { return this; }
}

async function boot({ search = "", missing = null, fixtures = null } = {}) {
  const elements = new Map();
  const element = (id) => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const requests = [];
  const helpButtons = helpTags.map((tag) => Object.assign(new Element(), { dataset: { help: tag[1] } }));
  const document = {
    body: new Element(),
    documentElement: { dataset: {} },
    listeners: {},
    addEventListener(event, listener) { this.listeners[event] = listener; },
    getElementById: (id) => {
      assert.ok(ids.has(id), `Missing HTML element: ${id}`);
      return element(id);
    },
    querySelector: element,
    querySelectorAll: (selector) => selector === "[data-help]" ? helpButtons : [],
    createElement: () => new Element(),
  };
  document.activeElement = document.body;
  const context = {
    document, URLSearchParams, Float32Array,
    location: { search },
    $arcgis: { import: async () => ArcGISObject },
    fetch: async (path) => {
      requests.push(path);
      if (missing && path.endsWith(missing)) return { ok: false, status: 404 };
      if (fixtures) {
        const data = fixtures[path.split("/").at(-1)];
        return { ok: true, json: async () => data, arrayBuffer: async () => data };
      }
      const bytes = await readFile(new URL(`../${path}`, import.meta.url));
      return {
        ok: true,
        json: async () => JSON.parse(bytes.toString()),
        arrayBuffer: async () => new Uint8Array(bytes).buffer,
      };
    },
  };
  const ready = vm.runInNewContext(
    `(async () => { ${source}\nreturn { state, graphics, meta, emb, similarities, select, setMode, render }; })()`,
    context, { filename: "app.js" },
  );
  return { ready, element, requests, helpButtons, document };
}

// Every exported view on disk (data/ and views/*/ are generated and gitignored).
const root = new URL("../", import.meta.url);
const views = existsSync(new URL("views/", root)) ? readdirSync(new URL("views/", root)).map((name) => `views/${name}`) : [];
const folders = ["data", ...views].filter((folder) => existsSync(new URL(`${folder}/meta.json`, root)));
test("at least one exported view exists", () => assert.ok(folders.length, "export a view to data/ or views/<name>/ first"));

for (const folder of folders) {
  test(`${folder}: render every mode, preset, and cluster isolation`, async () => {
    const { ready, element, requests } = await boot({ search: `?data=${folder}` });
    const app = await ready;
    assert.equal(requests.length, 3);
    assert.equal(app.graphics.length, app.meta.count);
    assert.ok(app.graphics.every((g) => g.symbol.color === g.attributes.pca));
    app.select(0);
    const matches = element("matches").children;
    assert.equal(matches.length, 6);
    assert.ok(matches.every((item) => item.value !== "0"));
    assert.ok(app.graphics.every((g) => g.symbol.color.every(Number.isFinite)));
    const first = app.similarities(0);
    assert.equal(app.similarities(0), first);
    assert.ok(Math.abs(first.sims[0] - 1) < 1e-5);
    app.setMode("pca");
    app.setMode("similar");
    assert.equal(app.similarities(0), first);
    assert.notEqual(app.similarities(1), first);
    app.select(1);
    assert.equal(app.state.selected, 1);
    assert.equal(app.state.second, null);
    assert.equal(element("operation").hidden, true);
    element("inputs").listeners.calciteSegmentedControlChange({ target: { value: "pair" } });
    app.select(0);
    assert.equal(element("similar-body").hidden, true);
    assert.equal(element("matches").children.length, 0);
    app.select(1);
    assert.equal(app.state.selected, 0);
    assert.equal(app.state.second, 1);
    for (const operation of ["add", "subtract", "multiply"]) {
      element("operation").listeners.calciteSegmentedControlChange({ target: { value: operation } });
      const combined = app.similarities(0, 1, operation);
      assert.equal(app.similarities(0, 1, operation), combined);
      assert.ok(combined.sims.every((score) => Number.isFinite(score) && score >= -1 && score <= 1));
      assert.ok(element("matches").children.every((item) => !["0", "1"].includes(item.value)));
      assert.ok(app.graphics.every((g) => g.symbol.color.every((value) => Number.isFinite(value) && value >= 0 && value <= 255)));
    }
    for (const k of [2, 5, 8]) {
      Object.assign(app.state, { method: "kmeans", k });
      app.setMode("clusters");
      assert.equal(element("clusters").children.length, k);
    }
    app.state.method = "dbscan";
    for (let eps = 0; eps < app.meta.dbscan.presets.length; eps++) {
      app.state.eps = eps;
      app.render();
      assert.equal(element("clusters").children.length, app.meta.dbscan.presets[eps].clusters + 1);
    }
    app.state.isolated = -1;
    app.render();
    assert.ok(app.graphics.every((g) => g.attributes.db4 === -1 ? g.symbol.color === "#c3c2b7" : g.symbol.color[3] === 0));
  });
}

async function bootVectors() {
  const q = Math.SQRT1_2;
  const vectors = [[1, 0, 0], [0, 1, 0], [q, q, 0], [q, -q, 0], [-1, 0, 0], [0, 0, 1], [q, 0, q], [-q, q, 0]];
  return boot({ fixtures: {
    "meta.json": { count: vectors.length, dim: 3, image: { extent: [0, 0, 1, 1], wkid: 4326 }, k_range: [2, 8], dbscan: { presets: Array(5).fill({}) }, pca_variance: [1, 0, 0] },
    "chips.geojson": { features: vectors.map((_, id) => ({ geometry: { coordinates: [] }, properties: { id, row: id, col: 0, pca: "#000000" } })) },
    "embeddings.f32": new Float32Array(vectors.flat()).buffer,
  } });
}

test("vector algebra finds the expected directions and excludes the inputs", async () => {
  const { ready } = await bootVectors();
  const app = await ready;
  for (const [a, b, operation, expected] of [[0, 1, "add", 2], [0, 1, "subtract", 3], [1, 0, "subtract", 7], [2, 6, "multiply", 0]]) {
    const result = app.similarities(a, b, operation);
    assert.equal(result.ranked[0], expected);
    assert.ok(Math.abs(result.sims[expected] - 1) < 1e-6);
    assert.ok(result.ranked.every((id) => id !== a && id !== b));
  }
  const added = app.similarities(0, 1, "add");
  assert.ok(Math.abs(added.sims[0] - Math.SQRT1_2) < 1e-6);
  assert.ok(Math.abs(added.sims[3]) < 1e-6);
  assert.notEqual(app.similarities(0, 1, "subtract"), added);
});

test("two clicks set A and B, later clicks replace B, and swap/clear update the query", async () => {
  const { ready, element } = await bootVectors();
  const app = await ready;
  element("inputs").listeners.calciteSegmentedControlChange({ target: { value: "pair" } });
  assert.equal(app.state.selected, null);
  app.select(0);
  assert.equal(element("vector-controls").hidden, false);
  assert.equal(element("swap-vectors").disabled, true);
  assert.equal(app.state.second, null);
  assert.equal(element("similar-body").hidden, true);
  assert.equal(element("matches").children.length, 0);
  assert.match(element("vector-status").textContent, /Select B/);
  element("operation").listeners.calciteSegmentedControlChange({ target: { value: "subtract" } });
  assert.equal(element("similar-body").hidden, true);
  element("operation").listeners.calciteSegmentedControlChange({ target: { value: "add" } });
  app.select(1);
  assert.equal(element("matches").children[0].value, "2");
  assert.equal(element("swap-vectors").disabled, false);
  assert.equal(app.graphics[1].symbol.outline.style, "dash");
  element("operation").listeners.calciteSegmentedControlChange({ target: { value: "subtract" } });
  assert.equal(element("matches").children[0].value, "3");
  element("swap-vectors").listeners.click();
  assert.equal(element("matches").children[0].value, "7");
  assert.equal(app.state.selected, 1);
  app.select(6);
  assert.equal(app.state.selected, 1);
  assert.equal(app.state.second, 6);
  element("clear-vectors").listeners.click();
  assert.equal(app.state.selected, null);
  assert.equal(app.state.second, null);
  assert.equal(element("vector-controls").hidden, true);
  assert.equal(element("similar-body").hidden, true);
  assert.equal(element("matches").children.length, 0);
  assert.ok(app.graphics.every((g) => g.symbol.outline.width === 0));
  app.select(5);
  assert.equal(app.state.selected, 5);
  assert.equal(app.state.second, null);
});

test("zero-length sums, differences, and products hide stale matches and recover", async () => {
  const { ready, element } = await bootVectors();
  const app = await ready;
  element("inputs").listeners.calciteSegmentedControlChange({ target: { value: "pair" } });
  for (const [a, b, operation] of [[0, 4, "add"], [0, 0, "subtract"], [0, 1, "multiply"]]) {
    element("clear-vectors").listeners.click();
    app.select(a);
    element("operation").listeners.calciteSegmentedControlChange({ target: { value: operation } });
    app.select(b);
    assert.equal(app.similarities(a, b, operation), null);
    assert.equal(element("similar-body").hidden, true);
    assert.equal(element("matches").children.length, 0);
    assert.match(element("vector-status").textContent, /zero-length vector/);
    assert.ok(app.graphics.every((g) => g.symbol.color.every(Number.isFinite)));
  }
  element("operation").listeners.calciteSegmentedControlChange({ target: { value: "add" } });
  assert.equal(element("similar-body").hidden, false);
  assert.equal(element("matches").children[0].value, "2");
});

test("A only keeps the original one-click similarity for map and match-list selections", async () => {
  const { ready, element } = await bootVectors();
  const app = await ready;
  assert.equal(app.state.inputs, "single");
  app.select(0);
  assert.equal(element("similar-body").hidden, false);
  assert.equal(element("operation").hidden, true);
  assert.equal(element("vector-b").hidden, true);
  assert.equal(element("swap-vectors").hidden, true);
  app.select(1);
  assert.equal(app.state.selected, 1);
  assert.equal(app.state.second, null);
  assert.equal(element("similar-body").hidden, false);
  assert.ok(element("matches").children.every((item) => item.value !== "1"));
  assert.ok(app.similarities(1).ranked.includes(0));
  element("matches").listeners.calciteListItemSelect({ target: { value: "2" } });
  assert.equal(app.state.selected, 2);
  assert.equal(app.state.second, null);
  assert.equal(element("similar-body").hidden, false);
});

test("switching input modes clears selections, outlines, and stale results", async () => {
  const { ready, element } = await bootVectors();
  const app = await ready;
  app.select(0);
  element("inputs").listeners.calciteSegmentedControlChange({ target: { value: "pair" } });
  assert.equal(app.state.selected, null);
  assert.equal(app.state.second, null);
  assert.equal(element("similar-body").hidden, true);
  assert.equal(element("matches").children.length, 0);
  app.select(0);
  assert.equal(element("similar-body").hidden, true);
  app.select(1);
  assert.equal(element("similar-body").hidden, false);
  element("inputs").listeners.calciteSegmentedControlChange({ target: { value: "single" } });
  assert.equal(app.state.selected, null);
  assert.equal(app.state.second, null);
  assert.equal(element("similar-body").hidden, true);
  assert.equal(element("operation").hidden, true);
  assert.equal(element("operation-row").hidden, true);
  assert.equal(element("swap-row").hidden, true);
  assert.ok(app.graphics.every((g) => g.symbol.outline.width === 0));
  app.select(2);
  assert.equal(element("similar-body").hidden, false);
  assert.equal(app.state.second, null);
});

test("help covers every capability and follows the selected control without changing the query", async () => {
  const { ready, element, helpButtons } = await bootVectors();
  const app = await ready;
  const topics = {
    map: { map: "Map" },
    mode: { pca: "Spectrum", similar: "Similar", clusters: "Clusters" },
    inputs: { single: "A only", pair: "A and B" },
    operation: { add: "Add A + B", subtract: "Subtract A − B", multiply: "Multiply A × B" },
    swap: { swap: "Swap A/B" },
    clear: { clear: "Clear" },
    score: { score: "Cosine similarity" },
    matches: { matches: "Closest chips" },
    method: { kmeans: "K-means", dbscan: "DBSCAN" },
    count: { count: "Number of clusters" },
    radius: { radius: "Neighbourhood radius (eps)" },
    isolate: { isolate: "Show one cluster" },
    opacity: { opacity: "Overlay opacity (%)" },
  };
  assert.deepEqual(helpButtons.map((button) => button.dataset.help).sort(), Object.keys(topics).sort());
  for (const tag of helpTags) assert.match(tag[0], /label="Help for [^"]+"/);
  for (const button of helpButtons) {
    const key = button.dataset.help;
    for (const [value, title] of Object.entries(topics[key])) {
      if (key in app.state) app.state[key] = value;
      const before = JSON.stringify(app.state);
      button.listeners.click();
      assert.equal(element("help-popup").open, true);
      assert.equal(element("help-popup").referenceElement, button);
      assert.equal(element("help-popup").heading, title);
      assert.equal(element("help-popup").label, `${title} help`);
      assert.equal(JSON.stringify(app.state), before);
      const [description, steps] = element("help-content").children;
      assert.ok(description.textContent.length > 50, title);
      for (const sentence of description.textContent.split(/[.!?]\s+/)) {
        assert.ok(sentence.trim().split(/\s+/).length <= 25, `${title}: ${sentence}`);
      }
      assert.ok(steps.children.length > 0, title);
      for (const step of steps.children) {
        assert.match(step.textContent, /^(Select|Drag|Read|Move|Compare|In A and B, select) /);
        assert.ok(step.textContent.split(/\s+/).length <= 20, `${title}: ${step.textContent}`);
        assert.doesNotMatch(step.textContent, /;|\bthen\b/i);
      }
    }
  }
});

test("closing help returns lost focus without moving focus from another control", async () => {
  const { ready, element, helpButtons, document } = await bootVectors();
  await ready;
  const button = helpButtons[0];
  button.listeners.click();
  element("help-popup").open = false;
  element("help-popup").listeners.calcitePopoverClose();
  assert.equal(button.focused, true);
  button.focused = false;
  document.activeElement = element("inputs");
  element("help-popup").listeners.calcitePopoverClose();
  assert.equal(button.focused, false);
  document.activeElement = document.body;
  element("help-popup").open = true;
  element("help-popup").listeners.calcitePopoverClose();
  assert.equal(button.focused, false);
  element("help-popup").open = false;
  element("help-popup").referenceElement = "map-help";
  assert.doesNotThrow(() => element("help-popup").listeners.calcitePopoverClose());
});

test("pointer movement clears keyboard help without changing the selected query", async () => {
  const { ready, document } = await bootVectors();
  const app = await ready;
  app.select(0);
  const before = JSON.stringify(app.state);
  assert.equal(document.documentElement.dataset.keyboardNavigation, undefined);
  document.listeners.keydown({ key: "Tab" });
  assert.equal(document.documentElement.dataset.keyboardNavigation, "");
  document.listeners.pointermove({ pointerType: "mouse" });
  assert.equal(document.documentElement.dataset.keyboardNavigation, undefined);
  document.listeners.keydown({ key: "Enter" });
  assert.equal(document.documentElement.dataset.keyboardNavigation, "");
  document.listeners.pointerdown({ pointerType: "mouse" });
  assert.equal(document.documentElement.dataset.keyboardNavigation, undefined);
  document.listeners.keydown({ key: "Tab" });
  assert.equal(document.documentElement.dataset.keyboardNavigation, "");
  assert.equal(JSON.stringify(app.state), before);
});

test("missing data reports HTTP status in the panel", async () => {
  const { ready, element } = await boot({ missing: "meta.json" });
  await assert.rejects(ready, /Could not load data\/meta.json: HTTP 404/);
  assert.match(element("subtitle").textContent, /HTTP 404/);
});

test("unsafe view paths are rejected before fetching", async () => {
  for (const path of ["../etc", "/data", "https://example.com", "views//example"]) {
    const { ready, requests } = await boot({ search: `?data=${encodeURIComponent(path)}` });
    await assert.rejects(ready, /invalid data folder/);
    assert.equal(requests.length, 0);
  }
});
