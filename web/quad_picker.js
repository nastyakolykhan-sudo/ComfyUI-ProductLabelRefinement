import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Auto Draft's quad, picked on the scene: click the packshot outline's bounding box corners in order (TL TR BR BL),
// drag them to adjust, zoom with the wheel. The picker only edits the quad STRING widget, as "x,y x,y x,y x,y" in
// scene px, so the node's inputs, saved workflows and the API runs are unchanged.

const NODE_CLASS = "ImagePostAutoDraft";
const CORNERS = ["TL", "TR", "BR", "BL"];
const CORNER_NAMES = ["top-left", "top-right", "bottom-right", "bottom-left"];
const ACCENT = "#22d3ee";
const WARN = "#f59e0b";
const HANDLE_R = 6;     // screen px
const MAX_SCALE = 16;   // canvas px per scene px at the deepest zoom
const REPORT_MIN = 160; // px for the draft's report box under the picker

const widget = (node, name) => node.widgets?.find((w) => w.name === name);
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), Math.max(lo, hi));

function el(tag, style = {}, props = {}) {
    const e = document.createElement(tag);
    Object.assign(e.style, style);
    return Object.assign(e, props);
}

const BUTTON_STYLE = {
    background: "#2a2a2a", color: "#ddd", border: "1px solid #444", borderRadius: "6px",
    font: "12px sans-serif", padding: "3px 10px", cursor: "pointer", whiteSpace: "nowrap",
};

// The quad as the node reads it (nodes.py _quad): 8 numbers are 4 corners, 4 numbers a box x0,y0,x1,y1.
function parseQuad(text) {
    const v = (String(text ?? "").match(/[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?/g) || []).map(Number);
    if (!v.length) return { points: [] };
    if (v.length === 4) {
        const [x0, y0, x1, y1] = v;
        return { points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], box: true };
    }
    if (v.length === 8) return { points: [0, 2, 4, 6].map((i) => [v[i], v[i + 1]]) };
    return { points: [], error: `${v.length} numbers: the quad needs 4 corners (8 numbers) or a box (4)` };
}

const formatQuad = (points) => points.map(([x, y]) => `${Math.round(x)},${Math.round(y)}`).join(" ");

// TL TR BR BL runs clockwise on screen, and stays clockwise however the product is turned.
function quadProblem(p) {
    let sign = 0;
    for (let i = 0; i < 4; i++) {
        const [a, b, c] = [p[i], p[(i + 1) % 4], p[(i + 2) % 4]];
        const cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if (Math.abs(cross) < 1e-9) continue;
        if (sign && Math.sign(cross) !== sign) return "the corners cross: place them in order TL TR BR BL";
        sign = Math.sign(cross);
    }
    return sign < 0 ? "the corners run anticlockwise: place them in order TL TR BR BL" : null;
}

// A Load Image's file value, "sub/name.png [input]" -> /view URL.
function viewURL(value) {
    let type = "input";
    let name = String(value).trim();
    const m = name.match(/^(.*) \[(input|output|temp)\]$/);
    if (m) [, name, type] = m;
    const i = name.lastIndexOf("/");
    const subfolder = i < 0 ? "" : name.slice(0, i);
    const filename = i < 0 ? name : name.slice(i + 1);
    return api.apiURL(`/view?${new URLSearchParams({ filename, subfolder, type })}`);
}

// The image feeding `scene`: a Load Image's file (through Reroutes), else the upstream node's preview.
function sceneSource(node) {
    const slot = node.inputs?.findIndex((i) => i.name === "scene") ?? -1;
    if (slot < 0 || node.inputs[slot].link == null) return { message: "Connect a Load Image to scene" };
    let origin = node.getInputNode?.(slot);
    for (let hops = 0; origin && origin.type === "Reroute" && hops < 16; hops++) origin = origin.getInputNode?.(0);
    if (!origin) return { message: "Connect a Load Image to scene" };
    const file = widget(origin, "image");
    if (file && typeof file.value === "string" && file.value && Array.isArray(file.options?.values)) {
        return { url: viewURL(file.value), label: file.value };
    }
    const src = origin.imgs?.[0]?.src;
    if (src) return { url: src, label: origin.title || origin.type };
    return { message: `No image from ${origin.title || origin.type} yet: run the graph once, or use a Load Image` };
}

class QuadPicker {
    constructor(node) {
        this.node = node;
        this.points = [];         // scene px, in TL TR BR BL order; fewer than 4 while picking
        this.box = false;         // the quad text is a box x0,y0,x1,y1
        this.error = null;
        this.lastText = null;     // the quad text the points were read from or written as
        this.image = null;
        this.sourceURL = null;
        this.sourceLabel = "";
        this.message = "Connect a Load Image to scene";
        this.view = { zoom: 1, cx: 0, cy: 0 };  // zoom relative to fit; scene px at the canvas centre
        this.hover = null;        // scene px under the pointer
        this.hot = -1;            // corner under the pointer, or being dragged
        this.selected = -1;       // corner moved by the arrow keys
        this.drag = null;

        this.element = el("div", {
            display: "flex", flexDirection: "column", width: "100%", height: "100%", gap: "4px",
            font: "12px sans-serif", color: "#ccc", userSelect: "none",
        });
        this.header = el("div", { display: "flex", alignItems: "center", gap: "6px", minHeight: "24px" });
        this.status = el("div", {
            flex: "1 1 auto", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
        });
        this.fitButton = el("button", BUTTON_STYLE, { textContent: "Fit", title: "Show the whole scene" });
        this.clearButton = el("button", BUTTON_STYLE, { textContent: "Clear", title: "Empty the quad and pick the corners again" });
        this.header.append(this.status, this.fitButton, this.clearButton);

        this.frame = el("div", {
            position: "relative", flex: "1 1 auto", minHeight: "120px", background: "#0b0b0b",
            borderRadius: "8px", overflow: "hidden",
        });
        this.canvas = el("canvas", {
            position: "absolute", inset: "0", width: "100%", height: "100%", touchAction: "none", outline: "none",
        }, { tabIndex: 0 });
        this.frame.append(this.canvas);

        this.footer = el("div", {
            display: "flex", gap: "12px", font: "11px monospace", color: "#888", minHeight: "16px", whiteSpace: "nowrap",
            overflow: "hidden",
        });
        this.readout = el("div", { flex: "1 1 auto", overflow: "hidden", textOverflow: "ellipsis" });
        this.cursorText = el("div", { flex: "0 0 auto" });
        this.footer.append(this.readout, this.cursorText);
        this.element.append(this.header, this.frame, this.footer);

        const c = this.canvas;
        c.addEventListener("pointerdown", (e) => this.onPointerDown(e));
        c.addEventListener("pointermove", (e) => this.onPointerMove(e));
        c.addEventListener("pointerup", (e) => this.onPointerUp(e));
        c.addEventListener("pointercancel", (e) => this.onPointerUp(e));
        c.addEventListener("pointerleave", () => {
            if (this.drag) return;
            this.hover = null;
            this.hot = -1;
            this.draw();
        });
        c.addEventListener("wheel", (e) => this.onWheel(e), { passive: false });
        c.addEventListener("keydown", (e) => this.onKeyDown(e));
        c.addEventListener("contextmenu", (e) => e.preventDefault());
        // In case the scene or the quad changed while the graph wasn't redrawn.
        this.element.addEventListener("pointerenter", () => this.refresh());
        this.fitButton.addEventListener("click", () => this.fit());
        this.clearButton.addEventListener("click", () => this.clear());
        new ResizeObserver(() => this.draw()).observe(this.frame);
        this.update();
    }

    // --- the quad widget ------------------------------------------------------

    get quadWidget() { return widget(this.node, "quad"); }

    readQuad() {
        const text = this.quadWidget?.value ?? "";
        this.lastText = text;
        const { points, box, error } = parseQuad(text);
        this.points = points;
        this.box = !!box;
        this.error = error ?? null;
        if (this.selected >= this.points.length) this.selected = -1;
    }

    writeQuad() {
        const w = this.quadWidget;
        if (!w) return;
        const text = this.points.length === 4 ? formatQuad(this.points) : "";
        this.box = false;
        this.error = null;
        this.lastText = text;
        if (w.value === text) return;
        w.value = text;
        this.node.setDirtyCanvas?.(true, true);
        // The editor records undo steps on mouseup, which the picker's pointer handling suppresses.
        app.extensionManager?.workflow?.activeWorkflow?.changeTracker?.checkState?.();
    }

    clear() {
        this.points = [];
        this.selected = -1;
        this.writeQuad();
        this.update();
    }

    // Picks up a new scene, a typed quad or a size change; cheap enough to run on every node redraw.
    refresh() {
        let changed = false;
        if ((this.quadWidget?.value ?? "") !== this.lastText && !this.drag) {
            this.readQuad();
            changed = true;
        }
        const src = sceneSource(this.node);
        if (src.url !== this.sourceURL) {
            this.load(src);
            changed = true;
        } else if (!src.url && src.message !== this.message) {
            this.message = src.message;
            changed = true;
        }
        const zoom = this.cssZoom();
        if (zoom !== this.drawnZoom) changed = true;
        if (changed) this.update();
    }

    load(src) {
        this.sourceURL = src.url ?? null;
        this.sourceLabel = src.label ?? "";
        if (!src.url) {
            this.image = null;
            this.message = src.message;
            return;
        }
        this.message = "Loading the scene…";
        const img = new Image();
        img.onload = () => {
            if (this.sourceURL !== src.url) return;
            const sameSize = this.image && this.image.naturalWidth === img.naturalWidth
                && this.image.naturalHeight === img.naturalHeight;
            this.image = img;
            if (!sameSize) {
                this.fit(false);
                this.fitNode();
            }
            this.update();
        };
        img.onerror = () => {
            if (this.sourceURL !== src.url) return;
            this.image = null;
            this.message = `Can't load ${src.label || "the scene"}`;
            this.update();
        };
        img.src = src.url;
    }

    // --- view -----------------------------------------------------------------

    get W() { return this.frame.clientWidth; }
    get H() { return this.frame.clientHeight; }

    // The scene at the node's width, plus the header and footer rows: the picture gets the room, not the text boxes.
    minHeight() {
        const img = this.image;
        const aspect = img ? img.naturalHeight / img.naturalWidth : 0.75;
        return clamp(Math.round((this.node.size[0] - 20) * aspect) + 52, 280, 640);
    }

    // A new scene's shape may need a taller node.
    fitNode() {
        const min = this.node.computeSize?.()[1];
        if (min > this.node.size[1]) this.node.setSize([this.node.size[0], min]);
    }

    // The draft's report box (image_post.js) arrives under the picker after a run: room to read it.
    fitReport() {
        const report = widget(this.node, "image_post_text");
        if (!report || report.options.getMinHeight) return;
        report.options.getMinHeight = () => REPORT_MIN;
        this.fitNode();
        this.node.setDirtyCanvas?.(true, true);
    }

    // DOM widgets are CSS-scaled with the graph zoom.
    cssZoom() {
        const r = this.canvas.getBoundingClientRect();
        return (this.canvas.clientWidth && r.width / this.canvas.clientWidth) || 1;
    }

    fitScale() {
        const img = this.image;
        if (!img || !this.W || !this.H) return 1;
        return Math.min(this.W / img.naturalWidth, this.H / img.naturalHeight);
    }

    transform() {
        const s = this.fitScale() * this.view.zoom;
        return { s, ox: this.W / 2 - this.view.cx * s, oy: this.H / 2 - this.view.cy * s };
    }

    toScene([x, y]) {
        const { s, ox, oy } = this.transform();
        return [(x - ox) / s, (y - oy) / s];
    }

    toCanvas([x, y]) {
        const { s, ox, oy } = this.transform();
        return [x * s + ox, y * s + oy];
    }

    fit(redraw = true) {
        const img = this.image;
        this.view = { zoom: 1, cx: img ? img.naturalWidth / 2 : 0, cy: img ? img.naturalHeight / 2 : 0 };
        if (redraw) this.draw();
    }

    // Keep part of the scene on screen while panning and zooming.
    clampView() {
        const img = this.image;
        if (!img) return;
        const s = this.fitScale() * this.view.zoom;
        const mx = Math.max(0, img.naturalWidth / 2 - this.W / (2 * s)) + 0.25 * this.W / s;
        const my = Math.max(0, img.naturalHeight / 2 - this.H / (2 * s)) + 0.25 * this.H / s;
        this.view.cx = clamp(this.view.cx, img.naturalWidth / 2 - mx, img.naturalWidth / 2 + mx);
        this.view.cy = clamp(this.view.cy, img.naturalHeight / 2 - my, img.naturalHeight / 2 + my);
    }

    // --- interaction ----------------------------------------------------------

    pointer(e) {
        const r = this.canvas.getBoundingClientRect();
        const z = this.cssZoom();
        return [(e.clientX - r.left) / z, (e.clientY - r.top) / z];
    }

    hitTest(p) {
        const reach = (HANDLE_R + 4) / Math.min(1, this.cssZoom());
        let best = -1, bestD = reach;
        this.points.forEach((q, i) => {
            const [x, y] = this.toCanvas(q);
            const d = Math.hypot(x - p[0], y - p[1]);
            if (d <= bestD) {
                best = i;
                bestD = d;
            }
        });
        return best;
    }

    onPointerDown(e) {
        if (!this.image || (e.button !== 0 && e.button !== 1)) return;
        e.preventDefault();
        e.stopPropagation();
        this.canvas.focus();
        this.refresh();
        const p = this.pointer(e);
        const corner = e.button === 0 && !e.shiftKey ? this.hitTest(p) : -1;
        if (corner >= 0) {
            const [x, y] = this.toCanvas(this.points[corner]);
            this.drag = { kind: "corner", index: corner, grab: [p[0] - x, p[1] - y] };
            this.selected = corner;
        } else if (e.button === 0 && !e.shiftKey && this.points.length < 4) {
            this.points.push(this.toScene(p));
            this.error = null;
            this.selected = this.points.length - 1;
            this.drag = { kind: "corner", index: this.selected, grab: [0, 0] };
        } else {
            this.drag = { kind: "pan", from: p, cx: this.view.cx, cy: this.view.cy };
        }
        this.hot = this.drag.kind === "corner" ? this.drag.index : -1;
        this.canvas.setPointerCapture(e.pointerId);
        this.update();
    }

    onPointerMove(e) {
        if (!this.image) return;
        const p = this.pointer(e);
        this.hover = this.toScene(p);
        if (!this.drag) {
            this.hot = this.hitTest(p);
            this.canvas.style.cursor = this.hot >= 0 ? "grab" : this.points.length < 4 && !e.shiftKey ? "crosshair" : "move";
            this.draw();
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        if (this.drag.kind === "corner") {
            this.canvas.style.cursor = "grabbing";
            this.points[this.drag.index] = this.toScene([p[0] - this.drag.grab[0], p[1] - this.drag.grab[1]]);
            this.hover = this.points[this.drag.index];
        } else {
            const s = this.fitScale() * this.view.zoom;
            this.view.cx = this.drag.cx - (p[0] - this.drag.from[0]) / s;
            this.view.cy = this.drag.cy - (p[1] - this.drag.from[1]) / s;
            this.clampView();
        }
        this.update();
    }

    onPointerUp(e) {
        if (!this.drag) return;
        this.canvas.releasePointerCapture?.(e.pointerId);
        const moved = this.drag.kind === "corner";
        this.drag = null;
        if (moved && this.points.length === 4) this.writeQuad();
        this.update();
    }

    onWheel(e) {
        if (!this.image) return;
        e.preventDefault();
        e.stopPropagation();
        const p = this.pointer(e);
        const before = this.toScene(p);
        const delta = e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
        const maxZoom = Math.max(1, MAX_SCALE / this.fitScale());
        this.view.zoom = clamp(this.view.zoom * Math.exp(-delta * 0.002), 1, maxZoom);
        // Zoom about the pointer: the scene px under it stays put.
        const s = this.fitScale() * this.view.zoom;
        this.view.cx = before[0] - (p[0] - this.W / 2) / s;
        this.view.cy = before[1] - (p[1] - this.H / 2) / s;
        this.clampView();
        this.hover = this.toScene(p);
        this.update();
    }

    onKeyDown(e) {
        const step = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] }[e.key];
        if (step && this.selected >= 0 && this.selected < this.points.length) {
            const k = e.shiftKey ? 10 : 1;
            const q = this.points[this.selected];
            this.points[this.selected] = [Math.round(q[0]) + step[0] * k, Math.round(q[1]) + step[1] * k];
            if (this.points.length === 4) this.writeQuad();
        } else if (e.key === "Escape" || e.key === "Tab") {
            this.selected = -1;
            if (e.key === "Tab") return;
        } else {
            return;
        }
        e.preventDefault();
        e.stopPropagation();
        this.update();
    }

    // --- drawing --------------------------------------------------------------

    update() {
        this.updateText();
        this.draw();
    }

    updateText() {
        const n = this.points.length;
        let text, color = "#ddd";
        if (!this.image) {
            text = this.message;
            color = "#999";
        } else if (this.error) {
            text = `quad: ${this.error}. Clear it to pick the corners`;
            color = WARN;
        } else if (n < 4) {
            text = `Click the ${CORNER_NAMES[n]} corner of the product (${n + 1} of 4)`;
        } else {
            const problem = quadProblem(this.points);
            text = problem ? `Check the quad: ${problem}`
                : this.box ? "A box: drag a corner to make it a quad" : "Drag a corner to adjust · arrows nudge 1 px";
            color = problem ? WARN : "#ddd";
        }
        this.status.textContent = text;
        this.status.style.color = color;
        const img = this.image;
        this.status.title = img ? `${this.sourceLabel} · ${img.naturalWidth}×${img.naturalHeight} · the wheel zooms; `
            + "middle-drag or Shift-drag pans, and so does dragging empty space once the 4 corners are placed" : "";
        this.readout.textContent = n
            ? this.points.map((q, i) => `${CORNERS[i]} ${Math.round(q[0])},${Math.round(q[1])}`).join("  ")
            : img ? `scene ${img.naturalWidth}×${img.naturalHeight}` : "";
        this.readout.style.color = n === 4 ? "#bbb" : "#888";
        const h = this.hover;
        this.cursorText.textContent = img && h
            ? `x ${Math.round(h[0])} y ${Math.round(h[1])}${this.view.zoom > 1.001 ? `  ×${this.view.zoom.toFixed(1)}` : ""}`
            : "";
    }

    draw() {
        const c = this.canvas;
        const W = this.W, H = this.H;
        if (!W || !H) return;
        const zoom = this.cssZoom();
        this.drawnZoom = zoom;
        const res = (window.devicePixelRatio || 1) * clamp(zoom, 0.5, 3);
        if (c.width !== Math.round(W * res) || c.height !== Math.round(H * res)) {
            c.width = Math.round(W * res);
            c.height = Math.round(H * res);
        }
        const ctx = c.getContext("2d");
        ctx.setTransform(res, 0, 0, res, 0, 0);
        ctx.clearRect(0, 0, W, H);
        const img = this.image;
        if (!img) {
            ctx.fillStyle = "#777";
            ctx.font = "12px sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(this.message, W / 2, H / 2, W - 20);
            return;
        }
        const { s, ox, oy } = this.transform();
        ctx.imageSmoothingEnabled = s * zoom < 3;  // show scene pixels when zoomed in
        ctx.drawImage(img, ox, oy, img.naturalWidth * s, img.naturalHeight * s);

        const u = 1 / Math.min(1, zoom);  // line widths and handles stay readable when the graph is zoomed out
        const pts = this.points.map((q) => this.toCanvas(q));
        const n = pts.length;
        const problem = n === 4 && quadProblem(this.points);
        const stroke = (path, color, width, dash = []) => {
            ctx.setLineDash(dash.map((d) => d * u));
            ctx.lineJoin = "round";
            ctx.strokeStyle = "rgba(0,0,0,0.65)";
            ctx.lineWidth = (width + 2) * u;
            ctx.stroke(path);
            ctx.strokeStyle = color;
            ctx.lineWidth = width * u;
            ctx.stroke(path);
            ctx.setLineDash([]);
        };
        if (n) {
            const path = new Path2D();
            pts.forEach(([x, y], i) => (i ? path.lineTo(x, y) : path.moveTo(x, y)));
            if (n === 4) {
                path.closePath();
                ctx.fillStyle = problem ? "rgba(245,158,11,0.12)" : "rgba(34,211,238,0.10)";
                ctx.fill(path);
            }
            stroke(path, problem ? WARN : ACCENT, 1.5);
        }
        // While picking: where the next corner joins the outline.
        if (n && n < 4 && this.hover && !this.drag) {
            const [hx, hy] = this.toCanvas(this.hover);
            const path = new Path2D();
            path.moveTo(...pts[n - 1]);
            path.lineTo(hx, hy);
            if (n === 3) path.lineTo(...pts[0]);
            stroke(path, ACCENT, 1, [4, 4]);
        }
        // Corners: a ring with a dot on the exact point and the corner's name outside the quad.
        const cx = pts.reduce((a, p) => a + p[0], 0) / (n || 1), cy = pts.reduce((a, p) => a + p[1], 0) / (n || 1);
        ctx.font = `bold ${11 * u}px sans-serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        pts.forEach(([x, y], i) => {
            const active = i === this.hot || i === this.selected;
            ctx.beginPath();
            ctx.arc(x, y, HANDLE_R * u, 0, 2 * Math.PI);
            ctx.lineWidth = 3.5 * u;
            ctx.strokeStyle = "rgba(0,0,0,0.65)";
            ctx.stroke();
            ctx.lineWidth = 1.5 * u;
            ctx.strokeStyle = active ? "#fff" : ACCENT;
            ctx.stroke();
            ctx.fillStyle = active ? "#fff" : ACCENT;
            ctx.fillRect(x - 0.75 * u, y - 0.75 * u, 1.5 * u, 1.5 * u);
            let dx = x - cx, dy = y - cy;
            const d = Math.hypot(dx, dy);
            if (n < 2 || d < 1) {
                [dx, dy] = [[-1, -1], [1, -1], [1, 1], [-1, 1]][i];
            } else {
                dx /= d;
                dy /= d;
            }
            const lx = x + dx * 16 * u, ly = y + dy * 16 * u;
            ctx.lineWidth = 3 * u;
            ctx.strokeStyle = "rgba(0,0,0,0.75)";
            ctx.strokeText(CORNERS[i], lx, ly);
            ctx.fillStyle = active ? "#fff" : ACCENT;
            ctx.fillText(CORNERS[i], lx, ly);
        });
        // The next corner's name by the pointer.
        if (n < 4 && this.hover && !this.drag && this.hot < 0) {
            const [hx, hy] = this.toCanvas(this.hover);
            ctx.textAlign = "left";
            ctx.lineWidth = 3 * u;
            ctx.strokeStyle = "rgba(0,0,0,0.75)";
            ctx.strokeText(CORNERS[n], hx + 10 * u, hy - 10 * u);
            ctx.fillStyle = "#fff";
            ctx.fillText(CORNERS[n], hx + 10 * u, hy - 10 * u);
        }
    }
}

app.registerExtension({
    name: "ImagePost.QuadPicker",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const picker = new QuadPicker(this);
            this.quadPicker = picker;
            const quad = widget(this, "quad");
            if (quad) {
                const callback = quad.callback;
                quad.callback = function () {
                    const r = callback?.apply(this, arguments);
                    picker.refresh();
                    return r;
                };
            }
            // Added after every other widget and never serialized: UI workflows store widget values by position.
            const w = this.addDOMWidget("quad_picker", "quad_picker", picker.element, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => picker.minHeight(),
            });
            w.serialize = false;
            this.setSize([Math.max(this.size[0], 460), this.size[1]]);
            picker.fitNode();
            requestAnimationFrame(() => picker.refresh());
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            requestAnimationFrame(() => this.quadPicker?.refresh());
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const r = onConnectionsChange?.apply(this, arguments);
            requestAnimationFrame(() => this.quadPicker?.refresh());
            return r;
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function () {
            const r = onExecuted?.apply(this, arguments);
            requestAnimationFrame(() => this.quadPicker?.fitReport());
            return r;
        };

        // Catches a new file picked on the upstream Load Image and a quad set from elsewhere.
        const onDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function () {
            const r = onDrawForeground?.apply(this, arguments);
            this.quadPicker?.refresh();
            return r;
        };
    },
});
