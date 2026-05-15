import { app } from "../../scripts/app.js";

const NODE_NAME = "BadmanColorMatch";
const COMBINE_NODE_NAME = "BadmanColorMatchCombine";

// Widgets only present in build mode.
const BUILD_WIDGETS = ["method", "save_as", "overwrite"];
// Widgets only present in preset mode.
const PRESET_WIDGETS = ["preset_name"];
// Optional input sockets only present in build mode.
// Order matters: we re-add in this order so the node looks consistent.
const BUILD_INPUTS = [
    { name: "source_ref", type: "IMAGE" },
    { name: "target_ref", type: "IMAGE" },
    { name: "source_mask", type: "MASK" },
    { name: "target_mask", type: "MASK" },
];

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (visible) {
        if (widget._origType !== undefined) widget.type = widget._origType;
        widget.computeSize = widget._origComputeSize || null;
        widget.hidden = false;
    } else {
        if (widget._origType === undefined) widget._origType = widget.type;
        if (!widget._origComputeSize) widget._origComputeSize = widget.computeSize;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4];
        widget.hidden = true;
    }
}

function findInputIndex(node, name) {
    return node.inputs?.findIndex((i) => i.name === name) ?? -1;
}

function removeInputByName(node, name) {
    const idx = findInputIndex(node, name);
    if (idx >= 0) node.removeInput(idx);  // handles link disconnect internally
}

function ensureInputByName(node, name, type) {
    if (findInputIndex(node, name) >= 0) return;
    node.addInput(name, type);
}

function applyMode(node) {
    const toggle = findWidget(node, "use_preset");
    if (!toggle) return;
    const usePreset = !!toggle.value;

    for (const n of BUILD_WIDGETS) setWidgetVisible(findWidget(node, n), !usePreset);
    for (const n of PRESET_WIDGETS) setWidgetVisible(findWidget(node, n), usePreset);

    if (usePreset) {
        for (const { name } of BUILD_INPUTS) removeInputByName(node, name);
    } else {
        for (const { name, type } of BUILD_INPUTS) ensureInputByName(node, name, type);
    }

    node.setSize(node.computeSize());
    node.setDirtyCanvas(true, true);
}

async function refreshPresets(node) {
    try {
        const r = await fetch("/badman/color_match/presets");
        if (!r.ok) return;
        const data = await r.json();
        if (!Array.isArray(data.presets)) return;
        // Update every widget whose name starts with "preset" (preset_name, preset_1..N).
        for (const w of node.widgets || []) {
            if (!w?.name?.startsWith("preset")) continue;
            if (!Array.isArray(w.options?.values)) continue;
            const current = w.value;
            w.options.values = data.presets;
            if (!data.presets.includes(current)) w.value = data.presets[0];
        }
    } catch (e) {
        console.warn("[Badman ColorMatch] preset refresh failed", e);
    }
}

// ---------- live preview ----------

function installPreview(node) {
    const wrap = document.createElement("div");
    wrap.style.cssText =
        "display:flex;flex-direction:column;gap:2px;padding:4px;box-sizing:border-box;";
    const img = document.createElement("img");
    img.style.cssText =
        "width:100%;display:block;background:#222;border-radius:4px;" +
        "image-rendering:auto;object-fit:contain;min-height:80px;";
    const status = document.createElement("div");
    status.style.cssText =
        "font-size:10px;color:#888;min-height:12px;font-family:monospace;";
    status.textContent = "run workflow once to seed preview";
    wrap.appendChild(img);
    wrap.appendChild(status);

    node.addDOMWidget("preview", "preview", wrap, {
        serialize: false,
        getHeight: () => 240,
    });

    node._emtdPreview = { img, status };
}

function schedulePreview(node, delay = 180) {
    clearTimeout(node._emtdPreviewTimer);
    node._emtdPreviewTimer = setTimeout(() => fetchPreview(node), delay);
}

async function fetchPreview(node) {
    const refs = node._emtdPreview;
    if (!refs) return;
    const { img, status } = refs;

    const getVal = (name) => findWidget(node, name)?.value;
    const body = {
        node_id: String(node.id),
        use_preset: !!getVal("use_preset"),
        preset_name: getVal("preset_name"),
        method: getVal("method"),
        strength: getVal("strength"),
    };

    status.textContent = "updating…";
    try {
        const r = await fetch("/badman/color_match/preview", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
        });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) {
            status.textContent = data.error || `error ${r.status}`;
            return;
        }
        img.src = data.image;
        status.textContent = "";
    } catch (e) {
        status.textContent = String(e);
    }
}

function wireLiveWidgets(node) {
    const names = ["use_preset", "preset_name", "method", "strength"];
    for (const n of names) {
        const w = findWidget(node, n);
        if (!w) continue;
        const orig = w.callback;
        w.callback = function (v) {
            orig?.apply(this, arguments);
            schedulePreview(node);
        };
    }
}

app.registerExtension({
    name: "Badman.ColorMatch",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === COMBINE_NODE_NAME) {
            const MIN_SLOTS = 2;
            // Count declared preset_N widgets on the node (matches MAX_SLOTS in py).
            const countPresetWidgets = (node) =>
                (node.widgets || []).filter((w) => /^preset_\d+$/.test(w.name)).length;

            const applyVisibleCount = (node) => {
                const total = countPresetWidgets(node);
                const n = Math.max(
                    MIN_SLOTS,
                    Math.min(total, node.properties.visible_presets ?? MIN_SLOTS),
                );
                node.properties.visible_presets = n;
                for (let i = 1; i <= total; i++) {
                    const w = findWidget(node, `preset_${i}`);
                    setWidgetVisible(w, i <= n);
                }
                node.setSize(node.computeSize());
                node.setDirtyCanvas(true, true);
            };

            const onCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onCreated?.apply(this, arguments);
                this.properties = this.properties || {};
                if (this.properties.visible_presets == null) {
                    this.properties.visible_presets = MIN_SLOTS;
                }

                // Add + / - control buttons placed just below the last preset slot.
                this.addWidget("button", "+ preset slot", null, () => {
                    const total = countPresetWidgets(this);
                    this.properties.visible_presets = Math.min(
                        total,
                        (this.properties.visible_presets ?? MIN_SLOTS) + 1,
                    );
                    applyVisibleCount(this);
                });
                this.addWidget("button", "- preset slot", null, () => {
                    this.properties.visible_presets = Math.max(
                        MIN_SLOTS,
                        (this.properties.visible_presets ?? MIN_SLOTS) - 1,
                    );
                    // Reset hidden slot back to <none> so it doesn't contribute.
                    const n = this.properties.visible_presets;
                    const w = findWidget(this, `preset_${n + 1}`);
                    if (w && w.options?.values?.length) w.value = w.options.values[0];
                    applyVisibleCount(this);
                });

                refreshPresets(this);
                applyVisibleCount(this);
            };

            // Re-apply visibility after deserialization picks up widget values.
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                onConfigure?.apply(this, arguments);
                // If the user had slots with real values, expand to cover them.
                const total = countPresetWidgets(this);
                let lastUsed = MIN_SLOTS;
                for (let i = 1; i <= total; i++) {
                    const w = findWidget(this, `preset_${i}`);
                    if (w && w.value && w.value !== "<none>") lastUsed = i;
                }
                this.properties = this.properties || {};
                this.properties.visible_presets = Math.max(
                    this.properties.visible_presets ?? MIN_SLOTS,
                    lastUsed,
                );
                applyVisibleCount(this);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function () {
                onExecuted?.apply(this, arguments);
                refreshPresets(this);
            };

            // Refresh this node's dropdowns whenever any workflow run completes,
            // so newly-saved presets from the main node show up immediately.
            const api = app.api;
            if (api && !nodeType._emtdCombineHooked) {
                nodeType._emtdCombineHooked = true;
                api.addEventListener("executed", () => {
                    for (const n of app.graph?._nodes || []) {
                        if (n.type === COMBINE_NODE_NAME) refreshPresets(n);
                    }
                });
            }
            return;
        }
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const toggle = findWidget(this, "use_preset");
            if (toggle) {
                const orig = toggle.callback;
                toggle.callback = (v) => {
                    orig?.call(toggle, v);
                    applyMode(this);
                    schedulePreview(this);
                };
            }

            installPreview(this);
            wireLiveWidgets(this);
            refreshPresets(this).then(() => applyMode(this));
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (msg) {
            onExecuted?.apply(this, arguments);
            // Cache is fresh after execute, pull both preset list and preview.
            refreshPresets(this);
            schedulePreview(this, 50);
        };
    },
});
