/**
 * BFS Image Planner: the Shot Planner panel for one still image (MiniMax H3 Image). Everything that only exists for a
 * video (timeline, cuts, shots, continuity, audio, queue loop, stored shots, mask videos, tracking) is left out; the
 * pictures, the prompt helpers, the selection, the generation modes, Colored Mark, {target}, the VLM helpers, the
 * generation size and the second pass are the same.
 *
 * <Picture 1> is the image (the picture H3 Image edits); the references are <Picture 2>, <Picture 3>, ...
 * All state lives in the hidden `plan` STRING widget (JSON), read by bfs_h3_image_planner.py.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { wheelToCanvas } from "./bfs_wheel.js";
import { createApp, ref, reactive, computed, onMounted, onBeforeUnmount, Teleport } from "./vendor/vue.esm-browser.prod.mjs";
import { h, rawH, t, uiLanguage, setUiLanguage, pill, hint, check, section, select, fld, refBox, promptChips, MAX_REFS,
         MARK_PALETTE, pictureColor, markName, markRegions, MASK_DEFAULTS, VLM_DEFAULTS, secondPassSize } from "./shotloop/common.js";
import { styles } from "./shotloop/styles.js";
import { busyBar } from "./shotloop/view_modal.js";

const NODE = "BFSImagePlanner";
const viewUrl = name => {
  if (!name) return "";
  const i = name.lastIndexOf("/");
  const sub = i >= 0 ? name.slice(0, i) : "", file = i >= 0 ? name.slice(i + 1) : name;
  return api.apiURL(`/view?filename=${encodeURIComponent(file)}&subfolder=${encodeURIComponent(sub)}&type=input`);
};
const store = {
  get: (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};
const TABS = [["image", "🖼 Image"], ["prompts", "📝 Pictures & prompt"], ["edit", "✏ Edit"], ["run", "▶ Run"]];
const VARIANTS = ["MiniMax H3 Image", "p-h3-image"];
const PRESETS = ["short", "object", "full body", "head / face", "face attributes", "outfit", "scene / setting", "custom"];
const AUTO_DEFAULTS = { enabled: false, style: "structured", prefix: "" };
const AUTO_REVISION = "image-v1";
const DEFAULTS = {
  image: "", variant: VARIANTS[0], global_refs: [], global_prompt: "", picture_colors: [],
  generation_size_mode: "native", megapixels: 4.0, multiple: 32, custom_width: 2048, custom_height: 2048,
  mask: {}, mask_cfg: {}, crop: false, inpaint: false, paste: false, paste_text: "", strength: 1.0,
  color_guide: false, color_guide_picture: 2, color_guide_style: "overlay", color_guide_opacity: 0.35, color_guide_regions: [],
  target: "", vlm_cfg: {}, picture_details: {}, picture_cfg: {}, auto_prompt: {}, auto_written: {}, image_note: "",
  second_pass: { enabled: false, mode: "scale", scale: 2, width: 2048, height: 2048, megapixels: 8, align: 32 },
};
// [key, name, crop, inpaint, paste] (the same generation modes as the Shot Planner)
const MODES = [["frame", "Full frame", false, false, false], ["paste", "Frame + paste", false, false, true], ["mask", "Mask only", false, true, false],
               ["crop", "Crop", true, false, false], ["cropmask", "Crop + mask", true, true, false]];
const MODE_INFO = {
  frame: "The whole image is regenerated.",
  paste: "The whole image is generated; BFS Image Planner Paste Back pastes only the subject onto the original (background exact).",
  mask: "Only the selection is regenerated; the rest of the image stays exactly as it was.",
  crop: "A box around the selection is regenerated at full size (more detail for a small subject), then pasted back.",
  cropmask: "Inside the crop, only the selection is regenerated: the most detail, background kept.",
};
const modeOf = p => (MODES.find(m => m[2] === !!p.crop && m[3] === !!p.inpaint && m[4] === (!!p.paste && !p.crop && !p.inpaint)) || MODES[0])[0];
const modeName = p => MODES.find(m => m[0] === modeOf(p))[1];
const grid = (x, m) => Math.max(m, Math.round(x / m) * m);
// the canvas, the same rules as bfs_h3_image_planner.generation_size
function genSize(w, h, p) {
  if (!w || !h) return { w: 0, h: 0 };
  const mode = p.generation_size_mode || "native", mult = Math.max(32, Math.floor((+p.multiple || 32) / 32) * 32);
  if (mode === "custom") return { w: grid(+p.custom_width || 2048, mult), h: grid(+p.custom_height || 2048, mult) };
  if (mode === "megapixels") { const s = Math.sqrt((+p.megapixels || 4) * 1024 * 1024 / (w * h)); return { w: grid(w * s, mult), h: grid(h * s, mult) }; }
  if (mode === "original") return { w: grid(w, mult), h: grid(h, mult) };
  const s = p.variant === VARIANTS[1] ? Math.min(1, Math.sqrt(2912 * 2912 / (w * h))) : Math.min(Math.max(Math.min(w, h), 1024), 2048) / Math.min(w, h);
  return { w: grid(w * s, 32), h: grid(h * s, 32) };
}

function Panel(io) {
  const plan = reactive({ ...DEFAULTS });
  const files = reactive({ images: [], vlm: false });
  const src = reactive({ w: 0, h: 0 });
  const busy = ref(""), error = ref("");
  const status = reactive({ stage: "", done: 0, total: 0 });
  const busySince = ref(0), now = ref(Date.now()), upPct = ref(-1);
  const savedTab = store.get("bfs.imageplanner.tab", "image");
  const tab = ref(TABS.some(([k]) => k === savedTab) ? savedTab : "image");
  const maskPrev = ref(null);      // {src, coverage, empty, box}
  const targetCrop = ref("");
  const sug = ref(null);           // {segment, image} from the VLM
  const modal = reactive({ open: false, region: 1, groups: {}, points: [], text: "", prev: null, busy: "" });

  const load = () => {
    let p = {};
    try { p = JSON.parse(io.getPlan() || "{}"); } catch { p = {}; }
    Object.assign(plan, JSON.parse(JSON.stringify(DEFAULTS)), p);
    plan.mask_cfg = { ...MASK_DEFAULTS, ...(p.mask_cfg || {}) };
    plan.vlm_cfg = { ...VLM_DEFAULTS, ...(p.vlm_cfg || {}) };
    plan.auto_prompt = { ...AUTO_DEFAULTS, ...(p.auto_prompt || {}) };
    plan.second_pass = { ...DEFAULTS.second_pass, ...(p.second_pass || {}) };
    if (!Array.isArray(plan.global_refs)) plan.global_refs = [];
    plan.color_guide_regions = markRegions(plan);
  };
  const save = () => {
    while (plan.global_refs.length && !plan.global_refs[plan.global_refs.length - 1]) plan.global_refs.pop();
    io.setPlan(JSON.stringify(plan));
  };
  const setPlan = (k, v) => { plan[k] = v; save(); };
  const setMask = (k, v) => { if ((plan.mask || {})[k] === v) return; plan.mask = { ...(plan.mask || {}), [k]: v }; maskPrev.value = null; save(); };
  const setMaskCfg = (k, v) => { plan.mask_cfg = { ...plan.mask_cfg, [k]: v }; maskPrev.value = null; save(); };
  const setVlmCfg = (k, v) => { plan.vlm_cfg = { ...plan.vlm_cfg, [k]: v }; save(); };
  const setAuto = (k, v) => { plan.auto_prompt = { ...plan.auto_prompt, [k]: v }; save(); };
  const setMode = mode => { const m = MODES.find(x => x[0] === mode); Object.assign(plan, { crop: m[2], inpaint: m[3], paste: m[4] }); save(); };
  const refs = computed(() => plan.global_refs.filter(Boolean));
  const names = computed(() => [plan.image || "<image>", ...refs.value]);       // <Picture n> order
  const size = computed(() => genSize(src.w, src.h, plan));
  const hasSel = computed(() => !!(String(plan.mask?.text || "").trim() || (plan.mask?.points || []).length));
  const promptText = computed(() => plan.global_prompt || "");

  // ---- server
  const call = async (path, body, label) => {
    if (label) { busy.value = label; busySince.value = Date.now(); Object.assign(status, { stage: "", done: 0, total: 0 }); }
    error.value = "";
    try {
      const r = await api.fetchApi(`/bfs/imageplanner/${path}`, { method: "POST", body: JSON.stringify(body) });
      const j = await r.json(); if (j.error) throw new Error(j.error);
      return j;
    } catch (e) { error.value = String(e.message || e); return null; } finally { if (label) busy.value = ""; }
  };
  async function refreshFiles() {
    try { const r = await api.fetchApi("/bfs/imageplanner/files"); const j = await r.json(); files.images = j.images || []; files.vlm = !!j.vlm; } catch { /* optional */ }
  }
  const previewMask = async (override = null, intoModal = false) => {
    const label = "Segmenting with SAM 3… (the first time downloads/loads the model)";
    if (intoModal) modal.busy = label;
    const j = await call("mask", { plan: { ...plan }, mask: override }, intoModal ? "" : label);
    if (intoModal) { modal.busy = ""; modal.prev = j; } else if (j) maskPrev.value = j;
  };
  const describeTarget = async () => {
    const j = await call("target", { plan: { ...plan } }, "Selecting the subject (SAM 3) and describing it (VLM)…");
    if (!j) return;
    targetCrop.value = j.crop;
    if (j.desc) setPlan("target", j.desc);
    else error.value = "The SAM 3 cutout was made (shown), but no VLM is loaded to describe it. Connect Qwen3-VL to the planner's vlm input and queue once; or type the description yourself.";
  };
  const analyseVLM = async () => { const j = await call("vlm", { plan: { ...plan } }, "Asking the VLM about the image…"); if (j) sug.value = j; };
  const describePictures = async (list = names.value) => {
    const j = await call("describe", { plan: { ...plan }, pictures: list }, "Describing the pictures with the VLM…");
    if (j) { plan.picture_details = { ...plan.picture_details, ...j.pictures }; save(); }
  };
  // 🤖 Auto prompt: what the written prompt was made from (the same string as bfs_h3_image_planner.auto_for)
  const autoFor = () => {
    const marks = plan.color_guide ? [[+plan.color_guide_picture || 1, pictureColor(plan, +plan.color_guide_picture || 1)],
      ...markRegions(plan).filter(r => String(r.text || "").trim() || (r.points || []).length).map(r => [Math.max(1, +r.picture || 2), pictureColor(plan, Math.max(1, +r.picture || 2))])] : [];
    if (marks.length && (plan.color_guide_style || "overlay") !== "overlay") marks.push(plan.color_guide_style);
    const A = { ...AUTO_DEFAULTS, ...plan.auto_prompt };
    return JSON.stringify([plan.global_prompt || "", names.value, marks, modeOf(plan), String(plan.mask?.text || ""), plan.target || "",
      A.style, A.prefix, AUTO_REVISION]);
  };
  const writePrompt = async () => {
    const j = await call("autoprompt", { plan: { ...plan } }, "Writing the prompt with the VLM…");
    if (j) { plan.auto_written = { text: j.text, for: j.for }; save(); }
  };

  // ---- uploads and pictures
  async function upload(file) {
    busy.value = `Uploading ${file.name}…`; upPct.value = 0;
    try {
      const fd = new FormData(); fd.append("image", file); fd.append("type", "input"); fd.append("overwrite", "true");
      const r = await api.fetchApi("/upload/image", { method: "POST", body: fd });
      const j = await r.json();
      await refreshFiles();
      return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
    } catch (e) { error.value = `Upload failed: ${e}`; return ""; } finally { busy.value = ""; upPct.value = -1; }
  }
  const pick = (multiple, cb) => {
    const inp = document.createElement("input"); inp.type = "file"; inp.accept = "image/*"; inp.multiple = multiple;
    inp.onchange = async () => { const out = []; for (const f of inp.files) { const n = await upload(f); if (n) out.push(n); } if (out.length) cb(out); };
    inp.click();
  };
  const RECENT_KEY = "bfs.shotloop.recentRefs";   // shared with the Shot Planner
  const recent = ref(store.get(RECENT_KEY, []).filter(x => typeof x === "string"));
  const remember = name => { if (!name) return; recent.value = [name, ...recent.value.filter(x => x !== name)].slice(0, 10); store.set(RECENT_KEY, recent.value); };
  const putRefs = (k, list) => {
    list.slice(0, Math.max(0, MAX_REFS - 1 - k)).forEach((name, j) => { while (plan.global_refs.length < k + j) plan.global_refs.push(""); plan.global_refs[k + j] = name; remember(name); });
    save();
  };
  const setImage = name => { plan.image = name; maskPrev.value = null; sug.value = null; targetCrop.value = ""; plan.image_note = ""; save(); };
  // what refBox calls (target is always "global": one image has one set of pictures)
  const refApi = {
    viewUrl,
    setRef: (_, k, name) => { if (name) { while (plan.global_refs.length < k) plan.global_refs.push(""); plan.global_refs[k] = name; remember(name); } else plan.global_refs.splice(k, 1); save(); },
    uploadRef: (_, k, many = false) => pick(many, list => putRefs(k, list)),
    dropRefs: async (_, k, e) => {
      const at = k ?? plan.global_refs.length;
      const name = e.dataTransfer?.getData("text/bfs-ref");
      if (name) { putRefs(at, [name]); return; }
      const out = [];
      for (const f of [...(e.dataTransfer?.files || [])].filter(f => f.type.startsWith("image/"))) { const n = await upload(f); if (n) out.push(n); }
      if (out.length) putRefs(at, out);
    },
  };
  const setPictureColor = (k, color) => {
    const list = [...(plan.picture_colors || [])];
    while (list.length < k) list.push(pictureColor(plan, list.length + 1));
    list[k - 1] = color; plan.picture_colors = list; save();
  };

  // ---- Colored Mark regions: 1 = the selection (plan.mask), 2.. = color_guide_regions
  const setRegion = (j, k, v) => { plan.color_guide_regions = markRegions(plan).map((r, q) => (q === j ? { ...r, [k]: v } : r)); save(); };
  const addRegion = () => {
    const used = new Set([+plan.color_guide_picture || 2, ...markRegions(plan).map(r => r.picture)]);
    let k = 2; while (used.has(k) && k < Math.max(2, names.value.length)) k++;
    plan.color_guide_regions = [...markRegions(plan), { picture: k, text: "", points: [], key: 0 }]; save();
  };
  const removeRegion = j => { plan.color_guide_regions = markRegions(plan).filter((_, q) => q !== j); save(); };
  const setMarkStyle = style => { Object.assign(plan, { color_guide_style: style, color_guide_opacity: style === "overlay" ? 0.35 : 1 }); save(); if (maskPrev.value) previewMask(); };
  const clearMask = () => { Object.assign(plan, { mask: {}, color_guide_regions: [], crop: false, inpaint: false, paste: false }); maskPrev.value = null; save(); };

  // ---- points window
  const openPoints = (region = 1) => {
    const groups = { 1: { points: [...(plan.mask?.points || [])], text: plan.mask?.text || "" } };
    if (plan.color_guide) markRegions(plan).forEach((r, j) => { groups[j + 2] = { points: [...(r.points || [])], text: r.text || "" }; });
    const g = groups[region] || groups[1];
    Object.assign(modal, { open: true, region: groups[region] ? region : 1, groups, points: [...g.points], text: g.text, prev: null, busy: "" });
  };
  const switchRegion = r => {
    modal.groups[modal.region] = { points: [...modal.points], text: modal.text };
    const g = modal.groups[r] || { points: [], text: "" };
    Object.assign(modal, { region: r, points: [...g.points], text: g.text, prev: null });
  };
  const savePoints = () => {
    modal.groups[modal.region] = { points: [...modal.points], text: modal.text };
    const g1 = modal.groups[1];
    plan.mask = { ...(plan.mask || {}), points: [...g1.points], text: g1.text, key: 0 };
    if (plan.color_guide) plan.color_guide_regions = markRegions(plan).map((r, j) => { const g = modal.groups[j + 2]; return g ? { ...r, points: [...g.points], text: g.text } : r; });
    maskPrev.value = modal.region === 1 ? modal.prev : null; save(); modal.open = false;
  };

  // ---- checks
  const detailsMissing = computed(() => {
    const p = promptText.value, want = new Set();
    if (p.includes("{details}")) want.add(2);
    if (p.includes("{details2}")) want.add(3);
    if (p.includes("{image}")) want.add(1);
    for (const m of p.matchAll(/\{details@Picture\s*(\d+)\}/g)) want.add(+m[1]);
    return [...want].some(k => names.value[k - 1] && !plan.picture_details?.[names.value[k - 1]]);
  });
  const checks = computed(() => {
    const out = [], add = (lvl, text) => out.push({ lvl, text }), p = promptText.value;
    if (!plan.image) add("warn", "no image: pick one in the Image tab (or connect the planner's image input)");
    if ((plan.crop || plan.inpaint || plan.paste) && !hasSel.value) add("warn", `${modeName(plan)} needs a selection (text or points, Edit tab; or the planner's mask input)`);
    if (plan.color_guide && !hasSel.value) add("warn", "Colored Mark needs a selection");
    if (plan.color_guide && !refs.value.length) add("warn", "Colored Mark: add the reference picture(s) the marked subjects become");
    if (plan.color_guide && plan.crop && markRegions(plan).some(r => String(r.text || "").trim() || (r.points || []).length))
      add("warn", "Colored Mark: more than one marked subject needs Crop off");
    if (maskPrev.value?.empty) add("warn", "the mask preview found nothing");
    if (/\{target\}/.test(p) && !plan.target) add(files.vlm ? "info" : "warn", "the prompt uses {target} but there is no target description" + (files.vlm ? " (the VLM writes it at run time)" : ""));
    if (detailsMissing.value) add("info", "the prompt uses {details} but its pictures have no description yet");
    if (/\{details2?\}/.test(p) && (plan.vlm_cfg?.describe_preset || "short") !== "short")
      add("warn", "{details} sits inside a sentence of the prompt: use the 'short' describe preset, the other presets write a paragraph");
    const A = plan.auto_prompt || {};
    if (A.enabled && plan.auto_written?.text && plan.auto_written.for !== autoFor()) add("warn", "the written prompt is older than the request, pictures or marks: 🤖 Write again");
    if (!p && !(A.enabled && plan.auto_written?.text)) add("info", "no prompt (fine when the node's prompt input is connected)");
    if (!refs.value.length) add("info", "no reference picture (an edit from the prompt alone)");
    if (names.value.length > MAX_REFS) add("warn", `${names.value.length} pictures: H3 Image takes up to ${MAX_REFS}`);
    return out;
  });

  // pictures the server filled while running (written prompt, descriptions, target, image note): kept in the plan
  const onFilled = e => {
    const d = e.detail || {}; if ((d.image || "") !== (plan.image || "")) return;
    if (d.auto_written) plan.auto_written = d.auto_written;
    if (d.picture_details) plan.picture_details = { ...plan.picture_details, ...d.picture_details };
    if (d.target && !plan.target) plan.target = d.target;
    if (d.image_note) plan.image_note = d.image_note;
    save();
  };
  const onStatus = e => { if (busy.value || modal.busy) Object.assign(status, e.detail || {}); };
  let clock = 0;
  onMounted(() => {
    load(); refreshFiles();
    api.addEventListener("bfs-imageplanner-filled", onFilled); api.addEventListener("bfs-shotloop-status", onStatus);
    clock = setInterval(() => { if (busy.value || modal.busy) now.value = Date.now(); }, 500);
  });
  onBeforeUnmount(() => { api.removeEventListener("bfs-imageplanner-filled", onFilled); api.removeEventListener("bfs-shotloop-status", onStatus); clearInterval(clock); });
  io.expose({ reload: () => { load(); maskPrev.value = null; } });

  const c = { busySince, now, upPct, status };

  // ---------------------------------------------------------------- views
  function imageTab() {
    const source = section("Image (<Picture 1>)", [
      h("div", { class: "row" }, [
        h("div", { style: "flex:1;min-width:180px" }, [h("select", { value: plan.image, onChange: e => setImage(e.target.value) },
          [h("option", { value: "" }, "— choose an image from the input folder —"), ...files.images.map(v => h("option", { value: v }, v))])]),
        h("button", { class: plan.image ? "" : "pri", onClick: () => pick(false, l => setImage(l[0])) }, "⬆ Upload"),
        h("button", { onClick: refreshFiles, title: "refresh the input folder" }, "↻"),
      ]),
      plan.image ? h("div", { class: "player", style: "margin-top:6px",
        onDragover: e => e.preventDefault(),
        onDrop: async e => { e.preventDefault(); const f = [...(e.dataTransfer?.files || [])].find(x => x.type.startsWith("image/")); if (f) { const n = await upload(f); if (n) setImage(n); } } }, [
        h("img", { src: viewUrl(plan.image), style: "max-height:320px;max-width:100%;border-radius:6px",
          onLoad: e => { src.w = e.target.naturalWidth; src.h = e.target.naturalHeight; } }),
        h("div", { class: "pinfo" }, [
          src.w ? h("div", { class: "row" }, [pill(`${src.w}×${src.h}`), pill(`generate at ${size.value.w}×${size.value.h}`, "acc")]) : null,
          hint("This image is <Picture 1>, the picture MiniMax H3 Image edits. Drop another image here to replace it. A connected image input wins over it."),
        ]),
      ]) : h("div", { class: "empty" }, [h("b", "Pick, upload or drop an image to start"),
        "It gets its own reference pictures, prompt and selection; the conditioning node turns them into MiniMax H3 Image conditioning."]),
    ]);
    const sm = plan.generation_size_mode || "native";
    const fields = [
      fld("Model", select(plan.variant, VARIANTS, v => setPlan("variant", v)), "sets the native canvas"),
      fld("Size mode", select(sm, [["native", "Native (what the model's node picks)"], ["megapixels", "Megapixel target"], ["original", "Original image"], ["custom", "Custom dimensions"]],
        v => setPlan("generation_size_mode", v))),
    ];
    if (sm === "megapixels") fields.push(fld("Megapixels", h("input", { type: "number", step: 0.1, min: 0.1, value: plan.megapixels,
      onChange: e => setPlan("megapixels", parseFloat(e.target.value) || 4) })));
    if (sm === "custom") for (const [k, l] of [["custom_width", "Custom width"], ["custom_height", "Custom height"]])
      fields.push(fld(l, h("input", { type: "number", step: 32, min: 32, value: plan[k], onChange: e => setPlan(k, Math.max(32, parseInt(e.target.value) || 2048)) })));
    if (sm !== "native") fields.push(fld("Size multiple", h("input", { type: "number", step: 32, min: 32, value: plan.multiple,
      onChange: e => setPlan("multiple", Math.max(32, parseInt(e.target.value) || 32)) })));
    fields.push(fld("Output size", size.value.w ? `${size.value.w}×${size.value.h}` : "—", plan.crop ? "a crop uses this rule on the crop" : ""));
    return h("div", { class: "card" }, [source, plan.image ? section("Generation size", h("div", { class: "grid" }, fields),
      { sub: "native: H3 Image keeps the short edge between 1024 and 2048" }) : null]);
  }

  function promptsTab() {
    const VC = plan.vlm_cfg || {}, A = plan.auto_prompt || {};
    const marker = k => h("label", { class: "guidecolor" }, [h("span", { class: "hint" }, "Mark color"),
      h("select", { value: pictureColor(plan, k + 2), title: `Colored Mark color of <Picture ${k + 2}>`, onChange: e => setPictureColor(k + 2, e.target.value) },
        MARK_PALETTE.map(([v, n]) => h("option", { value: v }, n)))]);
    const urlOf = (name, j) => (j === 0 ? viewUrl(plan.image) : viewUrl(name));
    const pics = section("Pictures & prompt", [
      h("div", { class: "row", style: "align-items:flex-start;gap:8px" }, [
        h("div", { class: "rslot" }, [h("div", { class: "refslot", title: "<Picture 1>: the image (Image tab)", onClick: () => { tab.value = "image"; } },
          plan.image ? [h("img", { src: viewUrl(plan.image) }), h("span", { class: "rnum" }, "1")] : ["🖼 1"]), h("div", { class: "rbtns" }, [hint("image")])]),
        h("div", { style: "flex:1;min-width:0" }, [refBox(refApi, "global", plan.global_refs, [], marker, 1)]),
      ]),
      h("div", { class: "pbox", style: "margin-top:6px" }, [
        h("textarea", { style: "min-height:78px", value: plan.global_prompt,
          placeholder: A.enabled ? "Request (🤖 Auto prompt writes the H3 Image prompt from it), e.g. 'replace the woman with the character of Picture 2'"
            : "Prompt. <Picture 1> is the image, <Picture 2>... the references. Placeholders: {target} {details} {details2} {details@Picture n} {image} {shot}",
          onChange: e => setPlan("global_prompt", e.target.value) }),
        promptChips(refApi, names.value, v => setPlan("global_prompt", v), 0, urlOf),
      ]),
      h("div", { class: "hint", style: "margin-top:4px" }, ["Placeholders: ", h("b", "{target}"), " what changes (Edit tab) · ", h("b", "{details}"), " / ",
        h("b", "{details2}"), " the first / second reference's description · ", h("b", "{details@Picture n}"), " any picture's · ",
        h("b", "{image}"), " the image's · ", h("b", "{shot}"), " the VLM's description of the image. A connected prompt input overrides this prompt."]),
      recent.value.length ? h("div", { class: "recent" }, [hint("recent (click = add, or drag onto a slot):"),
        ...recent.value.map(n => h("img", { src: viewUrl(n), title: n, draggable: true, onDragstart: e => e.dataTransfer.setData("text/bfs-ref", n),
          onClick: () => putRefs(plan.global_refs.length, [n]) })),
        h("button", { class: "sm", onClick: () => { recent.value = []; store.set(RECENT_KEY, []); } }, "✕ clear")]) : null,
    ], { sub: "<Picture 1> = the image, the references follow" });
    const w = plan.auto_written || {}, old = w.text && w.for !== autoFor();
    const auto = section("🤖 Auto prompt (H3 Image)", [
      h("div", { class: "row", style: "gap:12px;align-items:center" }, [
        check(A.enabled, v => setAuto("enabled", v), "Write the prompt with the VLM",
          "On: the prompt box holds your request; the VLM writes the MiniMax H3 Image prompt from the image, the pictures and their descriptions, the request, the Colored Marks, the selection and the mode."),
        fld("Style", select(A.style || "structured", [["structured", "Structured (subject_definitions / summary / retention)"], ["short", "Short edit request"]],
          v => setAuto("style", v), { style: "width:auto" })),
        fld("Prefix", h("input", { type: "text", value: A.prefix || "", style: "width:150px", placeholder: "e.g. character_swap:",
          title: "Put in front of the written prompt: a LoRA's trigger. Empty = none.", onChange: e => setAuto("prefix", e.target.value) })),
        h("button", { class: w.text ? "" : "pri", disabled: !!busy.value || !plan.image, onClick: writePrompt }, w.text ? "🤖 Write again" : "🤖 Write"),
        w.text ? h("button", { class: "dng", title: "forget it: the next run writes a new one", onClick: () => setPlan("auto_written", {}) }, "✕") : null,
      ]),
      A.enabled ? h("div", { class: "note", style: "margin-top:6px;padding:6px 8px" }, [
        h("div", { class: "row", style: "gap:6px;margin-bottom:4px" }, [h("b", "Written prompt"), old ? pill("older than the request / pictures / marks", "warn") : null]),
        h("textarea", { style: "min-height:140px;font-size:11px", value: w.text || "",
          placeholder: "not written yet: 🤖 Write, or the planner's VLM writes it when the workflow runs",
          onChange: e => { plan.auto_written = { text: e.target.value, for: plan.auto_written?.for ?? autoFor() }; save(); } }),
      ]) : hint("Off: the prompt box is the prompt.", "display:block;margin-top:4px"),
    ], { sub: "request + image + pictures -> MiniMax H3 Image prompt", right: A.enabled ? pill("on", "ok") : pill("off") });
    const vlm = section("VLM", [
      h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
        check(VC.auto_segment, v => setVlmCfg("auto_segment", v), "Suggest the selection text"),
        check(VC.auto_shot !== false, v => setVlmCfg("auto_shot", v), "Fill {shot} in the prompt", "Write {shot}: it becomes the VLM's description of the image (framing, setting, light)"),
      ]),
      h("div", { class: "grid" }, [
        fld("Max tokens", h("input", { type: "number", min: 64, max: 4096, step: 32, value: VC.max_tokens, onChange: e => setVlmCfg("max_tokens", parseInt(e.target.value) || 1024) })),
        fld("Image max side (VLM)", select(String(VC.image_max_side || 1024), [448, 672, 896, 1024, 1280, 1536, 2048].map(n => [String(n), `${n} px`]),
          v => setVlmCfg("image_max_side", parseInt(v) || 1024))),
      ]),
      h("textarea", { style: "margin-top:6px;min-height:44px", placeholder: "Extra instruction for the VLM (optional), e.g. 'segment the woman, not the man'",
        value: VC.instruction || "", onChange: e => setVlmCfg("instruction", e.target.value) }),
      h("div", { class: "row", style: "margin-top:6px" }, [
        h("button", { class: "pri", disabled: !!busy.value || !plan.image, onClick: analyseVLM }, "🤖 Analyse image"),
        sug.value?.segment ? h("button", { onClick: () => setMask("text", sug.value.segment) }, `Use selection: ${sug.value.segment}`) : null,
        sug.value?.image ? h("button", { onClick: () => setPlan("image_note", sug.value.image), title: sug.value.image }, "Use as {shot}") : null,
      ]),
      h("textarea", { style: "margin-top:6px;min-height:40px", placeholder: "{shot}: the image's description (empty = the VLM writes it at run time)",
        value: plan.image_note || "", onChange: e => setPlan("image_note", e.target.value) }),
      hint("Connect a Qwen3-VL to the planner's vlm input and run the workflow once: then these buttons work.", "display:block;margin-top:4px"),
    ], { open: false, sub: "suggests the selection and a description of the image", right: files.vlm ? pill("connected", "ok") : pill("not connected") });
    const desc = section("Describe pictures ({details})", [
      h("div", { class: "row", style: "gap:8px" }, [
        fld("Describe", select(VC.describe_preset, PRESETS, v => setVlmCfg("describe_preset", v), { style: "width:auto" })),
        h("button", { class: "pri", disabled: !!busy.value || !plan.image, onClick: () => describePictures() }, "📝 Describe pictures"),
        h("button", { disabled: !!busy.value || !plan.image, title: "a new seed (and some temperature if it is 0), then describe again",
          onClick: () => { if (!(+VC.temperature > 0)) setVlmCfg("temperature", 0.7); setVlmCfg("seed", (+VC.seed || 0) + 1); describePictures(); } }, "🎲"),
        fld("Temperature", h("input", { type: "number", min: 0, max: 1.5, step: 0.1, value: VC.temperature ?? 0, style: "width:70px",
          onChange: e => setVlmCfg("temperature", Math.max(0, parseFloat(e.target.value) || 0)) })),
        fld("Seed", h("input", { type: "number", min: 0, step: 1, value: VC.seed ?? 0, style: "width:80px", onChange: e => setVlmCfg("seed", Math.max(0, parseInt(e.target.value) || 0)) })),
        check(VC.fit_prompt !== false, v => setVlmCfg("fit_prompt", v), "fit the prompt", "The VLM writes each description for the prompt's sentence about that picture"),
      ]),
      VC.describe_preset === "custom" ? h("textarea", { style: "margin-top:6px;min-height:52px", value: VC.describe_custom || "",
        placeholder: "Your instruction for the VLM", onChange: e => setVlmCfg("describe_custom", e.target.value) }) : null,
      h("div", { class: "dlist" }, names.value.map((name, k) => {
        const own = plan.picture_cfg?.[name] || {};
        return h("div", { class: "drow" }, [
          h("div", { class: "dthumbs" }, [h("img", { src: k === 0 ? viewUrl(plan.image) : viewUrl(name) })]),
          h("div", { style: "flex:1" }, [
            h("div", { class: "row", style: "gap:6px;margin-bottom:3px" }, [hint(k === 0 ? "<Picture 1> · the image" : `<Picture ${k + 1}>`), h("span", { class: "grow" }),
              select(own.preset || "", [["", `default (${VC.describe_preset || "short"})`], ...PRESETS],
                v => { plan.picture_cfg = { ...plan.picture_cfg, [name]: { ...own, preset: v } }; save(); }, { style: "width:auto" }),
              h("button", { class: "sm", disabled: !!busy.value, onClick: () => describePictures([name]) }, "🔄")]),
            own.preset === "custom" ? h("textarea", { style: "min-height:36px;margin-bottom:3px", value: own.custom || "",
              onChange: e => { plan.picture_cfg = { ...plan.picture_cfg, [name]: { ...own, custom: e.target.value } }; save(); } }) : null,
            h("textarea", { style: "min-height:44px", value: plan.picture_details?.[name] || "",
              placeholder: "no description yet: 📝 Describe, 🔄, write your own, or leave it to the VLM at run time",
              onChange: e => { plan.picture_details = { ...plan.picture_details, [name]: e.target.value }; save(); } }),
          ]),
        ]);
      })),
    ], { open: false, sub: "the text of {details} / {details@Picture n} / {image}" });
    return h("div", { class: "card" }, [pics, auto, vlm, desc]);
  }

  function editTab() {
    if (!plan.image) return h("div", { class: "card empty" }, [h("b", "No image yet"), "Pick an image in the Image tab."]);
    const cg = !!plan.color_guide, regions = markRegions(plan), mp = maskPrev.value;
    const picturePick = (k, onPick) => {
      const n = Math.max(names.value.length, k, 2), color = pictureColor(plan, k), name = names.value[k - 1];
      return h("span", { class: "row", style: "gap:4px;align-items:center" }, [
        name && k > 1 ? h("img", { class: "mthumb", src: viewUrl(name), style: `--mark:${color}` }) : h("span", { class: "mthumb", style: `--mark:${color};display:inline-block;background:${color}` }),
        h("select", { value: String(k), style: "width:auto", title: "The picture this subject becomes; the mark is drawn in its color", onChange: e => onPick(+e.target.value) },
          Array.from({ length: Math.min(MAX_REFS, n) - 1 }, (_, q) => h("option", { value: String(q + 2) },
            `Picture ${q + 2} · ${markName(pictureColor(plan, q + 2))}${names.value[q + 1] ? "" : " (no picture)"}`))),
      ]);
    };
    const label = n => (cg ? h("span", { class: "hint", style: "white-space:nowrap;min-width:58px" }, `Subject ${n}`) : null);
    const becomes = (k, onPick) => (cg ? [h("span", { class: "hint" }, "→"), picturePick(k, onPick)] : []);
    const subject1 = h("div", { class: "mreg" }, [label(1),
      h("input", { type: "text", value: plan.mask?.text || "", style: "flex:1;min-width:160px",
        placeholder: "what to select, in English: woman in pink top, red car, black dog… (Enter = preview)",
        title: "A short phrase (noun + one or two traits); SAM 3 finds it. Points, when set, win over the text.",
        onInput: e => setMask("text", e.target.value),
        onKeydown: e => { if (e.key === "Enter") { e.preventDefault(); setMask("text", e.target.value); if (hasSel.value) previewMask(); } } }),
      h("button", { onClick: () => openPoints(1) }, (plan.mask?.points || []).length ? `🎯 Points (${plan.mask.points.length})` : "🎯 Points…"),
      h("button", { class: hasSel.value && !mp ? "pri" : "", disabled: !!busy.value || !hasSel.value, onClick: () => previewMask() }, "👁 Preview"),
      ...becomes(+plan.color_guide_picture || 2, k => { setPlan("color_guide_picture", k); if (mp) previewMask(); }),
      h("button", { class: "dng", disabled: !(hasSel.value || plan.crop || plan.inpaint || plan.paste), title: "Clear the selection and the mode", onClick: clearMask }, "✕"),
    ]);
    const more = cg ? regions.map((r, j) => h("div", { class: "mreg" }, [label(j + 2),
      h("input", { type: "text", value: r.text || "", style: "flex:1;min-width:160px", placeholder: "what to select, e.g. the man on the left",
        onChange: e => setRegion(j, "text", e.target.value) }),
      h("button", { onClick: () => openPoints(j + 2) }, (r.points || []).length ? `🎯 Points (${r.points.length})` : "🎯 Points…"),
      h("button", { disabled: !!busy.value || !(r.text || (r.points || []).length),
        onClick: () => previewMask({ text: r.text || "", points: r.points || [], max_objects: 1, preview_color: pictureColor(plan, r.picture) }) }, "👁 Preview"),
      ...becomes(r.picture, k => setRegion(j, "picture", k)),
      h("button", { class: "dng", onClick: () => removeRegion(j) }, "✕"),
    ])) : [];
    const cur = modeOf(plan);
    const who = section("What changes", [
      h("div", { class: "row", style: "align-items:center;gap:10px;margin-bottom:4px" }, [
        check(cg, v => setPlan("color_guide", v), "🎨 Colored Mark",
          "Paint each selected subject with the color of the picture it becomes; the prompt says which color becomes which picture. It does not turn on inpainting."),
        cg ? select(plan.color_guide_style, [["overlay", "translucent"], ["hatching", "hatching + contour"], ["normal map", "normal map (needs normals)"]], setMarkStyle, { style: "width:auto" }) : null,
        cg ? h("span", { class: "row", style: "gap:6px;align-items:center" }, [hint(`opacity ${Number(plan.color_guide_opacity ?? 0.35).toFixed(2)}`),
          h("input", { type: "range", min: 0.05, max: plan.color_guide_style === "overlay" ? 0.8 : 1, step: 0.01, value: plan.color_guide_opacity ?? 0.35, style: "width:140px",
            onInput: e => { plan.color_guide_opacity = parseFloat(e.target.value); }, onChange: save })]) : null,
      ]),
      cg && plan.color_guide_style === "normal map" ? h("div", { class: "note", style: "margin:2px 0 4px;padding:6px 8px" },
        "Normal map: run MoGe on the planner's image output (Render MoGe Geometry, normal_directx) and connect it to BFS Image Planner H3 Conditioning → normals. The preview shows a tint.") : null,
      subject1, ...more,
      cg ? h("div", { class: "row", style: "margin:4px 0;align-items:center;gap:8px" }, [
        h("button", { disabled: 1 + regions.length >= MAX_REFS, onClick: addRegion }, "+ Add subject"),
        names.value.length <= 1 + regions.length ? hint("add pictures (Pictures & prompt tab) to give each subject its own") : null]) : null,
      cg && !hasSel.value ? h("div", { class: "err" }, "Colored Mark needs a selection: a text or points.") : null,
      mp ? h("div", { class: "mstrip" }, [h("img", { src: mp.src, style: "max-height:220px;width:auto" }),
        hint(mp.empty ? "nothing found" : `mask covers ${(mp.coverage * 100).toFixed(1)}%${plan.crop ? " · yellow = crop box" : ""}`)]) : null,
      h("div", { class: "row", style: "margin-top:8px;align-items:center" }, [h("span", { style: "white-space:nowrap" }, "Generate"),
        select(cur, MODES.map(m => [m[0], m[1]]), setMode, { style: "width:auto" }), hint(MODE_INFO[cur])]),
      (plan.crop || plan.inpaint || plan.paste) && !hasSel.value ? h("div", { class: "err" }, `${modeName(plan)} needs a selection (text or points).`) : null,
      plan.inpaint ? h("div", { class: "row", style: "margin-top:6px;align-items:center",
        title: "How much of the selection is regenerated. 0.8-0.9 keeps pose, outline and lighting while still changing it." }, [
        h("span", { style: "white-space:nowrap" }, `Mask strength ${Number(plan.strength).toFixed(2)}`),
        h("input", { type: "range", min: 0.3, max: 1, step: 0.01, value: plan.strength, style: "flex:1;max-width:320px",
          onInput: e => { plan.strength = parseFloat(e.target.value); }, onChange: save }),
        h("button", { class: "ghost", disabled: plan.strength >= 1, onClick: () => setPlan("strength", 1) }, "reset")]) : null,
      plan.paste ? h("div", { class: "row", style: "margin-top:6px;align-items:center", title: "What the NEW subject is, so its outline on the result is pasted too. Empty = person." }, [
        h("span", { style: "white-space:nowrap" }, "new subject"),
        h("input", { type: "text", value: plan.paste_text || "", placeholder: "person (or dog, car… what the result shows)", style: "flex:1;max-width:420px",
          onChange: e => setPlan("paste_text", e.target.value) })]) : null,
      h("div", { class: "row", style: "margin-top:8px;align-items:center" }, [
        targetCrop.value ? h("img", { src: targetCrop.value, style: "height:56px;border-radius:6px" }) : null,
        h("span", { class: "hint", style: "white-space:nowrap" }, "{target}"),
        h("input", { type: "text", value: plan.target, style: "flex:1;min-width:200px", placeholder: "what is replaced, for {target} in the prompt (e.g. the young woman in a pink top)",
          onChange: e => setPlan("target", e.target.value) }),
        h("button", { disabled: !!busy.value || !hasSel.value, title: "SAM 3 cuts the selection out and the VLM describes it", onClick: describeTarget }, "🧑 Describe"),
      ]),
    ], { sub: "the selection, its picture and how it is generated" });
    const MC = plan.mask_cfg || {};
    const mnum = (k, step, l, help, scale = 1) => fld(l, h("input", { type: "number", step, min: 0, value: +((MC[k] ?? 0) * scale).toFixed(3),
      onChange: e => setMaskCfg(k, (parseFloat(e.target.value) || 0) / scale) }), help);
    const settings = section("Mask settings", [
      h("div", { class: "row", style: "gap:14px;margin-bottom:6px" }, [
        check(MC.fill_holes, v => setMaskCfg("fill_holes", v), "Fill holes"), check(MC.invert, v => setMaskCfg("invert", v), "Invert", "Use everything except the selection")]),
      h("div", { class: "grid" }, [
        mnum("expand", 1, "Expand (px)", "grow the mask"), mnum("threshold", 0.05, "Detection threshold"), mnum("max_objects", 1, "Max objects", "matches of the text"),
        mnum("padding", 1, "Crop padding (%)", "crop modes", 100), mnum("feather", 1, "Feather (px)", "paste back"),
        fld("Paste back", h("select", { value: MC.paste, onChange: e => setMaskCfg("paste", e.target.value) },
          [h("option", { value: "mask" }, "only the mask (feathered)"), h("option", { value: "box" }, "the whole box (feathered)")]), "crop modes"),
        mnum("blockify", 1, "Blockify (px, 0 = off)"),
      ]),
      hint("The planner's optional mask input (white = the subject) replaces SAM 3.", "display:block;margin-top:4px"),
    ], { open: false, sub: "SAM 3, crop and paste-back" });
    return h("div", { class: "card" }, [who, settings]);
  }

  function runTab() {
    const list = checks.value, warns = list.filter(x => x.lvl === "warn").length;
    const sp = { ...DEFAULTS.second_pass, ...(plan.second_pass || {}) };
    const set = (k, v) => setPlan("second_pass", { ...sp, [k]: v });
    const num = (k, step, min, max) => h("input", { type: "number", step, min, max, value: sp[k], onChange: e => { const v = parseFloat(e.target.value); if (Number.isFinite(v)) set(k, v); } });
    const out = secondPassSize(size.value.w, size.value.h, sp);
    return h("div", { class: "card" }, [
      section("Checks", list.length ? h("div", { class: "checks" }, list.map(x => h("div", { class: "ck" }, [h("span", { class: "lvl" }, x.lvl === "warn" ? "⚠" : "ℹ"), h("span", x.text)])))
        : h("div", { class: "hint" }, "✓ Nothing to fix."), { right: list.length ? pill(`${warns} warning(s) · ${list.length - warns} note(s)`, warns ? "warn" : "") : pill("all good", "ok") }),
      section("Graph", [hint("BFS Image Planner → plan → BFS Image Planner H3 Conditioning (clip, vae) → positive / latent → BasicGuider + SamplerCustomAdvanced → VAE Decode → BFS Image Planner Paste Back (plan) → Save Image. Paste Back pastes the crop modes and Frame + paste onto the original; in the other modes it passes the result through.", "display:block")]),
      section("Second pass (upscale)", [
        h("div", { class: "row" }, [check(sp.enabled, v => set("enabled", v), "Enable the second pass")]),
        h("div", { class: "grid", style: sp.enabled ? "" : "opacity:.55" }, [
          fld("Size", select(sp.mode, [["scale", "Scale by multiplier"], ["dims", "Target dimensions"], ["megapixels", "Megapixels"]], v => set("mode", v))),
          sp.mode === "scale" ? fld("Multiplier", num("scale", 0.05, 1, 4)) : null,
          sp.mode === "dims" ? fld("Width", num("width", 8, 64, 16384)) : null,
          sp.mode === "dims" ? fld("Height", num("height", 8, 64, 16384)) : null,
          sp.mode === "megapixels" ? fld("Megapixels", num("megapixels", 0.1, 0.1, 64)) : null,
          fld("Align (px)", num("align", 1, 1, 512), "32 recommended"),
          fld("Result", size.value.w ? `${size.value.w}×${size.value.h} → ${out.w}×${out.h}` : "—"),
        ]),
        hint("The planner outputs second_pass, upscale_width and upscale_height: upscale the denoised latent to that size and sample it again before decoding.", "display:block;margin-top:6px"),
      ], { open: !!sp.enabled, sub: sp.enabled ? `on · ${out.w}×${out.h}` : "off" }),
    ]);
  }

  function pointsModal() {
    if (!modal.open) return null;
    const regionPicture = r => (r === 1 ? +plan.color_guide_picture || 2 : markRegions(plan)[r - 2]?.picture) || 2;
    const regionColor = r => (cgOn() ? pictureColor(plan, regionPicture(r)) : "#ff283c");
    const cgOn = () => !!plan.color_guide;
    const regionsN = cgOn() ? 1 + markRegions(plan).length : 1;
    const dots = Array.from({ length: regionsN }, (_, q) => q + 1).flatMap(region => (region === modal.region ? modal.points : modal.groups[region]?.points || []).map((p, k) => h("div", {
      class: ["pt", cgOn() && "cg", region !== modal.region && "other", p.label ? "pos" : "neg"],
      style: `left:${p.x * 100}%;top:${p.y * 100}%;--point-color:${regionColor(region)}`,
      onPointerdown: e => { if (region !== modal.region) return; e.stopPropagation(); modal.points = modal.points.filter((_, j) => j !== k); modal.prev = null; } })));
    return h(Teleport, { to: "body" }, h("div", { class: "bsl", style: "background:none;padding:0" }, h("div", { class: "mmodal",
      onKeydown: e => e.stopPropagation(), onPointerdown: e => { if (e.target === e.currentTarget) modal.open = false; } }, [
      h("div", { class: "mbox" }, [
        h("div", { class: "row", style: "margin-bottom:6px" }, [h("span", { class: "sect" }, "Points"),
          cgOn() ? h("div", { class: "row" }, Array.from({ length: regionsN }, (_, q) => q + 1).map(r => h("button", { class: modal.region === r ? "pri" : "", onClick: () => switchRegion(r) },
            ` ${r} → Picture ${regionPicture(r)} · ${markName(regionColor(r))}`))) : null,
          hint("click = keep · right-click or shift+click = exclude · click a point to remove it")]),
        h("div", { class: "mstage" }, [h("div", { class: "mimg", onContextmenu: e => e.preventDefault(), onPointerdown: e => {
          const box = e.currentTarget.getBoundingClientRect();
          const x = (e.clientX - box.left) / box.width, y = (e.clientY - box.top) / box.height;
          modal.points = [...modal.points, { x: +x.toFixed(4), y: +y.toFixed(4), label: (e.button === 2 || e.shiftKey) ? 0 : 1 }]; modal.prev = null;
        } }, [h("img", { draggable: false, src: viewUrl(plan.image) }), ...dots])]),
        h("div", { class: "row", style: "margin-top:6px" }, [
          h("input", { type: "text", value: modal.text, style: "flex:1", placeholder: "text prompt (used when there are no points)", onChange: e => { modal.text = e.target.value; } }),
          h("button", { onClick: () => { modal.points = []; modal.prev = null; } }, "Clear points"),
          h("button", { class: "pri", disabled: !!modal.busy || !(modal.points.length || modal.text),
            onClick: () => previewMask({ points: modal.points, text: modal.text, ...(modal.region > 1 ? { max_objects: 1 } : {}), preview_color: regionColor(modal.region) }, true) }, "👁 Segment"),
        ]),
        modal.busy ? busyBar(c, modal.busy) : null,
        modal.prev ? h("div", { class: "mstrip" }, [h("img", { src: modal.prev.src, style: "max-height:160px;width:auto" }),
          hint(modal.prev.empty ? "nothing found" : `covers ${(modal.prev.coverage * 100).toFixed(1)}%`)]) : null,
        h("div", { class: "row", style: "margin-top:8px;justify-content:flex-end" }, [
          h("button", { onClick: () => { modal.open = false; } }, "Cancel"), h("button", { class: "pri", onClick: savePoints }, "Save")]),
      ]),
    ])));
  }

  return () => {
    const warns = checks.value.filter(x => x.lvl === "warn").length;
    const header = h("div", { class: "hdr" }, [
      h("span", { class: "ttl" }, "🖼 Image Planner"),
      plan.image && size.value.w ? pill(`${size.value.w}×${size.value.h} · ${names.value.length} picture(s) · ${modeName(plan)}`, "ok") : null,
      warns ? h("span", { class: "pill warn", style: "cursor:pointer", onClick: () => { tab.value = "run"; } }, `⚠ ${warns}`) : null,
      h("span", { class: "grow" }),
      h("label", { class: "langpick", title: t("Language") }, [h("span", { class: "hint" }, "🌐"),
        h("select", { value: uiLanguage.value, onChange: e => setUiLanguage(e.target.value) }, [
          rawH("option", { value: "en" }, "English"), rawH("option", { value: "pt" }, "Português"),
          rawH("option", { value: "es" }, "Español"), rawH("option", { value: "zh" }, "中文")])]),
      pill(plan.variant),
    ]);
    const tabs = h("div", { class: "tabs" }, TABS.map(([k, l]) => h("button", { class: ["tab", tab.value === k && "on"],
      onClick: () => { tab.value = k; store.set("bfs.imageplanner.tab", k); } }, [l, k === "run" && warns ? h("span", { class: "dot" }) : null])));
    const views = { image: imageTab, prompts: promptsTab, edit: editTab, run: runTab };
    return h("div", { class: "bsl root", tabindex: 0 }, [header, tabs,
      busy.value ? busyBar(c, busy.value) : null,
      error.value ? h("div", { class: "err" }, [h("span", { class: "grow" }, error.value), h("button", { class: "ghost", onClick: () => { error.value = ""; } }, "✕")]) : null,
      views[tab.value](), pointsModal()]);
  };
}

app.registerExtension({
  name: "BFSNodes.ImagePlanner",
  async nodeCreated(node) {
    if (node.comfyClass !== NODE) return;
    styles();
    const planWidget = node.widgets?.find(w => w.name === "plan");
    if (planWidget) {
      planWidget.type = "hidden"; planWidget.hidden = true; planWidget.computeSize = () => [0, -4];
      planWidget.options = { ...(planWidget.options || {}), hidden: true, serialize: true };
      planWidget.serializeValue = () => planWidget.value;
    }
    const host = document.createElement("div");
    host.style.cssText = "position:relative;width:100%;height:100%;min-width:760px;min-height:560px";
    node.addDOMWidget("bfs_image_planner", "div", host, { serialize: false, hideOnZoom: false });
    wheelToCanvas(host);
    const handle = {};
    createApp({ setup: () => Panel({
      getPlan: () => planWidget?.value ?? "{}",
      setPlan: v => { if (planWidget) planWidget.value = v; node.graph?.setDirtyCanvas(true); },
      expose: o => Object.assign(handle, o),
    }) }).mount(host);
    const prev = node.onConfigure;
    node.onConfigure = function () { const r = prev?.apply(this, arguments); setTimeout(() => handle.reload?.(), 0); return r; };
    node.size = [Math.max(node.size[0], 900), Math.max(node.size[1], 820)];
  },
});
