const RECORD_SECONDS = 12;
const EMOTIONS = ["neutral", "happy", "sad", "angry"];
const EMOTION_META = {
  neutral: { short: "NEU", color: "#2d6cff" },
  happy: { short: "HAP", color: "#0a9f72" },
  sad: { short: "SAD", color: "#65718c" },
  angry: { short: "ANG", color: "#ee4b62" }
};

const FREE_TASK = {
  emotion: "none",
  title: "Free recording",
  icon: "sparkles",
  script: "Record or upload any English utterance. If no browser transcript is available, the local Whisper ASR backend will transcribe it before the text branch runs.",
  hints: []
};

const FALLBACK_MODEL_PROFILES = [
  { id: "03b_acoustic_only_5fold", label: "Acoustic only, 5-fold", shortLabel: "Acoustic only 5-fold", status: "ready", runner: "03b_bridge", protocol: "5fold_session", summary: { WA: "67.42 +/- 1.40", UAR: "68.53 +/- 2.74", Macro_F1: "67.73 +/- 1.86", CCC_mean: "0.615 +/- 0.020" } },
  { id: "03b_acoustic_only_10fold", label: "Acoustic only, 10-fold", shortLabel: "Acoustic only 10-fold", status: "ready", runner: "03b_bridge", protocol: "10fold_speaker", summary: { WA: "70.01 +/- 3.37", UAR: "71.06 +/- 2.85", Macro_F1: "69.99 +/- 2.85", CCC_mean: "0.632 +/- 0.045" } },
  { id: "03b_concat_5fold", label: "Acoustic + frozen text concat, 5-fold", shortLabel: "Frozen text concat 5-fold", status: "ready", runner: "03b_bridge", protocol: "5fold_session", summary: { WA: "70.96 +/- 1.89", UAR: "72.28 +/- 2.51", Macro_F1: "71.35 +/- 2.06", CCC_mean: "0.630 +/- 0.016" } },
  { id: "03b_concat_10fold", label: "Acoustic + frozen text concat, 10-fold", shortLabel: "Frozen text concat 10-fold", status: "ready", runner: "03b_bridge", protocol: "10fold_speaker", summary: { WA: "73.10 +/- 3.38", UAR: "74.34 +/- 2.49", Macro_F1: "73.02 +/- 3.09", CCC_mean: "0.641 +/- 0.036" } },
  { id: "03b_bridge_5fold", label: "Acoustic + frozen text bridge, 5-fold", shortLabel: "Frozen text bridge 5-fold", status: "ready", runner: "03b_bridge", protocol: "5fold_session", summary: { WA: "71.02 +/- 1.79", UAR: "71.76 +/- 2.13", Macro_F1: "71.08 +/- 1.74", CCC_mean: "0.637 +/- 0.038" } },
  { id: "03b_bridge_10fold", label: "Acoustic + frozen text bridge, 10-fold", shortLabel: "Frozen text bridge 10-fold", status: "ready", runner: "03b_bridge", protocol: "10fold_speaker", summary: { WA: "72.52 +/- 4.58", UAR: "73.34 +/- 3.51", Macro_F1: "72.56 +/- 4.34", CCC_mean: "0.644 +/- 0.036" } },
  { id: "03b_bridge_rmm_10fold", label: "Acoustic + frozen text bridge + RMM, 10-fold", shortLabel: "Frozen text bridge RMM 10-fold", status: "ready", runner: "03b_bridge", protocol: "10fold_speaker", summary: { WA: "73.31 +/- 3.38", UAR: "73.94 +/- 2.32", Macro_F1: "73.15 +/- 3.07", CCC_mean: "0.642 +/- 0.037" } },
  { id: "03b_frozen_text_5fold", label: "Acoustic + frozen text bridge + RMM, 5-fold", shortLabel: "Frozen text bridge RMM 5-fold", status: "ready", runner: "03b_bridge", protocol: "5fold_session", summary: { WA: "70.05 +/- 0.94", UAR: "70.83 +/- 2.38", Macro_F1: "70.10 +/- 1.11", CCC_mean: "0.651 +/- 0.027" } },
  { id: "03b_frozen_text_10fold", label: "Acoustic + frozen text bridge + RMM original, 10-fold", shortLabel: "Frozen text bridge RMM original 10-fold", status: "ready", runner: "03b_bridge", protocol: "10fold_speaker", summary: { WA: "72.04 +/- 4.11", UAR: "73.78 +/- 3.03", Macro_F1: "72.32 +/- 4.22", CCC_mean: "0.640 +/- 0.041" } },
  { id: "03c_tuned_text_5fold", label: "Text tuned only, 5-fold", shortLabel: "Text tuned only 5-fold", status: "ready", runner: "03c_text_tuned", protocol: "5fold_session", summary: { WA: "66.28 +/- 2.28", UAR: "67.81 +/- 2.80", Macro_F1: "66.37 +/- 2.46", CCC_mean: "0.56 +/- 0.01" } },
  { id: "03c_tuned_text_10fold", label: "Text tuned only, 10-fold", shortLabel: "Text tuned only 10-fold", status: "ready", runner: "03c_text_tuned", protocol: "10fold_speaker", summary: { WA: "67.45 +/- 4.93", UAR: "69.47 +/- 5.23", Macro_F1: "67.13 +/- 5.04", CCC_mean: "0.57 +/- 0.04" } },
  { id: "03d_live_03b_03c_5fold", label: "Acoustic + text tuned expert fusion, 5-fold", shortLabel: "Acoustic + text tuned 5-fold", status: "ready", runner: "03d_live_weighted_03b_03c", protocol: "5fold_session", summary: { WA: "72.90 +/- 2.05", UAR: "73.56 +/- 2.71", Macro_F1: "73.09 +/- 2.13", CCC_mean: "0.652 +/- 0.024" } },
  { id: "03d_live_03b_03c_10fold", label: "Acoustic + text tuned expert fusion, 10-fold", shortLabel: "Acoustic + text tuned 10-fold", status: "ready", runner: "03d_live_weighted_03b_03c", protocol: "10fold_speaker", summary: { WA: "74.02 +/- 3.48", UAR: "74.64 +/- 3.38", Macro_F1: "74.05 +/- 3.63", CCC_mean: "0.661 +/- 0.033" } }
];
const IEMOCAP_STYLE_PROMPTS = [
  {
    title: "Repeated utterance A",
    script: "I almost got married two years ago.",
    origin: "IEMOCAP repeated transcript candidate",
  },
  {
    title: "Repeated utterance B",
    script: "No, you don't. I'm a pretty tough guy.",
    origin: "IEMOCAP repeated transcript candidate",
  },
  {
    title: "Repeated utterance C",
    script: "Yeah. I can see that.",
    origin: "IEMOCAP repeated transcript candidate",
  }
];

const TASK_GROUPS = [
  {
    emotion: "neutral",
    label: "Neutral",
    icon: "meh",
    prompts: IEMOCAP_STYLE_PROMPTS.map((prompt) => ({ ...prompt, hints: ["Balanced pitch", "Moderate energy", "Natural pace"] }))
  },
  {
    emotion: "happy",
    label: "Happy",
    icon: "smile",
    prompts: IEMOCAP_STYLE_PROMPTS.map((prompt) => ({ ...prompt, hints: ["Brighter pitch", "Warmer energy", "Livelier rhythm"] }))
  },
  {
    emotion: "sad",
    label: "Sad",
    icon: "frown",
    prompts: IEMOCAP_STYLE_PROMPTS.map((prompt) => ({ ...prompt, hints: ["Lower pitch", "Softer energy", "Slower pace"] }))
  },
  {
    emotion: "angry",
    label: "Angry",
    icon: "angry",
    prompts: IEMOCAP_STYLE_PROMPTS.map((prompt) => ({ ...prompt, hints: ["Higher intensity", "Tense voice", "Sharper emphasis"] }))
  }
];

const state = {
  task: FREE_TASK,
  selectedEmotion: "none",
  selectedPromptIndex: 0,
  queue: [],
  selectedQueueId: null,
  selectedResultId: null,
  compareTargetId: null,
  resultFeatureTab: "raw",
  resultFeatureSlides: { raw: 0, cnn1d: 0, cnn2d: 0, stats: 0 },
  modelProfiles: FALLBACK_MODEL_PROFILES,
  selectedModelProfileId: "03d_live_03b_03c_5fold",
  currentRecording: null,
  audioContext: null,
  stream: null,
  mediaRecorder: null,
  chunks: [],
  isRecording: false,
  startedAt: 0,
  liveAnalyser: null,
  animationId: 0,
  radarChart: null,
  recognition: null,
  sttSegments: [],
  sttDraft: "",
  sttSegmentStart: 0,
  queueOpen: false,
  isTabAnimating: false,
  liveWaveHistory: [],
  queueDrag: {
    active: false,
    moved: false,
    offsetX: 0,
    offsetY: 0,
    startX: 0,
    startY: 0
  },
  flowMap: {
    x: 18,
    y: 34,
    scale: 0.72,
    panning: false,
    draggingNode: null,
    moved: false,
    startX: 0,
    startY: 0,
    nodeStartLeft: 0,
    nodeStartTop: 0
  },
  pipelineTimers: {
    totalStart: 0,
    totalInterval: null,
    stepIntervals: new Map()
  }
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

document.addEventListener("DOMContentLoaded", init);

function init() {
  initIcons();
  renderTaskGrid();
  renderProbabilityBars();
  initLedMeter();
  selectTask(FREE_TASK);
  bindEvents();
  drawAmbient();
  drawOrb(0, new Uint8Array(512));
  drawEmptyWaveformCanvas($("#liveWaveCanvas"), "idle");
  drawEmptyWaveformCanvas($("#processWaveCanvas"), "queued audio");
  drawEmptySpectrogramCanvas($("#processMelCanvas"), "log-Mel preview");
  initRadar();
  renderQueue();
  renderProcessQueueSelector();
  renderResultHistory();
  renderModelProfileSelector();
  loadModelProfiles();
  restoreQueueBubblePosition();
  initQueueDrag();
  initFlowMap();
  if (window.gsap) {
    gsap.from(".app-header, .tab-screen.active .panel:not(.queue-drawer)", { opacity: 0, y: 16, duration: 0.6, stagger: 0.035, ease: "power3.out" });
  }
  $("#queueDrawer")?.removeAttribute("style");
}

function initIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function bindEvents() {
  document.addEventListener("pointerdown", unlockSoundOnce, { once: true, passive: true });
  document.addEventListener("keydown", unlockSoundOnce, { once: true });
  $$(".tab-btn").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });
  $("#shuffleTaskBtn").addEventListener("click", () => {
    playTone("click");
    const emotionPool = ["none", ...TASK_GROUPS.map((group) => group.emotion)];
    const emotion = emotionPool[Math.floor(Math.random() * emotionPool.length)];
    const group = TASK_GROUPS.find((item) => item.emotion === emotion);
    const promptIndex = group ? Math.floor(Math.random() * group.prompts.length) : 0;
    selectTask(buildTask(emotion, promptIndex));
  });
  $("#queueBubble").addEventListener("click", (event) => {
    if (state.queueDrag.moved) {
      event.preventDefault();
      state.queueDrag.moved = false;
      return;
    }
    toggleQueueDrawer();
  });
  $("#closeQueueBtn").addEventListener("click", closeQueueDrawer);
  $("#recordBtn").addEventListener("click", () => {
    playTone("click");
    if (state.isRecording) stopRecording();
    else startRecording();
  });
  $("#playBtn").addEventListener("click", () => {
    const item = state.currentRecording || getSelectedItem();
    if (!item?.audioUrl) return;
    playAudioUrl(item.audioUrl);
    playTone("play");
  });
  $("#uploadAudioBtn")?.addEventListener("click", () => $("#audioUploadInput")?.click());
  $("#audioUploadInput")?.addEventListener("change", handleAudioUpload);
  $("#addQueueBtn").addEventListener("click", addCurrentRecordingToQueue);
  $("#modelProfileSelect")?.addEventListener("change", (event) => {
    state.selectedModelProfileId = event.target.value || "03d_live_03b_03c_5fold";
    renderModelProfileSelector();
    renderProcessSelection();
    playTone("click");
  });
  $("#processQueueSelect")?.addEventListener("change", (event) => {
    state.selectedQueueId = event.target.value || null;
    renderQueue();
    renderProcessSelection();
    resetPipeline();
    playTone("click");
  });
  $("#resetBtn").addEventListener("click", resetCurrentRecording);
  $("#analyzeBtn").addEventListener("click", analyzeSelectedQueue);
  $("#backRecordBtn").addEventListener("click", () => switchTab("record"));
  $("#playResultBtn")?.addEventListener("click", playSelectedResultAudio);
  $("#downloadResultBtn")?.addEventListener("click", downloadSelectedAudio);
  $("#exportReportBtn").addEventListener("click", exportSelectedReport);
  $("#exportReportPackageBtn")?.addEventListener("click", exportSelectedReportPackage);
  $("#downloadBtn").addEventListener("click", downloadSelectedAudio);
  $("#compareTargetSelect")?.addEventListener("change", (event) => {
    state.compareTargetId = event.target.value || null;
    renderComparePanel();
    playTone("click");
  });
  document.addEventListener("click", (event) => {
    const tab = event.target.closest("[data-result-feature-tab]");
    if (!tab) return;
    state.resultFeatureTab = tab.dataset.resultFeatureTab || "raw";
    const item = getSelectedResultItem();
    if (item?.report) item.report.featureTab = state.resultFeatureTab;
    renderResultFeatureTabs();
    if (item) drawResultFeatureCanvases(item);
    playTone("tab");
  });
  bindProcessingExplainer();
  bindResultFeatureInspector();
}

function switchTab(tabName) {
  const button = $(`.tab-btn[data-tab="${tabName}"]`);
  if (!button) return;
  const current = $(".tab-screen.active");
  const next = $(`#${tabName}Tab`);
  if (!next || current === next || state.isTabAnimating) return;
  const currentName = current?.id?.replace("Tab", "") || "record";
  const order = ["record", "process", "results"];
  const direction = Math.sign(order.indexOf(tabName) - order.indexOf(currentName)) || 1;
  $$(".tab-btn").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabName));
  animateTabButton(button);
  if (tabName !== "record" && state.queueOpen) closeQueueDrawer(false);
  if (window.gsap && current) {
    state.isTabAnimating = true;
    next.classList.add("active");
    next.scrollTop = 0;
    gsap.killTweensOf([current, next]);
    gsap.set([current, next], {
      position: "absolute",
      inset: 0,
      width: "100%",
      height: "100%",
      transformOrigin: "center center"
    });
    gsap.set(next, { x: direction * 54, opacity: 0, scale: 0.985, filter: "blur(5px)" });
    gsap.to(current, {
      x: direction * -42,
      opacity: 0,
      scale: 0.985,
      filter: "blur(5px)",
      duration: 0.24,
      ease: "power2.inOut"
    });
    gsap.to(next, {
      x: 0,
      opacity: 1,
      scale: 1,
      filter: "blur(0px)",
      duration: 0.38,
      delay: 0.04,
      ease: "power3.out",
      onComplete: () => {
        current.classList.remove("active");
        gsap.set([current, next], { clearProps: "all" });
        state.isTabAnimating = false;
      }
    });
  } else {
    $$(".tab-screen").forEach((screen) => screen.classList.remove("active"));
    next.classList.add("active", "tab-pop-in");
    setTimeout(() => next.classList.remove("tab-pop-in"), 360);
  }
  playTone("tab");
}

function animateTabButton(button) {
  if (!window.gsap || !button) return;
  gsap.fromTo(button, { scale: 0.96 }, { scale: 1, duration: 0.28, ease: "back.out(2.6)" });
  const marker = button.querySelector("span");
  if (marker) gsap.fromTo(marker, { rotate: -18, scale: 0.72 }, { rotate: 0, scale: 1, duration: 0.34, ease: "back.out(2.8)" });
}

function unlockTab(tabName) {
  const button = $(`.tab-btn[data-tab="${tabName}"]`);
  if (button) button.disabled = false;
}

function bindProcessingExplainer() {
  $$("#acousticFeatureTabs button").forEach((button) => {
    button.addEventListener("click", (event) => {
      event.stopPropagation();
      selectAcousticFeature(button.dataset.feature);
      playTone("click");
    });
  });
  $(".advanced-flowchart")?.addEventListener("click", (event) => {
    if (state.flowMap.moved) {
      event.preventDefault();
      state.flowMap.moved = false;
      return;
    }
    if (event.target.closest(".feature-tabs, .compact-feature-tabs")) return;
    const target = event.target.closest(".inspectable");
    if (!target || !$(".advanced-flowchart").contains(target)) return;
    openInspectModal(target.dataset.inspect || target.dataset.step || "input");
  });
  $("#closeInspectBtn")?.addEventListener("click", closeInspectModal);
  $("[data-close-inspect]")?.addEventListener("click", closeInspectModal);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeInspectModal();
  });
}

function bindResultFeatureInspector() {
  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-result-inspect]");
    if (!target) return;
    const featureCard = target.closest(".feature-preview-card");
    if (featureCard && !featureCard.classList.contains("slide-active")) {
      const panel = featureCard.closest("[data-result-feature-panel]");
      const tab = panel?.dataset.resultFeaturePanel || state.resultFeatureTab || "raw";
      const cards = panel ? Array.from(panel.querySelectorAll(".feature-preview-card")) : [];
      const index = cards.indexOf(featureCard);
      if (index >= 0) {
        state.resultFeatureTab = tab;
        state.resultFeatureSlides[tab] = index;
        renderResultFeatureTabs();
        playTone("tab");
        event.preventDefault();
        event.stopPropagation();
      }
      return;
    }
    const kind = target.dataset.resultInspect || "wave";
    openInspectModal(kind, true);
  });
}

function initFlowMap() {
  const viewport = $("#flowMapViewport");
  const canvas = $("#flowMapCanvas");
  if (!viewport || !canvas) return;
  applyFlowMapTransform();
  updateFlowEdges();
  new ResizeObserver(() => updateFlowEdges()).observe(canvas);

  $("#flowZoomIn")?.addEventListener("click", () => zoomFlowMap(1.15));
  $("#flowZoomOut")?.addEventListener("click", () => zoomFlowMap(1 / 1.15));
  $("#flowZoomReset")?.addEventListener("click", () => resetFlowMapView());

  viewport.addEventListener("wheel", (event) => {
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.08 : 1 / 1.08;
    zoomFlowMap(factor, event.clientX, event.clientY);
  }, { passive: false });

  viewport.addEventListener("pointerdown", (event) => {
    const node = event.target.closest(".draggable-node");
    if (node && canvas.contains(node)) {
      const canDragNode = event.target.closest(".node-drag-handle, .node-head, .node-kicker") || event.target === node;
      if (canDragNode) {
        startFlowNodeDrag(event, node);
        return;
      }
    }
    if (event.target.closest(".flow-map-toolbar, button, canvas, .inspectable")) return;
    startFlowPan(event);
  });

  window.addEventListener("pointermove", handleFlowPointerMove);
  window.addEventListener("pointerup", stopFlowInteraction);
}

function applyFlowMapTransform() {
  const canvas = $("#flowMapCanvas");
  if (!canvas) return;
  canvas.style.transform = `translate(${state.flowMap.x}px, ${state.flowMap.y}px) scale(${state.flowMap.scale})`;
  const label = $("#flowZoomLabel");
  if (label) label.textContent = `${Math.round(state.flowMap.scale * 100)}%`;
  updateFlowEdges();
}

function zoomFlowMap(factor, clientX = null, clientY = null) {
  const viewport = $("#flowMapViewport");
  if (!viewport) return;
  const oldScale = state.flowMap.scale;
  const nextScale = clamp(oldScale * factor, 0.24, 2.25);
  const rect = viewport.getBoundingClientRect();
  const px = clientX == null ? rect.left + rect.width / 2 : clientX;
  const py = clientY == null ? rect.top + rect.height / 2 : clientY;
  const mapX = (px - rect.left - state.flowMap.x) / oldScale;
  const mapY = (py - rect.top - state.flowMap.y) / oldScale;
  state.flowMap.scale = nextScale;
  state.flowMap.x = px - rect.left - mapX * nextScale;
  state.flowMap.y = py - rect.top - mapY * nextScale;
  applyFlowMapTransform();
  playTone("click");
}

function resetFlowMapView() {
  state.flowMap.x = 18;
  state.flowMap.y = 34;
  state.flowMap.scale = 0.72;
  applyFlowMapTransform();
  playTone("reset");
}

function startFlowPan(event) {
  state.flowMap.panning = true;
  state.flowMap.moved = false;
  state.flowMap.startX = event.clientX;
  state.flowMap.startY = event.clientY;
  state.flowMap.nodeStartLeft = state.flowMap.x;
  state.flowMap.nodeStartTop = state.flowMap.y;
  $("#flowMapViewport")?.classList.add("panning");
  event.preventDefault();
}

function startFlowNodeDrag(event, node) {
  state.flowMap.draggingNode = node;
  state.flowMap.moved = false;
  state.flowMap.startX = event.clientX;
  state.flowMap.startY = event.clientY;
  state.flowMap.nodeStartLeft = parseFloat(node.style.left || node.offsetLeft || 0);
  state.flowMap.nodeStartTop = parseFloat(node.style.top || node.offsetTop || 0);
  node.classList.add("dragging-node");
  event.preventDefault();
  event.stopPropagation();
}

function handleFlowPointerMove(event) {
  if (state.flowMap.draggingNode) {
    const dx = (event.clientX - state.flowMap.startX) / state.flowMap.scale;
    const dy = (event.clientY - state.flowMap.startY) / state.flowMap.scale;
    if (Math.hypot(dx, dy) > 2) state.flowMap.moved = true;
    const node = state.flowMap.draggingNode;
    const canvas = $("#flowMapCanvas");
    const canvasWidth = canvas?.offsetWidth || 3600;
    const canvasHeight = canvas?.offsetHeight || 1800;
    const dragPadding = 2400;
    const nextLeft = state.flowMap.nodeStartLeft + dx;
    const nextTop = state.flowMap.nodeStartTop + dy;
    node.style.left = `${clamp(nextLeft, -dragPadding, canvasWidth + dragPadding - node.offsetWidth)}px`;
    node.style.top = `${clamp(nextTop, -dragPadding, canvasHeight + dragPadding - node.offsetHeight)}px`;
    updateFlowEdges();
    return;
  }
  if (state.flowMap.panning) {
    const dx = event.clientX - state.flowMap.startX;
    const dy = event.clientY - state.flowMap.startY;
    if (Math.hypot(dx, dy) > 4) state.flowMap.moved = true;
    state.flowMap.x = state.flowMap.nodeStartLeft + dx;
    state.flowMap.y = state.flowMap.nodeStartTop + dy;
    applyFlowMapTransform();
  }
}

function stopFlowInteraction() {
  if (state.flowMap.draggingNode) {
    state.flowMap.draggingNode.classList.remove("dragging-node");
    state.flowMap.draggingNode = null;
    setTimeout(() => { state.flowMap.moved = false; }, 120);
  }
  if (state.flowMap.panning) {
    state.flowMap.panning = false;
    $("#flowMapViewport")?.classList.remove("panning");
    setTimeout(() => { state.flowMap.moved = false; }, 120);
  }
}

function updateFlowEdges() {
  const canvas = $("#flowMapCanvas");
  if (!canvas) return;
  $$(".flow-edge[data-from][data-to]").forEach((edge) => {
    const from = $(`[data-node-id="${edge.dataset.from}"]`);
    const to = $(`[data-node-id="${edge.dataset.to}"]`);
    if (!from || !to) return;
    const a = nodeAnchor(from, "right");
    const b = nodeAnchor(to, "left");
    const dx = Math.max(110, Math.abs(b.x - a.x) * 0.52);
    const c1x = a.x + dx;
    const c2x = b.x - dx;
    edge.setAttribute("d", `M${a.x} ${a.y} C${c1x} ${a.y}, ${c2x} ${b.y}, ${b.x} ${b.y}`);
  });
}

function nodeAnchor(node, side = "right") {
  const left = parseFloat(node.style.left || node.offsetLeft || 0);
  const top = parseFloat(node.style.top || node.offsetTop || 0);
  const width = node.offsetWidth;
  const height = node.offsetHeight;
  return {
    x: side === "right" ? left + width : left,
    y: top + height / 2
  };
}

function selectAcousticFeature(featureName = "temporal") {
  $$("#acousticFeatureTabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.feature === featureName);
  });
  $$("[data-feature-panel]").forEach((panel) => {
    panel.classList.toggle("active-feature", panel.dataset.featurePanel === featureName);
  });
}

function openInspectModal(kind, preferResult = false) {
  const item = preferResult ? (getSelectedResultItem() || getSelectedItem()) : (getSelectedItem() || getSelectedResultItem());
  const report = item?.report || null;
  const content = buildInspectContent(kind, item, report);
  $("#inspectTag").textContent = content.tag;
  $("#inspectTitle").textContent = content.title;
  $("#inspectSubtitle").textContent = content.subtitle;
  $("#inspectBody").innerHTML = content.body;
  $("#inspectOverlay").classList.add("open");
  $("#inspectOverlay").setAttribute("aria-hidden", "false");
  document.body.style.overflow = "hidden";
  if (kind === "wave" && item?.samples) {
    drawSpeechWaveform($("#inspectWaveCanvas"), item.samples, item.sampleRate, item.duration, resolvedTranscriptSegments(item, report), item.duration, "");
  }
  if (kind === "temporal" && report?.mfcc) drawHeatmap($("#inspectMfccCanvas"), report.mfcc, "MFCC / cepstral sequence");
  if (kind === "spectral" && report?.logMel) {
    drawHeatmap($("#inspectMelCanvas"), report.logMel, "Log-Mel spectrogram");
  }
  if (kind === "spectral_delta" && report?.logMel) {
    drawHeatmap($("#inspectDeltaMelCanvas"), deltaMatrix(report.logMel), "Delta log-Mel approximation");
  }
  if (kind === "feature_trends" && report) {
    drawFeatureTrendCanvas($("#inspectTrendCanvas"), report, "Frame-level feature trends");
  }
  playTone("tab");
}

function closeInspectModal() {
  const overlay = $("#inspectOverlay");
  if (!overlay?.classList.contains("open")) return;
  overlay.classList.remove("open");
  overlay.setAttribute("aria-hidden", "true");
  document.body.style.overflow = "";
}

function buildInspectContent(kind, item, report) {
  const s = report?.stats || {};
  const metricGrid = `
    <div class="inspect-grid">
      <div class="inspect-metric"><span>Duration</span><strong>${item ? formatDuration(item.duration) : "--"}</strong></div>
      <div class="inspect-metric"><span>Frames</span><strong>${report?.frames?.length || "--"}</strong></div>
      <div class="inspect-metric"><span>STT lines</span><strong>${item?.sttSegments?.length || 0}</strong></div>
    </div>`;
  const formulas = {
    attention: `
      <span class="math-line">Attention(Q,K,V) = softmax(QK<sup>T</sup> / âˆšd<sub>k</sub>)V</span>
      <span class="math-note">Q asks what to look for, K locates matching evidence, V carries the information passed forward.</span>`,
    multiHead: `
      <span class="math-line">head<sub>i</sub> = Attention(QW<sub>i</sub><sup>Q</sup>, KW<sub>i</sub><sup>K</sup>, VW<sub>i</sub><sup>V</sup>)</span>
      <span class="math-line">MHA(Q,K,V) = Concat(head<sub>1</sub>, ..., head<sub>h</sub>)W<sup>O</sup></span>`,
    cross: `
      <span class="math-line">Q<sub>Aâ†’T</sub> = B<sub>A</sub>W<sup>Q</sup>, K<sub>T</sub> = Z<sub>T</sub>W<sup>K</sup>, V<sub>T</sub> = Z<sub>T</sub>W<sup>V</sup></span>
      <span class="math-line">H<sub>Aâ†’T</sub> = MHA(Q<sub>Aâ†’T</sub>, K<sub>T</sub>, V<sub>T</sub>)</span>
      <span class="math-line">H<sub>Tâ†’A</sub> = MHA(Q<sub>Tâ†’A</sub>, K<sub>A</sub>, V<sub>A</sub>)</span>`,
    fusion: `
      <span class="math-line">Î± = sigmoid(W<sub>g</sub>[h<sub>Aâ†’T</sub>; h<sub>Tâ†’A</sub>] + b<sub>g</sub>)</span>
      <span class="math-line">z<sub>fused</sub> = Î± Â· h<sub>Aâ†’T</sub> + (1 - Î±) Â· h<sub>Tâ†’A</sub></span>
      <span class="math-note">The balanced weight is computed inside fusion, after bridge cross-attention has aligned acoustic and text evidence.</span>`,
    rmm: `
      <span class="math-line">m<sub>A</sub>, m<sub>T</sub> ~ Bernoulli(1 - p<sub>mask</sub>)</span>
      <span class="math-line">z<sub>train</sub> = Fusion(m<sub>A</sub>z<sub>A</sub>, m<sub>T</sub>z<sub>T</sub>)</span>
      <span class="math-note">Random Modality Masking is a training regularizer; inference normally uses both modalities.</span>`,
    vad: `<span class="math-line">Å·<sub>VAD</sub> = [Å·<sub>valence</sub>, Å·<sub>arousal</sub>, Å·<sub>dominance</sub>] âˆˆ [0,1]<sup>3</sup></span>`,
    softmax: `<span class="math-line">p<sub>emotion</sub> = softmax(W<sub>cls</sub>z<sub>fused</sub> + b<sub>cls</sub>)</span>`,
    loss: `
      <span class="math-line">L = Î»<sub>cls</sub>CE(y, p) + Î»<sub>vad</sub>MAE(y<sub>VAD</sub>, Å·<sub>VAD</sub>) + Î»<sub>ccc</sub>(1 - CCC)</span>
      <span class="math-line">CCC = 2ÏÏƒ<sub>y</sub>Ïƒ<sub>Å·</sub> / (Ïƒ<sub>y</sub><sup>2</sup> + Ïƒ<sub>Å·</sub><sup>2</sup> + (Î¼<sub>y</sub> - Î¼<sub>Å·</sub>)<sup>2</sup>)</span>`,
    temporal: `
      <span class="math-line">X<sub>temporal</sub> = Concat(MFCC, Î”MFCC, Î”Â²MFCC, RMS, ZCR, F0, spectral LLD)</span>
      <span class="math-line">z<sub>temporal</sub> = Pool(BiRNN/TCN(Conv1D(X<sub>temporal</sub>)))</span>`,
    spectral: `
      <span class="math-line">X<sub>spectral</sub> = Stack(logMel, Î”logMel, Î”Â²logMel)</span>
      <span class="math-line">z<sub>spectral</sub> = GAP(SE-CNN2D(X<sub>spectral</sub>))</span>`,
    stats: `
      <span class="math-line">X<sub>stats</sub> = Î¦(X) = [mean, std, min, max, median, p10, p90, IQR]</span>
      <span class="math-line">z<sub>stats</sub> = MLP(StandardScaler(X<sub>stats</sub>))</span>`,
    e2v: `
      <span class="math-line">e = emotion2vec(waveform<sub>16kHz</sub>) âˆˆ R<sup>768</sup></span>
      <span class="math-line">z<sub>e2v</sub> = AdapterMLP(StandardScaler(e))</span>`
  };
  const transcriptHtml = item ? renderInspectTranscript(item) : `<div class="empty-state">No transcript has been loaded.</div>`;
  const defaults = {
    input: {
      tag: "Input",
      title: "Queued audio + script package",
      subtitle: "This is the only entry point of the Processing tab. A recorded queue item provides waveform samples and timestamped transcript.",
      body: `${metricGrid}<div class="explain-card"><span>Role</span><p>The dashed input box keeps two aligned views of the same take: acoustic waveform for speech features, and script/STT timestamps for the text branch and report context.</p></div>`
    },
    wave: {
      tag: "WAV audio",
      title: "Full waveform with timestamp axis",
      subtitle: "The waveform is decoded from the recorded browser audio and becomes the acoustic input.",
      body: `<canvas id="inspectWaveCanvas" class="inspect-hero-canvas" width="1200" height="330"></canvas>${metricGrid}<div class="explain-card"><span>Why it matters</span><p>Waveform amplitude over time is the raw signal. From it the demo derives frame-level energy, ZCR, spectral descriptors, log-Mel and MFCC-like features.</p></div>`
    },
    transcript: {
      tag: "Script / STT",
      title: "Timestamped speech-to-text view",
      subtitle: "The text branch uses transcript content; the report uses word timing for feedback display.",
      body: `<div class="inspect-transcript-lines">${transcriptHtml}</div>${formulaBlock("Text sequence", `<span class="math-line">text -> tokenizer -> input_ids + attention_mask</span><span class="math-line">Z<sub>T</sub> = Projection(TextEncoder(input_ids, attention_mask))</span>`)}`
    },
    acoustic: {
      tag: "Acoustic",
      title: "Four acoustic branches",
      subtitle: "The acoustic block keeps complementary views instead of compressing speech into one feature type too early.",
      body: `${metricGrid}<div class="explain-card"><span>Feature tabs</span><p>The four pills on this block are real tabs. Selecting Temporal, Spectral, Stats or E2V highlights the matching feature card and its downstream encoder token.</p></div><div class="explain-card"><span>Branch logic</span><p>A temporal branch captures frame-by-frame acoustic dynamics, a spectrogram branch captures time-frequency patterns, an emotion2vec branch contributes pretrained emotion representation, and a statistical branch preserves stable clip-level descriptors.</p></div>${formulaBlock("Acoustic token set", `<span class="math-line">Z<sub>A</sub> = [z<sub>temporal</sub>; z<sub>spectral</sub>; z<sub>stats</sub>; z<sub>e2v</sub>]</span><span class="math-line">A = SelfAttention(Z<sub>A</sub>)</span>`)}`
    },
    temporal: {
      tag: "Branch A",
      title: "Temporal acoustic branch",
      subtitle: "MFCC/delta/prosody/spectral low-level descriptors keep emotion changes along time.",
      body: `<canvas id="inspectMfccCanvas" class="inspect-hero-canvas" width="1100" height="320"></canvas>${metricsForStats(s)}${formulaBlock("Input", formulas.temporal)}<div class="explain-card"><span>Encoder</span><p>1D-CNN/TCN detects local acoustic patterns, BiLSTM/GRU reads context in both directions, and attention pooling emphasizes frames that contain stronger emotion evidence.</p></div>`
    },
    spectral: {
      tag: "Branch B",
      title: "Spectrogram branch",
      subtitle: "A 2D time-frequency view helps CNN filters see emotion-related spectral texture.",
      body: `<canvas id="inspectMelCanvas" class="inspect-hero-canvas" width="1000" height="300"></canvas>${metricGrid}${formulaBlock("Input", formulas.spectral)}<div class="explain-card"><span>Encoder</span><p>2D-CNN extracts local time-frequency patterns. SE/channel attention reweights useful feature maps before producing a spectral token.</p></div>`
    },
    feature_trends: {
      tag: "Feature trends",
      title: "Frame-level acoustic trend view",
      subtitle: "This view enlarges the running evidence used by temporal and statistical feature groups.",
      body: `<canvas id="inspectTrendCanvas" class="inspect-hero-canvas" width="1100" height="360"></canvas>${metricsForStats(s)}<div class="explain-card"><span>Signals shown</span><p>The trend panel summarizes RMS energy, pitch/F0, spectral centroid and ZCR over time. It is useful for checking whether a clip is low-energy, monotone, noisy, bright, or changing strongly across frames.</p></div>`
    },
    spectral_delta: {
      tag: "Branch B",
      title: "Delta log-Mel evidence",
      subtitle: "Delta features approximate how the spectral pattern changes between adjacent frames.",
      body: `<canvas id="inspectDeltaMelCanvas" class="inspect-hero-canvas" width="1100" height="320"></canvas>${formulaBlock("Delta spectral input", `<span class="math-line">Î”logMel[t] = logMel[t] - logMel[t - 1]</span><span class="math-line">X<sub>spectral</sub> = [logMel, Î”logMel, Î”Â²logMel]</span>`)}<div class="explain-card"><span>Why it matters</span><p>Emotion is often carried by movement, not only static spectrum. Delta and delta-delta log-Mel show whether the speaker changes intensity, brightness and articulation over time.</p></div>`
    },
    e2v: {
      tag: "Branch C",
      title: "Frozen emotion2vec branch",
      subtitle: "A pretrained speech emotion representation acts as a compact emotion prior.",
      body: `${metricGrid}${formulaBlock("Projection", formulas.e2v)}<div class="explain-card"><span>Why frozen</span><p>The demo keeps this encoder frozen to stay lightweight and to avoid overfitting the small SER datasets during live use.</p></div>`
    },
    stats: {
      tag: "Branch D",
      title: "Statistical branch",
      subtitle: "Clip-level descriptors preserve stable cues that sequence models may blur.",
      body: `${metricsForStats(s)}${formulaBlock("Feature vector", formulas.stats)}<div class="explain-card"><span>Role</span><p>The vector summarizes loudness, voicing, brightness, rolloff, contrast, pitch and stability. It can feed a small MLP or an RBF-SVM probability branch.</p></div>`
    },
    acoustic_attention: {
      tag: "Attention",
      title: "Acoustic self-attention",
      subtitle: "The four acoustic tokens exchange information before touching the text branch.",
      body: `${formulaBlock("Scaled dot-product attention", formulas.attention)}${formulaBlock("Multi-head attention", formulas.multiHead)}<div class="explain-card"><span>Function</span><p>Self-attention lets the model learn whether temporal dynamics, log-Mel texture, emotion2vec embedding or statistical descriptors should dominate for this utterance.</p></div>`
    },
    text_model: {
      tag: "Text",
      title: "Pretrained text branch",
      subtitle: "Transcript tokens are encoded by a RoBERTa language model and projected into the same fusion space.",
      body: `${transcriptHtml}${formulaBlock("Text encoder", `<span class="math-line">Z<sub>T</sub> = Projection(RoBERTa(tokens, mask))</span><span class="math-line">T = SelfAttention(Z<sub>T</sub>)</span>`)}<div class="explain-card"><span>Function</span><p>The branch adds linguistic context, while acoustic cues still decide how the sentence was spoken.</p></div>`
    },
    bridge: {
      tag: "Fusion",
      title: "Bridge cross-attention",
      subtitle: "Bridge tokens are the controlled meeting point between acoustic and text tokens.",
      body: `${formulaBlock("Cross-attention", `${formulas.cross}${formulas.attention}`)}<div class="explain-card"><span>Function</span><p>Instead of concatenating everything blindly, learnable bridge tokens query acoustic and text evidence and keep fusion more structured.</p></div>`
    },
    balanced_fusion: {
      tag: "Fusion",
      title: "Balanced fusion",
      subtitle: "A gate controls how much acoustic and text evidence enters the final decision.",
      body: `${formulaBlock("Balanced gated fusion", formulas.fusion)}${formulaBlock("Training regularizer", formulas.rmm)}<div class="explain-card"><span>Function</span><p>The gate prevents transcript information from overwhelming emotion cues in the voice. This follows the 03B idea of balanced bridge fusion and random modality masking.</p></div>`
    },
    heads: {
      tag: "Heads",
      title: "Emotion classifier and VAD regressor",
      subtitle: "The same fused vector produces categorical emotion and dimensional affect scores.",
      body: `${predictionMetrics(report)}${formulaBlock("Outputs", `${formulas.softmax}${formulas.vad}`)}${formulaBlock("Training objective", formulas.loss)}<div class="explain-card"><span>Function</span><p>Emotion uses four demo labels here. VAD describes whether the voice sounds positive/negative, excited/calm and dominant/weak.</p></div>`
    },
    output: {
      tag: "Output",
      title: "Final report output",
      subtitle: "The processed queue item becomes the Results tab report.",
      body: `${predictionMetrics(report)}<div class="explain-card"><span>Report contents</span><p>The output stores predicted emotion, confidence, VAD values, waveform, feature summary and timestamped transcript.</p></div>`
    }
  };
  return defaults[kind] || defaults.input;
}

function formulaBlock(label, html) {
  return `<div class="formula-card math-card"><span>${label}</span><div class="math-box">${html}</div></div>`;
}

function metricsForStats(s) {
  return `
    <div class="inspect-grid">
      <div class="inspect-metric"><span>RMS</span><strong>${Number.isFinite(s.rmsMean) ? s.rmsMean.toFixed(4) : "--"}</strong></div>
      <div class="inspect-metric"><span>ZCR</span><strong>${Number.isFinite(s.zcrMean) ? s.zcrMean.toFixed(4) : "--"}</strong></div>
      <div class="inspect-metric"><span>Centroid</span><strong>${Number.isFinite(s.centroidMean) ? Math.round(s.centroidMean) + " Hz" : "--"}</strong></div>
    </div>`;
}

function predictionMetrics(report) {
  const p = report?.prediction;
  return `
    <div class="inspect-grid">
      <div class="inspect-metric"><span>Emotion</span><strong>${p ? capitalize(p.emotion) : "--"}</strong></div>
      <div class="inspect-metric"><span>Confidence</span><strong>${p ? pct(p.confidence) : "--"}</strong></div>
      <div class="inspect-metric"><span>VAD score</span><strong>${p ? `${formatVadScore(p.valence)} / ${formatVadScore(p.arousal)} / ${formatVadScore(p.dominance)}` : "--"}</strong></div>
    </div>`;
}

function renderInspectTranscript(item) {
  const segments = item?.sttSegments?.length ? item.sttSegments : buildRecorderTimeline(item);
  if (!segments?.length) return `<div class="empty-state">No timestamped text yet.</div>`;
  return segments.map((segment) => `
    <div class="stt-line">
      <span>${formatDuration(segment.start || 0)} - ${formatDuration(segment.end || item.duration || RECORD_SECONDS)}</span>
      <p>${escapeHtml(segment.text || "")}</p>
    </div>
  `).join("");
}

function toggleQueueDrawer() {
  state.queueOpen = !state.queueOpen;
  $("#queueDrawer").removeAttribute("style");
  $("#queueDrawer").classList.toggle("open", state.queueOpen);
  $("#queueDrawer").setAttribute("aria-hidden", state.queueOpen ? "false" : "true");
  playTone("tab");
}

function openQueueDrawer() {
  state.queueOpen = true;
  $("#queueDrawer").removeAttribute("style");
  $("#queueDrawer").classList.add("open");
  $("#queueDrawer").setAttribute("aria-hidden", "false");
}

function closeQueueDrawer(play = true) {
  state.queueOpen = false;
  $("#queueDrawer").removeAttribute("style");
  $("#queueDrawer").classList.remove("open");
  $("#queueDrawer").setAttribute("aria-hidden", "true");
  if (play) playTone("click");
}

function initQueueDrag() {
  const bubble = $("#queueBubble");
  if (!bubble) return;
  bubble.addEventListener("pointerdown", startQueueDrag);
  window.addEventListener("pointermove", moveQueueDrag);
  window.addEventListener("pointerup", endQueueDrag);
}

function restoreQueueBubblePosition() {
  const bubble = $("#queueBubble");
  if (!bubble) return;
  try {
    const saved = JSON.parse(localStorage.getItem("speechDemoQueueBubble") || "null");
    if (!saved) return;
    setQueueBubblePosition(saved.x, saved.y);
  } catch {}
}

function startQueueDrag(event) {
  const bubble = $("#queueBubble");
  if (!bubble) return;
  const rect = bubble.getBoundingClientRect();
  state.queueDrag.active = true;
  state.queueDrag.moved = false;
  state.queueDrag.offsetX = event.clientX - rect.left;
  state.queueDrag.offsetY = event.clientY - rect.top;
  state.queueDrag.startX = event.clientX;
  state.queueDrag.startY = event.clientY;
  bubble.classList.add("dragging");
  bubble.setPointerCapture?.(event.pointerId);
}

function moveQueueDrag(event) {
  if (!state.queueDrag.active) return;
  const dx = event.clientX - state.queueDrag.startX;
  const dy = event.clientY - state.queueDrag.startY;
  if (Math.hypot(dx, dy) > 5) state.queueDrag.moved = true;
  setQueueBubblePosition(event.clientX - state.queueDrag.offsetX, event.clientY - state.queueDrag.offsetY);
}

function endQueueDrag() {
  if (!state.queueDrag.active) return;
  const bubble = $("#queueBubble");
  state.queueDrag.active = false;
  bubble?.classList.remove("dragging");
  if (!bubble) return;
  const rect = bubble.getBoundingClientRect();
  try {
    localStorage.setItem("speechDemoQueueBubble", JSON.stringify({ x: rect.left, y: rect.top }));
  } catch {}
  if (state.queueDrag.moved) {
    setTimeout(() => { state.queueDrag.moved = false; }, 120);
  }
}

function setQueueBubblePosition(x, y) {
  const bubble = $("#queueBubble");
  if (!bubble) return;
  const parent = $("#recordTab") || document.body;
  const parentRect = parent.getBoundingClientRect();
  const size = bubble.offsetWidth || 58;
  const minX = parentRect.left + 10;
  const minY = parentRect.top + 10;
  const maxX = parentRect.right - size - 10;
  const maxY = parentRect.bottom - size - 10;
  const nextX = clamp(x, minX, Math.max(minX, maxX));
  const nextY = clamp(y, minY, Math.max(minY, maxY));
  bubble.style.left = `${nextX - parentRect.left}px`;
  bubble.style.top = `${nextY - parentRect.top}px`;
  bubble.style.right = "auto";
}

function renderTaskGrid() {
  const groups = [{ emotion: "none", label: "None", icon: FREE_TASK.icon, subtitle: "Free record" }, ...TASK_GROUPS.map((group) => ({ ...group, subtitle: `${group.prompts.length} prompts` }))];
  $("#taskGrid").innerHTML = groups.map((group) => `
    <button class="task-card ${group.emotion === "none" ? "free-task" : ""}" type="button" data-emotion="${group.emotion}">
      <span class="task-icon"><i data-lucide="${group.icon}"></i></span>
      <span><strong>${group.label}</strong><small>${group.subtitle}</small></span>
    </button>
  `).join("");
  $$("#taskGrid .task-card").forEach((button) => {
    button.addEventListener("click", () => {
      playTone("click");
      selectTask(buildTask(button.dataset.emotion, 0));
    });
  });
  initIcons();
}

function selectTask(task) {
  const previousEmotion = state.selectedEmotion;
  state.task = task || FREE_TASK;
  state.selectedEmotion = state.task.emotion;
  state.selectedPromptIndex = state.task.promptIndex || 0;
  $("#taskTitle").textContent = task.title;
  $("#targetBadge").textContent = task.emotion === "none" ? "free" : task.emotion;
  $("#scriptText").textContent = task.script;
  if ($("#taskHintA")) $("#taskHintA").textContent = task.hints[0];
  if ($("#taskHintB")) $("#taskHintB").textContent = task.hints[1];
  if ($("#taskHintC")) $("#taskHintC").textContent = task.hints[2];
  $$("#taskGrid .task-card").forEach((button) => button.classList.toggle("active", button.dataset.emotion === task.emotion));
  renderPromptOptions();
  $("#recHint").textContent = task.emotion === "none" ? "Record any presentation-style sentence." : `Record with a ${task.emotion} delivery target.`;
  animatePromptChange(previousEmotion !== state.selectedEmotion);
}

function renderPromptOptions() {
  const container = $("#promptOptions");
  if (!container) return;
  if (state.selectedEmotion === "none") {
    container.innerHTML = `<button class="prompt-card active" type="button">Free speaking mode</button>`;
    return;
  }
  const group = TASK_GROUPS.find((item) => item.emotion === state.selectedEmotion);
  container.innerHTML = group.prompts.map((prompt, index) => `
    <button class="prompt-card ${index === state.selectedPromptIndex ? "active" : ""}" type="button" data-index="${index}">
      ${index + 1}. ${prompt.title}
    </button>
  `).join("");
  $$("#promptOptions .prompt-card").forEach((button) => {
    button.addEventListener("click", () => {
      playTone("click");
      selectTask(buildTask(state.selectedEmotion, Number(button.dataset.index || 0)));
    });
  });
}

function animatePromptChange(isEmotionSwitch = true) {
  const promptPanel = $(".studio-prompt");
  const activeTask = $(`#taskGrid .task-card[data-emotion="${state.selectedEmotion}"]`);
  if (promptPanel) {
    promptPanel.classList.remove("prompt-switching");
    void promptPanel.offsetWidth;
    promptPanel.classList.add("prompt-switching");
    setTimeout(() => promptPanel.classList.remove("prompt-switching"), 420);
  }
  if (activeTask) {
    activeTask.classList.remove("emotion-pop");
    void activeTask.offsetWidth;
    activeTask.classList.add("emotion-pop");
    setTimeout(() => activeTask.classList.remove("emotion-pop"), 520);
  }
  if (window.gsap && isEmotionSwitch) {
    gsap.fromTo(["#scriptText", "#promptOptions"], { opacity: 0, y: 8, filter: "blur(3px)" }, { opacity: 1, y: 0, filter: "blur(0px)", duration: 0.34, stagger: 0.04, ease: "power3.out" });
  }
}

function renderScriptTimeline(selector, timeline, currentTime = 0) {
  const container = $(selector);
  if (!container) return;
  container.innerHTML = timeline.map((entry, index) => {
    const start = Array.isArray(entry) ? entry[0] : entry.start;
    const end = Array.isArray(entry) ? entry[1] : entry.end;
    const text = Array.isArray(entry) ? entry[2] : entry.text;
    return `
    <div class="timeline-row" data-index="${index}" data-start="${start}" data-end="${end}">
      <div class="timecode"><strong>${formatTimecode(start)}</strong><span>${formatTimecode(end)}</span></div>
      <p>${escapeHtml(text)}</p>
      <i></i>
    </div>
  `;
  }).join("");
  updateScriptTimeline(selector, timeline, currentTime);
}

function updateScriptTimeline(selector, timeline, currentTime) {
  if (selector === "#scriptTimeline" && $("#timelineClock")) $("#timelineClock").textContent = `${currentTime.toFixed(1)}s`;
  $(`${selector}`)?.querySelectorAll(".timeline-row").forEach((row, index) => {
    const entry = timeline[index] || [0, 0, ""];
    const start = Array.isArray(entry) ? entry[0] : entry.start;
    const end = Array.isArray(entry) ? entry[1] : entry.end;
    const active = currentTime >= start && currentTime < end;
    const done = currentTime >= end;
    const progress = active ? clamp((currentTime - start) / Math.max(0.1, end - start), 0, 1) : done ? 1 : 0;
    row.classList.toggle("active", active);
    row.classList.toggle("done", done);
    row.style.setProperty("--line-progress", `${progress * 100}%`);
  });
}

function buildRecorderTimeline(item) {
  if (item.sttSegments?.length) {
    return item.sttSegments.map((segment) => ({
      start: segment.start,
      end: segment.end,
      text: segment.text
    }));
  }
  const sentences = splitSentences(item.task.script);
  const duration = Math.max(0.1, item.duration || RECORD_SECONDS);
  const step = duration / Math.max(1, sentences.length);
  return sentences.map((text, index) => ({
    start: index * step,
    end: Math.min(duration, (index + 1) * step),
    text
  }));
}

function resetSttTranscript(message = "Browser STT transcript will appear here with approximate timestamps.") {
  state.sttSegments = [];
  state.sttDraft = "";
  state.sttSegmentStart = 0;
  $("#sttStatus").textContent = "Idle";
  $("#sttTranscript").innerHTML = message;
}

function renderSttTranscript(selector, segments, draft = "") {
  const container = $(selector);
  if (!container) return;
  if (!segments.length && !draft) {
    container.innerHTML = "No recognized speech yet.";
    return;
  }
  const rows = segments.map((segment) => `
    <div class="stt-line">
      <span>${formatTimecode(segment.start)} - ${formatTimecode(segment.end)}</span>
      <p>${escapeHtml(segment.text)}</p>
    </div>
  `);
  if (draft) {
    rows.push(`
      <div class="stt-line">
        <span>live</span>
        <p>${escapeHtml(draft)}</p>
      </div>
    `);
  }
  container.innerHTML = rows.join("");
  container.scrollTop = container.scrollHeight;
}

function startSpeechRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  resetSttTranscript("Listening for speech-to-text transcript...");
  if (!SpeechRecognition) {
    $("#sttStatus").textContent = "Unavailable";
    $("#sttTranscript").innerHTML = "This browser does not support Web Speech API. Audio recording still works; the app will use recording duration to build an approximate timeline.";
    return;
  }
  try {
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.continuous = true;
    recognition.interimResults = true;
    state.recognition = recognition;
    state.sttSegmentStart = 0;
    recognition.onstart = () => {
      $("#sttStatus").textContent = "Listening";
    };
    recognition.onresult = (event) => {
      const elapsed = (performance.now() - state.startedAt) / 1000;
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        const text = result[0]?.transcript?.trim();
        if (!text) continue;
        if (result.isFinal) {
          state.sttSegments.push({
            start: state.sttSegmentStart,
            end: Math.min(RECORD_SECONDS, elapsed),
            text
          });
          state.sttSegmentStart = Math.min(RECORD_SECONDS, elapsed);
          state.sttDraft = "";
        } else {
          state.sttDraft = text;
        }
      }
      renderSttTranscript("#sttTranscript", state.sttSegments, state.sttDraft);
    };
    recognition.onerror = () => {
      $("#sttStatus").textContent = "STT issue";
    };
    recognition.onend = () => {
      if (state.isRecording) {
        try { recognition.start(); } catch {}
      } else {
        $("#sttStatus").textContent = state.sttSegments.length ? "Captured" : "Stopped";
      }
    };
    recognition.start();
  } catch (error) {
    console.warn(error);
    $("#sttStatus").textContent = "Unavailable";
  }
}

function stopSpeechRecognition() {
  if (!state.recognition) return;
  try {
    state.recognition.onend = null;
    state.recognition.stop();
  } catch {}
  state.recognition = null;
  if (state.sttDraft) {
    const elapsed = Math.min(RECORD_SECONDS, (performance.now() - state.startedAt) / 1000);
    state.sttSegments.push({
      start: state.sttSegmentStart,
      end: elapsed,
      text: state.sttDraft
    });
    state.sttDraft = "";
  }
  $("#sttStatus").textContent = state.sttSegments.length ? "Captured" : "Stopped";
  renderSttTranscript("#sttTranscript", state.sttSegments);
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    showToast("Microphone chá»‰ cháº¡y khi má»Ÿ báº±ng http://localhost:5174 hoáº·c HTTPS.");
    return;
  }
  try {
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
    });
    state.audioContext = new AudioContext();
    const source = state.audioContext.createMediaStreamSource(state.stream);
    state.liveAnalyser = state.audioContext.createAnalyser();
    state.liveAnalyser.fftSize = 2048;
    state.liveAnalyser.smoothingTimeConstant = 0.72;
    source.connect(state.liveAnalyser);

    state.chunks = [];
    state.currentRecording = null;
    state.liveWaveHistory = [];
    $("#waveDurationLabel").textContent = `0.0s / ${RECORD_SECONDS.toFixed(1)}s`;
    resetSttTranscript("Listening for speech-to-text transcript...");
    state.mediaRecorder = new MediaRecorder(state.stream);
    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) state.chunks.push(event.data);
    };
    state.mediaRecorder.onstop = handleRecordingStop;
    state.mediaRecorder.start();

    state.isRecording = true;
    state.startedAt = performance.now();
    $("#recordBtn").style.setProperty("--record-progress", "0%");
    $("#recordBtn").classList.add("recording");
    $("#recordIcon").setAttribute("data-lucide", "square");
    $("#recordLabel").textContent = "Stop";
    $("#recordStatus").textContent = "Recording";
    $("#globalStatus").textContent = "Recording voice";
    $(".status-dot").classList.add("active");
    $(".timer").classList.remove("over-limit");
    $("#timerNote").textContent = "/ 00:12.00";
    $("#orbLabel").textContent = "Live";
    $("#addQueueBtn").disabled = true;
    $("#playBtn").disabled = true;
    $("#recHint").textContent = "Recording active. Read the current prompt naturally.";
    initIcons();
    playTone("start");
    startSpeechRecognition();
    animateRecording();
  } catch (error) {
    console.error(error);
    showToast("KhÃ´ng má»Ÿ Ä‘Æ°á»£c microphone. HÃ£y dÃ¹ng localhost vÃ  cáº¥p quyá»n mic.");
  }
}

function stopRecording() {
  if (!state.isRecording) return;
  state.isRecording = false;
  cancelAnimationFrame(state.animationId);
  if (state.mediaRecorder?.state !== "inactive") state.mediaRecorder.stop();
  stopSpeechRecognition();
  state.stream?.getTracks().forEach((track) => track.stop());
  $("#recordBtn").classList.remove("recording");
  $("#recordIcon").setAttribute("data-lucide", "circle");
  $("#recordLabel").textContent = "Record";
  $("#recordStatus").textContent = "Decoding";
  $("#globalStatus").textContent = "Preparing queue item";
  $(".status-dot").classList.remove("active");
  $("#orbLabel").textContent = "Decode";
  initIcons();
  playTone("stop");
}

async function handleRecordingStop() {
  try {
    const blob = new Blob(state.chunks, { type: state.mediaRecorder.mimeType || "audio/webm" });
    const audioUrl = URL.createObjectURL(blob);
    setPlaybackSource(audioUrl);
    const arrayBuffer = await blob.arrayBuffer();
    const decodeContext = new AudioContext();
    const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer.slice(0));
    const samples = new Float32Array(audioBuffer.getChannelData(0));
    state.currentRecording = {
      id: `draft-${Date.now()}`,
      task: structuredCloneTask(state.task),
      blob,
      audioUrl,
      samples,
      sampleRate: audioBuffer.sampleRate,
      duration: audioBuffer.duration,
      sttSegments: state.sttSegments.map((segment) => ({ ...segment })),
      createdAt: new Date()
    };
    $("#recordStatus").textContent = "ASR running";
    $("#globalStatus").textContent = "Recognizing recorded speech";
    if (!$(".timer").classList.contains("over-limit")) $("#timerNote").textContent = "captured / 00:12.00";
    $("#recHint").textContent = "Recording saved. Local Whisper ASR is extracting the transcript...";
    $("#addQueueBtn").disabled = true;
    $("#playBtn").disabled = false;
    drawSpeechWaveform($("#liveWaveCanvas"), samples, audioBuffer.sampleRate, audioBuffer.duration, state.currentRecording.sttSegments, audioBuffer.duration, state.currentRecording.task.script);
    $("#waveDurationLabel").textContent = `${audioBuffer.duration.toFixed(1)}s / ${RECORD_SECONDS.toFixed(1)}s`;
    await decodeContext.close();
    await state.audioContext?.close();
    await hydrateAudioTranscript(state.currentRecording, {
      label: "recorded audio",
      script: state.currentRecording.task.script || "",
      successHint: "Transcript recognized from recorded audio. Add it to queue when ready.",
      emptyHint: "ASR finished, but no confident transcript segment was returned. You can still add the audio to queue.",
    });
    $("#recordStatus").textContent = "Ready to queue";
    $("#globalStatus").textContent = "Audio recorded";
    $("#addQueueBtn").disabled = false;
    showToast("Voice recorded and transcript updated. Add it to queue, then open Processing.");
    playTone("queueReady");
  } catch (error) {
    console.error(error);
    showToast("Decode audio lá»—i. Thá»­ ghi láº¡i báº±ng Chrome hoáº·c Edge.");
  }
}

async function handleAudioUpload(event) {
  const file = event.target.files?.[0];
  event.target.value = "";
  if (!file) return;
  if (state.isRecording) stopRecording();
  try {
    $("#recordStatus").textContent = "Loading file";
    $("#globalStatus").textContent = "Decoding uploaded audio";
    $("#recHint").textContent = "Decoding uploaded audio file...";
    const blob = file.slice(0, file.size, file.type || "audio/wav");
    const audioUrl = URL.createObjectURL(blob);
    setPlaybackSource(audioUrl);
    const arrayBuffer = await blob.arrayBuffer();
    const decodeContext = new AudioContext();
    const audioBuffer = await decodeContext.decodeAudioData(arrayBuffer.slice(0));
    const samples = new Float32Array(audioBuffer.getChannelData(0));
    state.sttSegments = [];
    state.currentRecording = {
      id: `draft-upload-${Date.now()}`,
      task: {
        emotion: "none",
        title: `Uploaded audio: ${file.name}`,
        icon: "upload",
        script: "",
        hints: []
      },
      blob,
      audioUrl,
      samples,
      sampleRate: audioBuffer.sampleRate,
      duration: audioBuffer.duration,
      sttSegments: [],
      createdAt: new Date(),
      source: "uploaded",
      fileName: file.name
    };
    if (getSelectedModelProfile()?.runner === "03c_text_tuned") {
      state.selectedModelProfileId = "03d_live_03b_03c_10fold";
      renderModelProfileSelector();
      renderProcessSelection();
      showToast("Audio-only upload will use the fusion profile so the model can hear the voice, not just the words.");
    }
    $("#sttStatus").textContent = "ASR running";
    renderSttTranscript("#sttTranscript", [], "Recognizing speech from uploaded audio...");
    $("#recordStatus").textContent = "Ready to queue";
    $("#globalStatus").textContent = "Audio file loaded";
    $("#timerText").textContent = formatRecordClock(Math.min(audioBuffer.duration, RECORD_SECONDS));
    $("#timerNote").textContent = `uploaded / ${formatDuration(audioBuffer.duration)}`;
    $("#recHint").textContent = "Audio file decoded. Add it to queue to process.";
    $("#addQueueBtn").disabled = false;
    $("#playBtn").disabled = false;
    drawSpeechWaveform($("#liveWaveCanvas"), samples, audioBuffer.sampleRate, audioBuffer.duration, state.currentRecording.sttSegments, audioBuffer.duration, "");
    $("#waveDurationLabel").textContent = `${audioBuffer.duration.toFixed(1)}s uploaded`;
    await decodeContext.close();
    await hydrateAudioTranscript(state.currentRecording, {
      label: "uploaded audio",
      script: "",
      successHint: "Transcript recognized from uploaded audio. Add it to queue when ready.",
      emptyHint: "ASR finished, but no confident transcript segment was returned."
    });
    showToast("Audio file loaded. Add it to queue, then open Processing.");
    playTone("queueReady");
  } catch (error) {
    console.error(error);
    $("#recordStatus").textContent = "Ready";
    $("#globalStatus").textContent = "Upload failed";
    $("#recHint").textContent = "Cannot decode this audio file. Try WAV, MP3, M4A, WebM or OGG.";
    showToast("KhÃ´ng Ä‘á»c Ä‘Æ°á»£c file audio nÃ y. Thá»­ WAV/MP3/M4A/WebM/OGG.");
  }
}

async function hydrateAudioTranscript(item, options = {}) {
  if (!item?.samples?.length) return;
  const label = options.label || "audio";
  const targetScript = options.script || item.task?.script || "";
  $("#sttStatus").textContent = "ASR running";
  renderSttTranscript("#sttTranscript", item.sttSegments || [], `Recognizing speech from ${label}...`);
  try {
    const asrResult = await requestBackendTranscript(item);
    const transcriptText = asrResult?.transcript?.text || "";
    const transcriptSegments = Array.isArray(asrResult?.transcript?.segments) && asrResult.transcript.segments.length
      ? asrResult.transcript.segments.map((segment) => ({
          start: Number(segment.start || 0),
          end: Number(segment.end || item.duration || RECORD_SECONDS),
          text: String(segment.text || "").trim()
        })).filter((segment) => segment.text)
      : (transcriptText ? buildUploadedAudioTimeline(item.duration, transcriptText, item.fileName) : []);
    item.sttSegments = transcriptSegments;
    item.transcriptText = transcriptText;
    state.sttSegments = transcriptSegments.map((segment) => ({ ...segment }));
    $("#sttStatus").textContent = transcriptSegments.length ? "ASR ready" : "ASR done";
    renderSttTranscript(
      "#sttTranscript",
      transcriptSegments,
      transcriptSegments.length ? "" : (transcriptText || "No transcript recognized.")
    );
    drawSpeechWaveform($("#liveWaveCanvas"), item.samples, item.sampleRate, item.duration, item.sttSegments, item.duration, targetScript);
    $("#recHint").textContent = transcriptSegments.length
      ? (options.successHint || "Transcript recognized. Add it to queue when ready.")
      : (options.emptyHint || "ASR finished, but no confident transcript segment was returned.");
  } catch (error) {
    console.error(error);
    $("#sttStatus").textContent = "ASR fallback";
    renderSttTranscript("#sttTranscript", [], "ASR is unavailable right now. You can still queue the audio and analyze later.");
    $("#recHint").textContent = "Audio decoded. Transcript will be filled again when you analyze the queue.";
  }
}

async function requestBackendTranscript(item) {
  if (!item?.samples?.length) return null;
  const payload = {
    sessionId: item.id,
    audioWavBase64: encodeWavBase64(item.samples, item.sampleRate || 48000),
    sampleRate: item.sampleRate || 48000,
    duration: item.duration || 0
  };
  const response = await fetch("/api/transcribe-audio", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok) throw new Error(`ASR request failed with HTTP ${response.status}`);
  return response.json();
}

function buildUploadedAudioTimeline(duration, script = "", fileName = "uploaded audio") {
  const text = (script || "").trim();
  if (!text) {
    return [{ start: 0, end: duration || RECORD_SECONDS, text: fileName }];
  }
  const sentences = text
    .split(/(?<=[.!?])\s+/)
    .map((line) => line.trim())
    .filter(Boolean);
  const parts = sentences.length ? sentences : [text];
  const span = (duration || RECORD_SECONDS) / parts.length;
  return parts.map((line, index) => ({
    start: index * span,
    end: Math.min(duration || RECORD_SECONDS, (index + 1) * span),
    text: line
  }));
}

function animateRecording() {
  const waveCanvas = $("#liveWaveCanvas");
  const waveCtx = waveCanvas.getContext("2d");
  const timeData = new Uint8Array(state.liveAnalyser.fftSize);
  const freqData = new Uint8Array(state.liveAnalyser.frequencyBinCount);
  const render = () => {
    const elapsed = (performance.now() - state.startedAt) / 1000;
    $("#timerText").textContent = formatRecordClock(elapsed);
    $("#recordBtn").style.setProperty("--record-progress", `${clamp(elapsed / RECORD_SECONDS, 0, 1) * 100}%`);
    state.liveAnalyser.getByteTimeDomainData(timeData);
    state.liveAnalyser.getByteFrequencyData(freqData);
    const level = computeByteRms(timeData);
    updateLiveWaveHistory(timeData, elapsed);
    drawLiveWave(waveCtx, waveCanvas, state.liveWaveHistory, elapsed, state.sttSegments, state.task.script);
    $("#waveDurationLabel").textContent = `${Math.min(elapsed, RECORD_SECONDS).toFixed(1)}s / ${RECORD_SECONDS.toFixed(1)}s`;
    drawOrb(level, freqData);
    updateLedMeter(level, freqData);
    if (elapsed >= RECORD_SECONDS) {
      $(".timer").classList.add("over-limit");
      $("#timerText").textContent = formatRecordClock(RECORD_SECONDS);
      $("#recordBtn").style.setProperty("--record-progress", "100%");
      $("#timerNote").textContent = "over 00:12.00";
      showToast("Recording reached 12 seconds and stopped automatically.");
      playTone("warning");
      stopRecording();
      return;
    }
    if (state.isRecording) state.animationId = requestAnimationFrame(render);
  };
  render();
}

function addCurrentRecordingToQueue() {
  if (!state.currentRecording) return;
  const profile = getSelectedModelProfile();
  const item = {
    ...state.currentRecording,
    id: `session-${Date.now()}`,
    status: "recorded",
    selectedModelProfileId: profile?.id || state.selectedModelProfileId,
    selectedModelProfile: profile || null,
    report: null
  };
  state.queue.unshift(item);
  state.currentRecording = null;
  $("#addQueueBtn").disabled = true;
  $("#recordStatus").textContent = "Queued";
  $("#globalStatus").textContent = "Queue ready";
  renderQueue();
  renderProcessQueueSelector();
  renderProcessSelection();
  resetPipeline();
  unlockTab("process");
  openQueueDrawer();
  showToast("Added to queue. Choose it in Processing when you are ready to analyze.");
  playTone("queue");
}

async function loadModelProfiles() {
  try {
    const response = await fetch("/api/model-profiles");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const profiles = data?.registry?.profiles;
    if (Array.isArray(profiles) && profiles.length) {
      state.modelProfiles = profiles;
      state.selectedModelProfileId = data.registry.defaultProfileId || state.selectedModelProfileId || profiles[0].id;
    }
  } catch (_error) {
    state.modelProfiles = FALLBACK_MODEL_PROFILES;
  }
  renderModelProfileSelector();
  renderProcessSelection();
  renderProcessQueueSelector();
}

function getSelectedModelProfile(id = state.selectedModelProfileId) {
  return state.modelProfiles.find((profile) => profile.id === id) || state.modelProfiles[0] || null;
}

function modelUsesText(profile = getSelectedModelProfile()) {
  if (!profile) return true;
  if (profile.runner === "03c_text_tuned" || profile.runner === "03d_live_weighted_03b_03c") return true;
  if (profile.runner === "03b_bridge") {
    const mode = profile.fusionMode || (profile.id || "").toLowerCase();
    return !String(mode).includes("acoustic_only");
  }
  return true;
}

function modelUsesAcoustic(profile = getSelectedModelProfile()) {
  return profile?.runner !== "03c_text_tuned";
}

function modelUsesBridgeFusion(profile = getSelectedModelProfile()) {
  if (!profile) return false;
  if (profile.runner === "03d_live_weighted_03b_03c") return true;
  const mode = profile.fusionMode || "";
  return String(mode).includes("bridge");
}

function formatModelProfileLabel(profile) {
  if (!profile) return "No model selected";
  const status = profile.status === "ready" ? "ready" : "reference only";
  return `${profile.shortLabel || profile.label} (${status})`;
}

function renderModelProfileSummary(profile) {
  if (!profile) return "No model profile loaded.";
  const statusText = profile.status === "ready"
    ? "Live inference enabled"
    : "Reference result only; standalone live runner is not enabled yet";
  const summary = profile.summary || {};
  const metrics = [
    ["WA", summary.WA],
    ["UAR", summary.UAR],
    ["Macro-F1", summary.Macro_F1],
    ["CCC", summary.CCC_mean]
  ].filter(([, value]) => value);
  const metricHtml = metrics.length
    ? `<span class="model-metric-row">${metrics.map(([label, value]) => `<em><b>${escapeHtml(label)}</b> ${escapeHtml(String(value))}</em>`).join("")}</span>`
    : `<span class="model-metric-row muted">No paper-style metric summary is registered for this profile.</span>`;
  const branchText = [
    modelUsesAcoustic(profile) ? "acoustic" : null,
    modelUsesText(profile) ? "text" : null
  ].filter(Boolean).join(" + ") || "none";
  return `
    <span class="model-profile-line">${escapeHtml(statusText)}. ${escapeHtml(profile.shortLabel || profile.label)}.</span>
    <span class="model-profile-line">Protocol: ${escapeHtml(profile.protocol || "not specified")} | Active branch: ${escapeHtml(branchText)}.</span>
    ${metricHtml}
  `;
}

function renderModelProfileSelector() {
  const select = $("#modelProfileSelect");
  if (!select) return;
  const profiles = state.modelProfiles.length ? state.modelProfiles : FALLBACK_MODEL_PROFILES;
  if (!profiles.some((profile) => profile.id === state.selectedModelProfileId)) {
    state.selectedModelProfileId = profiles[0]?.id || "03d_live_03b_03c_5fold";
  }
  select.innerHTML = profiles.map((profile) => `
    <option value="${escapeHtml(profile.id)}" ${profile.status !== "ready" ? "disabled" : ""}>
      ${escapeHtml(formatModelProfileLabel(profile))}
    </option>
  `).join("");
  select.value = state.selectedModelProfileId;
  const profile = getSelectedModelProfile();
  const summary = $("#modelProfileSummary");
  if (summary) summary.innerHTML = renderModelProfileSummary(profile);
}

function resetCurrentRecording() {
  state.currentRecording = null;
  state.chunks = [];
  state.liveWaveHistory = [];
  $("#playbackAudio").removeAttribute("src");
  $("#playBtn").disabled = true;
  $("#addQueueBtn").disabled = true;
  $("#timerText").textContent = "00:00.00";
  $("#recordBtn").style.setProperty("--record-progress", "0%");
  $(".timer").classList.remove("over-limit");
  $("#timerNote").textContent = "/ 00:12.00";
  $("#recordStatus").textContent = "Ready";
  $("#globalStatus").textContent = "Ready";
  $("#orbLabel").textContent = "Idle";
  $("#recHint").textContent = "Select a task, then record your voice.";
  resetSttTranscript();
  initLedMeter();
  drawEmptyWaveformCanvas($("#liveWaveCanvas"), "idle");
  $("#waveDurationLabel").textContent = `0.0s / ${RECORD_SECONDS.toFixed(1)}s`;
  drawOrb(0, new Uint8Array(512));
  playTone("reset");
}

function renderQueue() {
  $("#queueCount").textContent = `${state.queue.length} item${state.queue.length === 1 ? "" : "s"}`;
  $("#queueBubbleCount").textContent = state.queue.length;
  const container = $("#recordQueue");
  if (!state.queue.length) {
    container.innerHTML = `<div class="empty-state">No audio in queue yet.<br>Choose a task or None, record, then save the take here.</div>`;
    return;
  }
  container.innerHTML = state.queue.map((item) => `
    <button class="queue-item ${item.id === state.selectedQueueId ? "active" : ""}" type="button" data-id="${item.id}">
      <strong>${formatTaskLabel(item.task)}</strong>
      <span>${formatDuration(item.duration)} | ${formatClock(item.createdAt)} | STT ${item.sttSegments?.length || 0}</span>
      <span>${escapeHtml(item.report?.modelProfile?.shortLabel || item.selectedModelProfile?.shortLabel || "No model run yet")}</span>
      <small>${item.status}</small>
    </button>
  `).join("");
  $$("#recordQueue .queue-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedQueueId = button.dataset.id;
      renderQueue();
      renderProcessQueueSelector();
      renderProcessSelection();
      resetPipeline();
      unlockTab("process");
      playTone("click");
    });
  });
}

function renderProcessQueueSelector() {
  const select = $("#processQueueSelect");
  const summary = $("#processQueueSummary");
  if (!select) return;
  select.innerHTML = `<option value="">Choose a queue item</option>` + state.queue.map((item, index) => `
    <option value="${escapeHtml(item.id)}">
      ${index + 1}. ${escapeHtml(formatTaskLabel(item.task))} | ${formatDuration(item.duration)} | ${escapeHtml(item.status)}
    </option>
  `).join("");
  select.value = state.selectedQueueId || "";
  if (summary) {
    const item = getSelectedItem();
    summary.textContent = item
      ? `Selected ${formatTaskLabel(item.task)}. Click Analyze when you want to run the model.`
      : "Choose a recorded/uploaded audio item before analysis.";
  }
}

function getSelectedItem() {
  return state.queue.find((item) => item.id === state.selectedQueueId) || null;
}

function renderProcessSelection() {
  const item = getSelectedItem();
  if (!item) {
    $("#processTitle").textContent = "No queue item";
    $("#processScript").textContent = "Record a voice sample first, then load it here.";
    $("#processMeta").innerHTML = "";
    $("#analyzeBtn").disabled = true;
    renderProcessInputPreview(null);
    renderProcessQueueSelector();
    return;
  }
  $("#processTitle").textContent = formatTaskLabel(item.task);
  $("#processScript").textContent = item.task.script;
  const profile = getSelectedModelProfile();
  $("#processMeta").innerHTML = `
    <div><span>Duration</span><strong>${formatDuration(item.duration)}</strong></div>
    <div><span>Sample rate</span><strong>${item.sampleRate} Hz</strong></div>
    <div><span>Status</span><strong>${item.status}</strong></div>
    <div><span>STT lines</span><strong>${item.sttSegments?.length || 0}</strong></div>
    <div><span>Model</span><strong>${escapeHtml(profile?.shortLabel || profile?.label || "No model")}</strong></div>
  `;
  $("#analyzeBtn").disabled = profile?.status !== "ready";
  renderProcessQueueSelector();
  renderProcessInputPreview(item);
}

function renderProcessInputPreview(item) {
  const thumb = $("#processWaveThumb");
  if (!item) {
    drawEmptyWaveformCanvas(thumb, "empty");
    setText("#processAudioCaption", "No queue item loaded");
    setText("#processTranscriptPreview", "script + timestamp");
    setText("#processTranscriptCaption", "Speech-to-text lines appear after recording.");
    setText("#textTokenSummary", "No transcript token has been loaded yet.");
    return;
  }
  drawSpeechWaveform(thumb, item.samples, item.sampleRate, item.duration, [], item.duration, "");
  setText("#processAudioCaption", `${formatDuration(item.duration)} | ${item.sampleRate} Hz | ${formatTaskLabel(item.task)}`);
  const transcript = item.sttSegments?.length
    ? item.sttSegments.map((segment) => segment.text).join(" ")
    : "";
  const targetScript = item.task?.script || "";
  const profile = item.selectedModelProfile || getSelectedModelProfile();
  const needsText = modelUsesText(profile);
  const transcriptPreview = transcript || targetScript || (needsText
    ? "No browser transcript yet. Local Whisper ASR will transcribe this audio during model inference."
    : "Selected model is acoustic-only, so transcript is not required.");
  setText("#processTranscriptPreview", transcriptPreview);
  setText("#processTranscriptCaption", transcript
    ? `${item.sttSegments?.length || 0} browser STT line(s) | backend ASR can refine missing text`
    : (needsText
      ? (targetScript ? "Target script shown; backend ASR will fill recognized transcript" : "Backend ASR fills missing transcript")
      : "Transcript branch disabled for this acoustic-only profile"));
  setText("#textTokenSummary", transcript
    ? `Transcript length: ${transcript.split(/\s+/).filter(Boolean).length} words. The text encoder produces contextual sequence tokens for self-attention.`
    : (needsText
      ? `Script length: ${targetScript.split(/\s+/).filter(Boolean).length} words. Local Whisper ASR will replace this with recognized transcript when available.`
      : "Text branch is disabled. This profile uses only waveform-derived acoustic features."));
}

async function analyzeSelectedQueue() {
  const item = getSelectedItem();
  if (!item) return;
  const selectedProfile = getSelectedModelProfile();
  if (!selectedProfile || selectedProfile.status !== "ready") {
    showToast("This model profile is for report comparison only. Select a ready live model profile.");
    return;
  }
  resetPipeline();
  startPipelineTimer();
  $("#processStatus").textContent = "Running";
  $("#globalStatus").textContent = "Analyzing queued audio";
  $("#analyzeBtn").disabled = true;
  playTone("analyze");
  const needsAcoustic = modelUsesAcoustic(selectedProfile);
  const needsText = modelUsesText(selectedProfile);

  await runStep("load", 420);
  drawProcessWave(item);
  await runStep("wave", 520);
  const report = analyzeAudio(item.samples, item.sampleRate);
  report.duration = item.duration;
  report.targetEmotion = item.task.emotion === "none" ? null : item.task.emotion;
  report.script = item.task.script;
  report.recognizedTranscript = item.sttSegments || [];
  report.scriptTimeline = buildRecorderTimeline(item);

  if (needsAcoustic) {
    await runStep("features", 600);
    drawHeatmap($("#processMelCanvas"), report.logMel, "Log-Mel");
    renderProcessNumbers(report);
    await Promise.all([
      runStep("temporal", 520),
      runStep("spectral", 520),
      runStep("pretrained", 520),
      runStep("stats", 520)
    ]);
    await runStep("attention", 420);
  } else {
    $("#processStatus").textContent = "Text-only profile selected; acoustic feature branches are skipped";
    setText("#flowFeatureSummary", "Skipped: selected model uses transcript branch only.");
  }

  const needsBackendAsr = needsText && !item.sttSegments?.length;
  $("#processStatus").textContent = needsBackendAsr
    ? "Waiting for backend ASR transcript"
    : (needsText ? "Running selected model backend" : "Running acoustic-only backend");
  if (needsBackendAsr) {
    setText("#processTranscriptPreview", "Running local ASR before final fusion...");
    setText("#processTranscriptCaption", "Analyze is waiting here until transcript is ready.");
    setText("#textTokenSummary", "Text branch is paused until ASR returns recognized words.");
  } else if (!needsText) {
    setText("#processTranscriptPreview", "Transcript is not used by the selected acoustic-only profile.");
    setText("#processTranscriptCaption", "ASR skipped.");
    setText("#textTokenSummary", "Text branch skipped: acoustic-only inference.");
  }
  const modelResult = await runTimedWork("backend", () => request03BModelInference(item, report));
  if (!modelResult) {
    showToast("Selected model backend did not return a prediction. This run uses browser fallback, so the result is not the real trained model.");
  }
  report.modelSource = modelResult?.source || "browser-feature fallback";
  report.modelProfile = modelResult?.modelProfile || selectedProfile;
  report.modelProfileId = report.modelProfile?.id || selectedProfile.id;
  report.modelDebug = modelResult?.debug || null;
  report.modelAggregation = modelResult?.aggregation || null;
  report.modelSegments = modelResult?.segments || [];
  report.modelTranscript = modelResult?.transcript || null;
  if ((!report.recognizedTranscript || !report.recognizedTranscript.length) && modelResult?.transcript?.segments?.length) {
    report.recognizedTranscript = modelResult.transcript.segments.map((segment) => ({
      start: segment.start,
      end: segment.end,
      text: segment.text
    }));
    item.sttSegments = report.recognizedTranscript;
  }
  if (needsText && modelResult?.transcript?.text) {
    setText("#processTranscriptPreview", modelResult.transcript.text);
    setText("#processTranscriptCaption", `${modelResult.transcript.segments?.length || 1} ASR segment(s) | ${modelResult.transcript.source || "local ASR"}`);
    setText("#textTokenSummary", `ASR transcript length: ${modelResult.transcript.text.split(/\s+/).filter(Boolean).length} words. RoBERTa tokenizes this text for the selected text branch.`);
    renderProcessInputPreview(item);
  }
  if (needsText) {
    $("#processStatus").textContent = needsAcoustic
      ? "Transcript ready; running text branch and final fusion"
      : "Transcript ready; running text-only branch";
    await runStep("text", 360);
    await runStep("text_attention", 340);
  }
  report.prediction = modelResult?.prediction || predictAffect(report, report.targetEmotion);
  setText("#flowOutputSummary", `${capitalize(report.prediction.emotion)} | V ${formatVadScore(report.prediction.valence)} A ${formatVadScore(report.prediction.arousal)} D ${formatVadScore(report.prediction.dominance)}`);
  if (needsAcoustic && needsText) {
    await runStep("fusion", 520);
    if (modelUsesBridgeFusion(selectedProfile)) await runStep("bridge", 420);
  }
  await runStep("heads", 380);
  item.status = "processed";
  item.modelProfileUsed = report.modelProfile;
  item.modelProfileId = report.modelProfileId;
  item.report = report;
  state.selectedResultId = item.id;
  await runStep("report", 360);

  $("#processStatus").textContent = "Complete";
  $("#globalStatus").textContent = "Report ready";
  $("#analyzeBtn").disabled = false;
  stopPipelineTimer();
  renderQueue();
  renderResultHistory();
  renderSelectedResult();
  unlockTab("results");
  playTone("success");
  setTimeout(() => playTone(`emotion-${report.prediction.emotion}`), 180);
  showToast("Analysis complete. Open Results to review emotion, VAD and transcript.");
}

function startPipelineTimer() {
  stopPipelineTimer(false);
  state.pipelineTimers.totalStart = performance.now();
  const update = () => {
    setText("#processTotalTimer", `Total ${((performance.now() - state.pipelineTimers.totalStart) / 1000).toFixed(1)}s`);
  };
  update();
  state.pipelineTimers.totalInterval = setInterval(update, 100);
}

function stopPipelineTimer(keepFinal = true) {
  if (state.pipelineTimers.totalInterval) {
    clearInterval(state.pipelineTimers.totalInterval);
    state.pipelineTimers.totalInterval = null;
  }
  state.pipelineTimers.stepIntervals.forEach((interval) => clearInterval(interval));
  state.pipelineTimers.stepIntervals.clear();
  if (!keepFinal) setText("#processTotalTimer", "Total 0.0s");
}

async function runTimedWork(stepName, work) {
  const steps = $$(`[data-step="${stepName}"]`);
  if (stepName === "backend") {
    steps.push(...$$(`[data-node-id="bridge"], [data-node-id="output"]`));
  }
  startStepTimer(stepName, steps);
  try {
    return await work();
  } finally {
    finishStepTimer(stepName, steps);
  }
}

async function request03BModelInference(item, report) {
  if (!item?.samples?.length) return null;
  const profile = getSelectedModelProfile();
  const needsText = modelUsesText(profile);
  const transcript = item.sttSegments?.length
    ? item.sttSegments.map((segment) => segment.text).join(" ")
    : "";
  const payload = {
    sessionId: item.id,
    modelProfileId: profile?.id || state.selectedModelProfileId,
    audioWavBase64: encodeWavBase64(item.samples, item.sampleRate || 48000),
    transcript,
    autoTranscribe: needsText,
    transcriptSource: item.sttSegments?.length ? "browser_stt" : (needsText ? "backend_asr_required" : "not_required_acoustic_only"),
    targetEmotion: item.task?.emotion || "none",
    sampleRate: item.sampleRate || 48000,
    duration: item.duration || 0,
    clientFeatureSummary: {
    frameCount: report.frames.length,
    logMelShape: [report.logMel.length, report.logMel[0]?.length || 0],
    mfccShape: [report.mfcc.length, report.mfcc[0]?.length || 0],
    stats: report.stats
    }
  };

  try {
    const response = await fetch("/api/analyze-03b", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (!response.ok) return null;
    const data = await response.json();
    return normalize03BPrediction(data);
  } catch (error) {
    return null;
  }
}

function encodeWavBase64(samples, sampleRate) {
  const wavBuffer = encodeWavPcm16(samples, sampleRate);
  const bytes = new Uint8Array(wavBuffer);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function encodeWavPcm16(samples, sampleRate) {
  const channels = 1;
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const value = clamp(samples[i] || 0, -1, 1);
    view.setInt16(offset, value < 0 ? value * 0x8000 : value * 0x7fff, true);
    offset += 2;
  }
  return buffer;
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
}

function normalize03BPrediction(data) {
  const raw = data?.prediction || data;
  if (!raw) return null;
  const probabilities = raw.probabilities || raw.emotion_probs || raw.emotionProbabilities;
  const normalizedProbabilities = {};
  if (probabilities) {
    EMOTIONS.forEach((emotion, index) => {
      normalizedProbabilities[emotion] = Array.isArray(probabilities)
        ? Number(probabilities[index] || 0)
        : Number(probabilities[emotion] || 0);
    });
  }
  const total = Object.values(normalizedProbabilities).reduce((sum, value) => sum + value, 0);
  if (total > 0) {
    EMOTIONS.forEach((emotion) => {
      normalizedProbabilities[emotion] = clamp(normalizedProbabilities[emotion] / total, 0, 1);
    });
  }
  const emotion = raw.emotion || EMOTIONS.reduce((best, current) => (normalizedProbabilities[current] || 0) > (normalizedProbabilities[best] || 0) ? current : best, EMOTIONS[0]);
  const vad = raw.vad || {};
  const prediction = {
    emotion,
    probabilities: Object.keys(normalizedProbabilities).length ? normalizedProbabilities : Object.fromEntries(EMOTIONS.map((name) => [name, name === emotion ? 1 : 0])),
    valence: clamp(Number(raw.valence ?? vad.valence ?? vad[0] ?? 0.5), 0, 1),
    arousal: clamp(Number(raw.arousal ?? vad.arousal ?? vad[1] ?? 0.5), 0, 1),
    dominance: clamp(Number(raw.dominance ?? vad.dominance ?? vad[2] ?? 0.5), 0, 1),
    confidence: clamp(Number(raw.confidence ?? normalizedProbabilities[emotion] ?? 0), 0, 1)
  };
  const segments = Array.isArray(data?.segments)
    ? data.segments.map((segment) => ({
      index: Number(segment.index || 0),
      start: Number(segment.start || 0),
      end: Number(segment.end || 0),
      duration: Number(segment.duration || 0),
      diagnostics: Array.isArray(segment.diagnostics) ? segment.diagnostics : [],
      prediction: normalize03BPrediction({ prediction: segment.prediction })?.prediction
    })).filter((segment) => segment.prediction)
    : [];
  return {
    source: data?.source || "03B backend",
    modelProfile: data?.modelProfile || data?.received?.modelProfile || getSelectedModelProfile(data?.received?.modelProfileId),
    prediction,
    segments,
    aggregation: data?.aggregation || null,
    transcript: data?.transcript || null,
    debug: data?.debug || null
  };
}

function resetPipeline() {
  stopPipelineTimer(false);
  $$("[data-step]").forEach((step) => step.classList.remove("active", "done"));
  $$(".node-runtime").forEach((badge) => badge.remove());
  drawEmptyWaveformCanvas($("#processWaveCanvas"), "waiting for selected audio");
  drawEmptySpectrogramCanvas($("#processMelCanvas"), "waiting for feature extraction");
  if ($("#processNumbers")) $("#processNumbers").innerHTML = "";
  setText("#flowOutputSummary", "waiting");
  const item = getSelectedItem();
  renderProcessInputPreview(item || null);
}

async function runStep(stepName, ms) {
  const steps = $$(`[data-step="${stepName}"]`);
  startStepTimer(stepName, steps);
  playTone(stepName === "report" ? "success" : "tick");
  await delay(ms);
  finishStepTimer(stepName, steps);
}

function startStepTimer(stepName, steps) {
  const uniqueSteps = Array.from(new Set(steps)).filter(Boolean);
  uniqueSteps.forEach((step) => {
    step.classList.remove("done");
    step.classList.add("active");
  });
  const started = performance.now();
  const update = () => {
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    uniqueSteps.forEach((step) => setNodeRuntime(step, `${elapsed}s`));
  };
  update();
  if (state.pipelineTimers.stepIntervals.has(stepName)) {
    clearInterval(state.pipelineTimers.stepIntervals.get(stepName));
  }
  state.pipelineTimers.stepIntervals.set(stepName, setInterval(update, 100));
}

function finishStepTimer(stepName, steps) {
  const interval = state.pipelineTimers.stepIntervals.get(stepName);
  if (interval) {
    clearInterval(interval);
    state.pipelineTimers.stepIntervals.delete(stepName);
  }
  Array.from(new Set(steps)).filter(Boolean).forEach((step) => {
    step.classList.remove("active");
    step.classList.add("done");
  });
}

function setNodeRuntime(node, value) {
  const container = node.classList?.contains("explainer-node") ? node : node.closest?.(".explainer-node");
  if (!container) return;
  let badge = container.querySelector(":scope > .node-runtime");
  if (!badge) {
    badge = document.createElement("span");
    badge.className = "node-runtime";
    container.appendChild(badge);
  }
  badge.textContent = value;
}

function renderProcessNumbers(report) {
  const s = report.stats;
  const frameCount = report.frames.length;
  const melBins = report.logMel[0]?.length || 0;
  const mfccDim = report.mfcc[0]?.length || 0;
  setText("#temporalShape", `[B, ${mfccDim * 3 || 39}+LLD, ${frameCount}]`);
  setText("#spectralShape", `[B, 3, ${melBins || 40}, ${frameCount}]`);
  setText("#statsShape", `RMS ${s.rmsMean.toFixed(4)} | ZCR ${s.zcrMean.toFixed(4)} | centroid ${Math.round(s.centroidMean)} Hz`);
  if (!$("#processNumbers")) return;
  $("#processNumbers").innerHTML = `
    <div><span>Frames</span><strong>${report.frames.length}</strong></div>
    <div><span>Voiced</span><strong>${pct(s.voicedRatio)}</strong></div>
    <div><span>RMS</span><strong>${s.rmsMean.toFixed(4)}</strong></div>
    <div><span>ZCR</span><strong>${s.zcrMean.toFixed(4)}</strong></div>
    <div><span>Centroid</span><strong>${Math.round(s.centroidMean)} Hz</strong></div>
    <div><span>Pitch</span><strong>${s.pitchMean ? Math.round(s.pitchMean) + " Hz" : "--"}</strong></div>
  `;
}

function renderResultHistory() {
  const processed = state.queue.filter((item) => item.report);
  const container = $("#resultHistory");
  if (!processed.length) {
    container.innerHTML = `<div class="empty-state">No processed report yet.<br>Analyze one queue item first.</div>`;
    return;
  }
  container.innerHTML = processed.map((item) => `
    <button class="queue-item ${item.id === state.selectedResultId ? "active" : ""}" type="button" data-id="${item.id}">
      <strong>${capitalize(item.report.prediction.emotion)} result</strong>
      <span>${formatTaskLabel(item.task)} | ${pct(item.report.prediction.confidence)} confidence</span>
      <span>${escapeHtml(getItemModelLabel(item))}</span>
      <small>report</small>
    </button>
  `).join("");
  $$("#resultHistory .queue-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedResultId = button.dataset.id;
      if (state.compareTargetId === state.selectedResultId) state.compareTargetId = null;
      renderResultHistory();
      renderSelectedResult();
      playTone("click");
    });
  });
}

function getSelectedResultItem() {
  return state.queue.find((item) => item.id === state.selectedResultId && item.report) || null;
}

function getItemModelLabel(item) {
  const profile = item?.report?.modelProfile || item?.modelProfileUsed || item?.selectedModelProfile || null;
  return profile?.shortLabel || profile?.label || item?.report?.modelSource || "No model recorded";
}

function renderSelectedResult() {
  const item = getSelectedResultItem();
  if (!item) return;
  const report = item.report;
  if (report?.featureTab) state.resultFeatureTab = report.featureTab;
  if (report?.featureSlides) state.resultFeatureSlides = { ...state.resultFeatureSlides, ...report.featureSlides };
  const pred = report.prediction;
  $("#resultTitle").textContent = `${formatTaskLabel(item.task)} report`;
  $("#predictedEmotion").textContent = capitalize(pred.emotion);
  const sourceText = report.modelSource ? ` Source: ${report.modelSource}. Model: ${getItemModelLabel(item)}.` : ` Model: ${getItemModelLabel(item)}.`;
  $("#predictionSummary").textContent = item.task.emotion === "none"
    ? `Free recording; predicted ${pred.emotion} with ${pct(pred.confidence)} confidence.${sourceText}`
    : `Target ${item.task.emotion}; predicted ${pred.emotion} with ${pct(pred.confidence)} confidence.${sourceText}`;
  $("#valenceValue").textContent = formatVadScore(pred.valence);
  $("#arousalValue").textContent = formatVadScore(pred.arousal);
  $("#dominanceValue").textContent = formatVadScore(pred.dominance);
  updateProbabilityBars(pred.probabilities);
  updateRadar(pred);
  $("#rmsMetric").textContent = report.stats.rmsMean.toFixed(4);
  $("#zcrMetric").textContent = report.stats.zcrMean.toFixed(4);
  $("#centroidMetric").textContent = `${Math.round(report.stats.centroidMean)} Hz`;
  $("#pitchMetric").textContent = report.stats.pitchMean ? `${Math.round(report.stats.pitchMean)} Hz` : "--";
  renderScriptTimeline("#resultTranscript", report.scriptTimeline || buildRecorderTimeline(item), (report.duration || RECORD_SECONDS) + 1);
  renderSttTranscript("#resultSttTranscript", report.recognizedTranscript || []);
  renderSegmentDiagnostics(report);
  renderFeatureEvidence(item);
  renderCompareOptions();
  renderComparePanel();
  $("#playResultBtn").disabled = !item.audioUrl;
  $("#downloadResultBtn").disabled = !item.blob;
  $("#exportReportBtn").disabled = false;
  $("#exportReportPackageBtn").disabled = false;
  $("#downloadBtn").disabled = false;
}

function renderFeatureEvidence(item) {
  const report = item?.report;
  const grid = $("#featureEvidenceGrid");
  if (!report || !grid) return;
  const shapes = report.modelDebug?.feature_shapes || report.modelDebug?.featureShapes || {};
  const s = report.stats || {};
  const asShape = (value, fallback) => Array.isArray(value) ? value.join(" x ") : fallback;
  const rows = [
    ["Waveform", `${formatDuration(item.duration)} | ${item.sampleRate} Hz | mono`, "wave"],
    ["X_temporal", asShape(shapes.X_temporal, `${report.mfcc?.[0]?.length || 13} MFCC-like dims + LLD x ${report.frames?.length || 0} frames`), "temporal"],
    ["X_spectral", asShape(shapes.X_spectral, `${report.logMel?.[0]?.length || 40} Mel bins x ${report.logMel?.length || 0} frames`), "spectral"],
    ["X_stats", asShape(shapes.X_stats, `${Object.keys(s).length} browser summary values`), "stats"],
    ["X_e2v", asShape(shapes.X_e2v, "768-d pretrained speech embedding"), "e2v"],
    ["Transcript", `${(report.modelTranscript?.text || report.recognizedTranscript?.map((x) => x.text).join(" ") || "").split(/\s+/).filter(Boolean).length} words`, "transcript"],
    ["Feature trends", `RMS ${Number(s.rmsMean || 0).toFixed(4)} | ZCR ${Number(s.zcrMean || 0).toFixed(4)} | pitch ${s.pitchMean ? Math.round(s.pitchMean) + " Hz" : "--"}`, "feature_trends"],
    ["Model profile", getItemModelLabel(item), "output"]
  ];
  grid.innerHTML = rows.map(([label, value, inspect]) => `
    <button type="button" class="feature-evidence-tile" data-result-inspect="${escapeHtml(inspect)}">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value || "--"))}</strong>
      <em>Click to inspect</em>
    </button>
  `).join("");
  renderResultFeatureTabs();
  drawResultFeatureCanvases(item);
  renderFeatureMiniTables(item);
}

function renderResultFeatureTabs() {
  const active = state.resultFeatureTab || "raw";
  $$("[data-result-feature-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.resultFeatureTab === active);
  });
  $$("[data-result-feature-panel]").forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.resultFeaturePanel === active);
  });
  renderFeatureSlideDeck();
}

function stepResultFeatureSlide(direction) {
  const active = state.resultFeatureTab || "raw";
  const panel = $(`[data-result-feature-panel="${active}"]`);
  const cards = panel ? Array.from(panel.querySelectorAll(".feature-preview-card")) : [];
  if (!cards.length) return;
  const current = state.resultFeatureSlides[active] || 0;
  state.resultFeatureSlides[active] = (current + direction + cards.length) % cards.length;
  const item = getSelectedResultItem();
  if (item?.report) item.report.featureSlides = { ...state.resultFeatureSlides };
  renderFeatureSlideDeck();
  playTone("tab");
}

function renderFeatureSlideDeck() {
  const active = state.resultFeatureTab || "raw";
  const panel = $(`[data-result-feature-panel="${active}"]`);
  const cards = panel ? Array.from(panel.querySelectorAll(".feature-preview-card")) : [];
  if (!cards.length) {
    return;
  }
  let current = state.resultFeatureSlides[active] || 0;
  if (current >= cards.length) current = 0;
  state.resultFeatureSlides[active] = current;
  const item = getSelectedResultItem();
  if (item?.report) item.report.featureSlides = { ...state.resultFeatureSlides };
  const previous = (current - 1 + cards.length) % cards.length;
  const next = (current + 1) % cards.length;
  cards.forEach((card, index) => {
    card.classList.remove("slide-active", "slide-prev", "slide-next", "slide-hidden");
    if (index === current) card.classList.add("slide-active");
    else if (cards.length > 1 && index === previous) card.classList.add("slide-prev");
    else if (cards.length > 2 && index === next) card.classList.add("slide-next");
    else card.classList.add("slide-hidden");
  });
  cards.forEach((card, index) => {
    card.dataset.slidePosition = index === current ? "active" : (index === previous ? "previous" : (index === next ? "next" : "hidden"));
    card.setAttribute("aria-label", index === current ? "Inspect selected feature" : "Select this feature slide");
  });
}

function drawResultFeatureCanvases(item) {
  if (!item?.report) return;
  const report = item.report;
  drawSpeechWaveform($("#resultWaveCanvas"), item.samples, item.sampleRate, item.duration, resolvedTranscriptSegments(item, report), item.duration, "");
  drawHeatmap($("#resultMfccCanvas"), report.mfcc, "MFCC / cepstral sequence", { yLabel: "Coeff bin", xLabel: "Time frames" });
  drawFeatureTrendCanvas($("#resultFeatureTrendCanvas"), report, "Frame-level 1D acoustic trends");
  drawLineFeatureCanvas($("#resultEnergyZcrCanvas"), report, [
    { key: "rms", label: "RMS energy", color: "#2d6cff" },
    { key: "zcr", label: "Zero-crossing rate", color: "#df4f73" }
  ], "Energy and voicing boundary signals");
  drawLineFeatureCanvas($("#resultPitchCentroidCanvas"), report, [
    { key: "pitch", label: "Pitch/F0", color: "#0a9f72" },
    { key: "centroid", label: "Spectral centroid", color: "#f59f00" }
  ], "Prosody and spectral brightness");
  drawHeatmap($("#resultMelCanvas"), report.logMel, "Log-Mel spectrogram", { yLabel: "Mel bin", xLabel: "Time frames" });
  drawHeatmap($("#resultDeltaMelCanvas"), deltaMatrix(report.logMel), "Delta log-Mel", { yLabel: "Mel bin", xLabel: "Time frames" });
  drawHeatmap($("#resultDeltaDeltaMelCanvas"), deltaMatrix(deltaMatrix(report.logMel)), "Delta-delta log-Mel", { yLabel: "Mel bin", xLabel: "Time frames" });
  drawHeatmap($("#resultMfcc2dCanvas"), report.mfcc, "MFCC heatmap", { yLabel: "Coeff bin", xLabel: "Time frames" });
  drawStatsSummaryCanvas($("#resultStatsCanvas"), report);
  drawEmbeddingSummaryCanvas($("#resultE2vCanvas"), report);
  drawProbabilityCanvas($("#resultProbabilityCanvas"), report.prediction);
  drawTranscriptEvidenceCanvas($("#resultTranscriptCanvas"), item, report);
}

function renderFeatureMiniTables(item) {
  if (!item?.report) return;
  const report = item.report;
  const shapes = report.modelDebug?.feature_shapes || report.modelDebug?.featureShapes || {};
  const stats = report.stats || {};
  const shapeText = (value, fallback = "--") => Array.isArray(value) ? value.join(" x ") : fallback;
  const transcriptText = report.modelTranscript?.text || resolvedTranscriptSegments(item, report).map((segment) => segment.text).join(" ");
  fillFeatureTable("#resultRawFeatureTable", [
    ["Duration", formatDuration(item.duration)],
    ["Sample rate", `${item.sampleRate} Hz`],
    ["Frames", `${report.frames?.length || 0}`],
    ["Transcript words", `${transcriptText.split(/\s+/).filter(Boolean).length}`],
    ["Model used", getItemModelLabel(item)]
  ]);
  fillFeatureTable("#resultTemporalFeatureTable", [
    ["X_temporal shape", shapeText(shapes.X_temporal, `${report.frames?.length || 0} frames x temporal descriptors`)],
    ["MFCC", `${report.mfcc?.[0]?.length || 0} cepstral coefficients per frame`],
    ["Delta MFCC", "first-order temporal change"],
    ["Delta-delta MFCC", "second-order temporal acceleration"],
    ["RMS energy", Number(stats.rmsMean || 0).toFixed(4)],
    ["ZCR", Number(stats.zcrMean || 0).toFixed(4)],
    ["Pitch/F0", stats.pitchMean ? `${Math.round(stats.pitchMean)} Hz` : "--"],
    ["Spectral LLD", `centroid ${Math.round(stats.centroidMean || 0)} Hz, flux ${Number(stats.fluxMean || 0).toFixed(4)}`]
  ]);
  fillFeatureTable("#resultSpectralFeatureTable", [
    ["X_spectral shape", shapeText(shapes.X_spectral, `${report.logMel?.length || 0} frames x ${report.logMel?.[0]?.length || 0} Mel bins`)],
    ["Log-Mel", "time-frequency energy map"],
    ["Delta log-Mel", "frame-to-frame spectral movement"],
    ["Delta-delta log-Mel", "spectral acceleration approximation"],
    ["Mel bins", `${report.logMel?.[0]?.length || 0}`],
    ["Time frames", `${report.logMel?.length || 0}`]
  ]);
  fillFeatureTable("#resultStatsFeatureTable", [
    ["X_stats shape", shapeText(shapes.X_stats, `${Object.keys(stats).length} summary values`)],
    ["X_e2v shape", shapeText(shapes.X_e2v, "768-d pretrained speech embedding")],
    ["Probability head", "emotion classifier softmax"],
    ["VAD head", "valence, arousal, dominance regression"],
    ["Transcript branch", `${transcriptText.split(/\s+/).filter(Boolean).length} words before tokenizer`],
    ["Fusion", report.modelAggregation?.method ? report.modelAggregation.method.replaceAll("_", " ") : "selected model output"]
  ]);
}

function fillFeatureTable(selector, rows) {
  const node = $(selector);
  if (!node) return;
  node.innerHTML = rows.map(([label, value]) => `
    <div>
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value ?? "--"))}</strong>
    </div>
  `).join("");
}

function renderCompareOptions() {
  const select = $("#compareTargetSelect");
  if (!select) return;
  const current = getSelectedResultItem();
  const processed = state.queue.filter((item) => item.report && item.id !== current?.id);
  const previous = state.compareTargetId;
  select.innerHTML = `<option value="">Select another processed queue</option>` + processed.map((item) => `
    <option value="${item.id}">${escapeHtml(formatTaskLabel(item.task))} - ${escapeHtml(capitalize(item.report.prediction.emotion))} - ${escapeHtml(getItemModelLabel(item))}</option>
  `).join("");
  if (processed.some((item) => item.id === previous)) {
    select.value = previous;
  } else {
    state.compareTargetId = processed[0]?.id || null;
    select.value = state.compareTargetId || "";
  }
}

function renderComparePanel() {
  const base = getSelectedResultItem();
  const target = state.queue.find((item) => item.id === state.compareTargetId && item.report);
  const summary = $("#compareSummary");
  if (!summary) return;
  if (!base || !target) {
    summary.classList.add("empty-state");
    summary.innerHTML = "Analyze at least two queue items to compare their waveform, log-Mel and feature statistics.";
    ["#compareWaveA", "#compareWaveB", "#compareMelA", "#compareMelB"].forEach((selector) => drawEmptyCanvas($(selector), "waiting"));
    return;
  }
  summary.classList.remove("empty-state");
  const rows = buildComparisonRows(base, target);
  summary.innerHTML = `
    <div class="compare-title-line">
      <strong>A: ${escapeHtml(formatTaskLabel(base.task))} | ${escapeHtml(getItemModelLabel(base))}</strong>
      <strong>B: ${escapeHtml(formatTaskLabel(target.task))} | ${escapeHtml(getItemModelLabel(target))}</strong>
    </div>
    <div class="compare-table">
      ${rows.map((row) => `
        <div>
          <span>${escapeHtml(row.label)}</span>
          <strong>${escapeHtml(row.a)}</strong>
          <strong>${escapeHtml(row.b)}</strong>
          <em>${escapeHtml(row.delta)}</em>
        </div>
      `).join("")}
    </div>
  `;
  drawSpeechWaveform($("#compareWaveA"), base.samples, base.sampleRate, base.duration, base.report.recognizedTranscript || base.sttSegments || [], base.duration, base.report.script || "");
  drawSpeechWaveform($("#compareWaveB"), target.samples, target.sampleRate, target.duration, target.report.recognizedTranscript || target.sttSegments || [], target.duration, target.report.script || "");
  drawHeatmap($("#compareMelA"), base.report.logMel, "A log-Mel");
  drawHeatmap($("#compareMelB"), target.report.logMel, "B log-Mel");
}

function buildComparisonRows(aItem, bItem) {
  const a = aItem.report;
  const b = bItem.report;
  const numericRows = [
    ["Duration", aItem.duration, bItem.duration, (x) => `${Number(x || 0).toFixed(2)}s`],
    ["Confidence", a.prediction.confidence, b.prediction.confidence, pct],
    ["Valence", a.prediction.valence, b.prediction.valence, formatVadScore],
    ["Arousal", a.prediction.arousal, b.prediction.arousal, formatVadScore],
    ["Dominance", a.prediction.dominance, b.prediction.dominance, formatVadScore],
    ["RMS mean", a.stats.rmsMean, b.stats.rmsMean, (x) => Number(x || 0).toFixed(4)],
    ["ZCR mean", a.stats.zcrMean, b.stats.zcrMean, (x) => Number(x || 0).toFixed(4)],
    ["Pitch mean", a.stats.pitchMean || 0, b.stats.pitchMean || 0, (x) => x ? `${Math.round(x)} Hz` : "--"],
    ["Pause ratio", a.stats.pauseRatio, b.stats.pauseRatio, pct],
    ["Centroid", a.stats.centroidMean, b.stats.centroidMean, (x) => `${Math.round(x || 0)} Hz`]
  ];
  const rows = [{
    label: "Model profile",
    a: getItemModelLabel(aItem),
    b: getItemModelLabel(bItem),
    delta: getItemModelLabel(aItem) === getItemModelLabel(bItem) ? "same" : "different"
  }, {
    label: "Emotion",
    a: capitalize(a.prediction.emotion),
    b: capitalize(b.prediction.emotion),
    delta: a.prediction.emotion === b.prediction.emotion ? "same" : "different"
  }];
  const ppLabels = new Set(["Confidence", "Pause ratio"]);
  numericRows.forEach(([label, av, bv, fmt]) => {
    const delta = Number(bv || 0) - Number(av || 0);
    const deltaText = ppLabels.has(label)
      ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(1)} pp`
      : `${delta >= 0 ? "+" : ""}${delta.toFixed(label === "Duration" ? 2 : 4)}`;
    rows.push({
      label,
      a: fmt(av),
      b: fmt(bv),
      delta: deltaText
    });
  });
  return rows;
}

function renderSegmentDiagnostics(report) {
  const container = $("#segmentDiagnostics");
  if (!container) return;
  const segments = report.modelSegments || [];
  if (!segments.length) {
    container.innerHTML = "";
    return;
  }
  const aggregation = report.modelAggregation;
  const method = aggregation?.method ? aggregation.method.replaceAll("_", " ") : "duration weighted aggregation";
    container.innerHTML = `
      <div class="segment-head">
        <div>
          <p class="label">Acoustic segment diagnostics</p>
          <h3>${segments.length} audio segment(s) analyzed</h3>
        </div>
        <span>${escapeHtml(method)} | final summary may include text fusion</span>
      </div>
    <div class="segment-table">
      ${segments.map((segment) => {
        const p = segment.prediction;
        const notes = segment.diagnostics?.length ? segment.diagnostics.join(", ") : "stable";
        return `
          <div class="segment-row">
            <strong>${formatDuration(segment.start)}-${formatDuration(segment.end)}</strong>
            <span>${capitalize(p.emotion)}</span>
            <small>${pct(p.confidence)}</small>
            <div>
              <div class="segment-vad">
                <span>V ${formatVadScore(p.valence)}</span>
                <span>A ${formatVadScore(p.arousal)}</span>
                <span>D ${formatVadScore(p.dominance)}</span>
              </div>
              <small class="segment-note">${escapeHtml(notes)}</small>
            </div>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function renderProbabilityBars() {
  $("#probabilityBars").innerHTML = EMOTIONS.map((emotion) => `
    <div class="prob-row">
      <span>${capitalize(emotion)}</span>
      <div class="prob-track"><div class="prob-fill" id="prob-${emotion}" style="background:${EMOTION_META[emotion].color}"></div></div>
      <strong id="prob-label-${emotion}">0%</strong>
    </div>
  `).join("");
}

function updateProbabilityBars(probabilities) {
  EMOTIONS.forEach((emotion) => {
    const value = probabilities?.[emotion] || 0;
    $(`#prob-${emotion}`).style.width = `${value * 100}%`;
    $(`#prob-label-${emotion}`).textContent = pct(value);
  });
}

function initRadar() {
  if (!window.Chart) return;
  const ctx = $("#vadRadarCanvas").getContext("2d");
  state.radarChart = new Chart(ctx, {
    type: "radar",
    data: {
      labels: ["Valence", "Arousal", "Dominance"],
      datasets: [{ data: [0, 0, 0], fill: true, borderColor: "#2d6cff", backgroundColor: "rgba(45,108,255,0.18)", pointBackgroundColor: "#2d6cff" }]
    },
    options: {
      plugins: { legend: { display: false } },
      scales: { r: { min: 0, max: 1, ticks: { display: false }, grid: { color: "rgba(99,116,145,0.16)" }, angleLines: { color: "rgba(99,116,145,0.14)" } } }
    }
  });
}

function updateRadar(prediction) {
  if (!state.radarChart) return;
  state.radarChart.data.datasets[0].data = [prediction.valence, prediction.arousal, prediction.dominance];
  state.radarChart.update();
}

function exportSelectedReport() {
  const item = getSelectedResultItem();
  if (!item) return;
  playTone("success");
  const blob = new Blob([JSON.stringify(buildReportPayload(item), null, 2)], { type: "application/json" });
  downloadBlob(blob, `speech-emotion-report-${item.id}.json`);
}

function exportSelectedReportPackage() {
  const item = getSelectedResultItem();
  if (!item?.report) return;
  const compareItem = state.queue.find((queueItem) => queueItem.id === state.compareTargetId && queueItem.report);
  const payload = buildReportPayload(item);
  const assets = buildReportAssets(item);
  const compareAssets = compareItem ? buildReportAssets(compareItem) : null;
  const compareRows = compareItem ? buildComparisonRows(item, compareItem) : [];
  const transcriptRows = payload.recognizedTranscript || [];
  const featureRows = buildFeatureShapeRows(item);
  const compareBlock = compareItem ? `
    <section>
      <h2>So sÃ¡nh vá»›i queue B</h2>
      <p><strong>A:</strong> ${escapeHtml(formatTaskLabel(item.task))} | ${escapeHtml(getItemModelLabel(item))}</p>
      <p><strong>B:</strong> ${escapeHtml(formatTaskLabel(compareItem.task))} | ${escapeHtml(getItemModelLabel(compareItem))}</p>
      <table>
        <thead><tr><th>TÃ­n hiá»‡u</th><th>Audio A</th><th>Audio B</th><th>ChÃªnh lá»‡ch</th></tr></thead>
        <tbody>${compareRows.map((row) => `<tr><td>${escapeHtml(row.label)}</td><td>${escapeHtml(row.a)}</td><td>${escapeHtml(row.b)}</td><td>${escapeHtml(row.delta)}</td></tr>`).join("")}</tbody>
      </table>
      <div class="image-grid">
        <figure><img src="${assets.waveformPng}" alt="Waveform A"><figcaption>Waveform A</figcaption></figure>
        <figure><img src="${compareAssets.waveformPng}" alt="Waveform B"><figcaption>Waveform B</figcaption></figure>
        <figure><img src="${assets.logMelPng}" alt="Log-Mel A"><figcaption>Log-Mel A</figcaption></figure>
        <figure><img src="${compareAssets.logMelPng}" alt="Log-Mel B"><figcaption>Log-Mel B</figcaption></figure>
      </div>
    </section>
  ` : "";
  const html = `<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Speech emotion report - ${escapeHtml(item.id)}</title>
  <style>
    :root { color-scheme: light; font-family: Inter, Segoe UI, Arial, sans-serif; color: #172238; background: #f5f8fc; }
    body { margin: 0; padding: 28px; }
    main { max-width: 1180px; margin: 0 auto; display: grid; gap: 18px; }
    section { padding: 18px; border: 1px solid #dce6f4; border-radius: 12px; background: rgba(255,255,255,.92); box-shadow: 0 12px 36px rgba(30,49,78,.08); }
    h1, h2, h3, p { margin-top: 0; }
    h1 { font-size: 28px; margin-bottom: 6px; }
    h2 { font-size: 18px; margin-bottom: 12px; }
    p, li, td, th { font-size: 13px; line-height: 1.5; }
    .summary { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; }
    .card { padding: 12px; border: 1px solid #e1e9f5; border-radius: 10px; background: #f9fbff; }
    .card span { display: block; color: #64748b; font-size: 11px; font-weight: 800; text-transform: uppercase; }
    .card strong { display: block; margin-top: 5px; font-size: 20px; color: #0f1c33; }
    .image-grid { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 12px; }
    figure { margin: 0; padding: 10px; border: 1px solid #e1e9f5; border-radius: 10px; background: #fff; }
    figure img { width: 100%; display: block; border-radius: 8px; border: 1px solid #edf2f8; }
    figcaption { margin-top: 8px; color: #54647a; font-size: 12px; font-weight: 800; }
    table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 10px; }
    th, td { padding: 9px 10px; border-bottom: 1px solid #e8eef7; text-align: left; vertical-align: top; }
    th { color: #41516a; background: #eef4ff; font-size: 11px; text-transform: uppercase; }
    pre { max-height: 420px; overflow: auto; padding: 14px; border-radius: 10px; background: #101a2d; color: #e7eefc; font-size: 12px; }
    audio { width: 100%; }
    @media (max-width: 820px) { body { padding: 14px; } .summary, .image-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Speech Emotion + VAD Report</h1>
      <p>Queue <strong>${escapeHtml(item.id)}</strong> | táº¡o lÃºc ${escapeHtml(new Date().toLocaleString())} | model: ${escapeHtml(payload.modelProfile?.shortLabel || getItemModelLabel(item))} | source: ${escapeHtml(payload.modelSource || "browser/backend")}</p>
      <audio controls src="${assets.audioDataUri}"></audio>
    </section>
    <section>
      <h2>Káº¿t quáº£ chÃ­nh</h2>
      <div class="summary">
        <div class="card"><span>Emotion</span><strong>${escapeHtml(capitalize(payload.prediction.emotion))}</strong></div>
        <div class="card"><span>Confidence</span><strong>${escapeHtml(pct(payload.prediction.confidence))}</strong></div>
        <div class="card"><span>Valence score</span><strong>${escapeHtml(formatVadScore(payload.prediction.valence))}</strong></div>
        <div class="card"><span>Arousal / Dominance score</span><strong>${escapeHtml(formatVadScore(payload.prediction.arousal))} / ${escapeHtml(formatVadScore(payload.prediction.dominance))}</strong></div>
      </div>
    </section>
    <section>
      <h2>Feature evidence</h2>
      <table>
        <thead><tr><th>NhÃ³m Ä‘áº·c trÆ°ng</th><th>Shape / ná»™i dung</th></tr></thead>
        <tbody>${featureRows.map((row) => `<tr><td>${escapeHtml(row.label)}</td><td>${escapeHtml(row.value)}</td></tr>`).join("")}</tbody>
      </table>
      <div class="image-grid" style="margin-top:12px;">
        <figure><img src="${assets.waveformPng}" alt="Waveform"><figcaption>Waveform + transcript markers</figcaption></figure>
        <figure><img src="${assets.logMelPng}" alt="Log-Mel"><figcaption>Log-Mel spectrogram</figcaption></figure>
        <figure><img src="${assets.featureTrendPng}" alt="Feature trends"><figcaption>RMS, pitch, centroid, ZCR trends</figcaption></figure>
      </div>
    </section>
    <section>
      <h2>Transcript / timestamp</h2>
      <p><strong>Target script:</strong> ${escapeHtml(payload.script || item.task?.script || "")}</p>
      <table>
        <thead><tr><th>Start</th><th>End</th><th>Text</th></tr></thead>
        <tbody>${transcriptRows.length ? transcriptRows.map((row) => `<tr><td>${escapeHtml(formatDuration(row.start))}</td><td>${escapeHtml(formatDuration(row.end))}</td><td>${escapeHtml(row.text || "")}</td></tr>`).join("") : `<tr><td colspan="3">ChÆ°a cÃ³ ASR transcript cho queue nÃ y.</td></tr>`}</tbody>
      </table>
    </section>
    ${compareBlock}
    <section>
      <h2>Raw JSON</h2>
      <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
    </section>
  </main>
</body>
</html>`;
  playTone("success");
  downloadBlob(new Blob([html], { type: "text/html;charset=utf-8" }), `speech-emotion-report-package-${item.id}.html`);
}

function buildReportPayload(item) {
  const report = item.report;
  return {
    sessionId: item.id,
    createdAt: item.createdAt,
    task: item.task,
    sampleRate: item.sampleRate,
    duration: item.duration,
    prediction: report.prediction,
    modelSource: report.modelSource,
    modelProfile: report.modelProfile || item.modelProfileUsed || item.selectedModelProfile || null,
    modelProfileId: report.modelProfileId || item.modelProfileId || item.selectedModelProfileId || null,
    modelAggregation: report.modelAggregation,
    modelSegments: report.modelSegments,
    modelTranscript: report.modelTranscript,
    modelDebug: report.modelDebug,
    stats: report.stats,
    script: report.script,
    scriptTimeline: report.scriptTimeline,
    recognizedTranscript: report.recognizedTranscript,
    featureShapes: {
      frames: report.frames?.length || 0,
      logMel: [report.logMel?.length || 0, report.logMel?.[0]?.length || 0],
      mfcc: [report.mfcc?.length || 0, report.mfcc?.[0]?.length || 0]
    },
    framePreview: (report.frames || []).slice(0, 80)
  };
}

function buildFeatureShapeRows(item) {
  const report = item.report || {};
  const shapes = report.modelDebug?.feature_shapes || report.modelDebug?.featureShapes || {};
  const asShape = (value, fallback) => Array.isArray(value) ? value.join(" x ") : fallback;
  const text = report.modelTranscript?.text || report.recognizedTranscript?.map((x) => x.text).join(" ") || "";
  return [
    { label: "X_temporal", value: asShape(shapes.X_temporal, `${report.mfcc?.[0]?.length || 13} MFCC-like dims + LLD x ${report.frames?.length || 0} frames`) },
    { label: "X_spectral", value: asShape(shapes.X_spectral, `${report.logMel?.[0]?.length || 40} Mel bins x ${report.logMel?.length || 0} frames`) },
    { label: "X_stats", value: asShape(shapes.X_stats, `${Object.keys(report.stats || {}).length} summary values`) },
    { label: "X_e2v", value: asShape(shapes.X_e2v, "768-d pretrained speech embedding") },
    { label: "Transcript tokens", value: `${text.split(/\s+/).filter(Boolean).length} words before RoBERTa tokenization` },
    { label: "Model profile", value: getItemModelLabel(item) },
    { label: "Model source", value: report.modelSource || "browser fallback" }
  ];
}

function buildReportAssets(item) {
  const report = item.report || {};
  const waveCanvas = makeExportCanvas(1200, 300);
  const melCanvas = makeExportCanvas(1000, 300);
  const trendCanvas = makeExportCanvas(1200, 320);
  drawSpeechWaveform(waveCanvas, item.samples, item.sampleRate, item.duration, report.recognizedTranscript || item.sttSegments || [], item.duration, report.script || item.task?.script || "");
  drawHeatmap(melCanvas, report.logMel || [], "Log-Mel spectrogram");
  drawFeatureTrendCanvas(trendCanvas, report, "Frame-level feature trends");
  return {
    audioDataUri: `data:audio/wav;base64,${encodeWavBase64(item.samples, item.sampleRate || 48000)}`,
    waveformPng: waveCanvas.toDataURL("image/png"),
    logMelPng: melCanvas.toDataURL("image/png"),
    featureTrendPng: trendCanvas.toDataURL("image/png")
  };
}

function makeExportCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadSelectedAudio() {
  const item = getSelectedResultItem() || getSelectedItem();
  if (!item?.blob) return;
  playTone("play");
  const url = URL.createObjectURL(item.blob);
  const link = document.createElement("a");
  link.href = url;
  const fileName = item.fileName || (item.blob.type?.includes("wav") ? `speech-demo-${item.id}.wav` : `speech-demo-${item.id}.webm`);
  link.download = fileName;
  link.click();
  URL.revokeObjectURL(url);
}

function playSelectedResultAudio() {
  const item = getSelectedResultItem() || getSelectedItem();
  if (!item?.audioUrl) return;
  playAudioUrl(item.audioUrl);
  playTone("play");
}

function setPlaybackSource(audioUrl) {
  const audio = $("#playbackAudio");
  if (!audio || !audioUrl) return;
  try {
    audio.pause();
    audio.removeAttribute("src");
    audio.load();
  } catch {}
  audio.src = audioUrl;
  audio.load();
}

function playAudioUrl(audioUrl) {
  if (!audioUrl) return;
  const audio = $("#playbackAudio");
  setPlaybackSource(audioUrl);
  audio.currentTime = 0;
  audio.play().catch(() => showToast("Cannot play this audio in the browser. Please use the saved file instead."));
}

function analyzeAudio(samples, sampleRate) {
  const frameSize = 1024;
  const hop = 512;
  const window = hann(frameSize);
  const melBank = buildMelFilterBank(sampleRate, frameSize, 40);
  const frames = [];
  const logMel = [];
  const mfcc = [];
  let prevSpectrum = null;
  const frameCount = Math.max(1, Math.floor((samples.length - frameSize) / hop) + 1);

  for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
    const start = frameIndex * hop;
    const frame = new Float32Array(frameSize);
    let rmsSum = 0;
    let zcr = 0;
    for (let i = 0; i < frameSize; i++) {
      const value = samples[start + i] || 0;
      frame[i] = value * window[i];
      rmsSum += value * value;
      if (i > 0) {
        const prev = samples[start + i - 1] || 0;
        if ((value >= 0 && prev < 0) || (value < 0 && prev >= 0)) zcr++;
      }
    }
    const spectrum = magnitudeSpectrum(frame);
    const spectral = spectralFeatures(spectrum, sampleRate, prevSpectrum);
    const mel = applyMelBank(spectrum, melBank);
    const cepstral = dct(mel.map((v) => Math.log(v + 1e-8)), 13);
    const pitch = frameIndex % 2 === 0 ? estimatePitch(samples, start, frameSize, sampleRate) : NaN;
    frames.push({ time: start / sampleRate, rms: Math.sqrt(rmsSum / frameSize), zcr: zcr / frameSize, pitch, ...spectral });
    logMel.push(mel.map((v) => Math.log10(v + 1e-8)));
    mfcc.push(cepstral);
    prevSpectrum = spectrum;
  }

  const rmsValues = frames.map((f) => f.rms);
  const threshold = Math.max(0.010, percentile(rmsValues, 70) * 0.26, mean(rmsValues) * 0.58);
  frames.forEach((frame, index) => {
    frame.pitch = Number.isFinite(frame.pitch) ? frame.pitch : nearestPitch(frames, index);
    frame.voiced = frame.rms > threshold;
  });

  return { sampleRate, frameSize, hop, samples: downsample(samples, 1400), frames, logMel, mfcc, stats: summarizeFrames(frames) };
}

function predictAffect(report, targetEmotion) {
  const s = report.stats;
  const energy = normalize(s.rmsMean, 0.012, 0.10);
  const energyVar = normalize(s.rmsStd, 0.004, 0.055);
  const sharpness = normalize(s.zcrMean, 0.025, 0.16);
  const brightness = normalize(s.centroidMean, 700, 3600);
  const flux = normalize(s.fluxMean, 0.0005, 0.12);
  const pitch = normalize(s.pitchMean || 0, 90, 260);
  const valence = clamp(0.30 + 0.25 * energy + 0.18 * s.stability + 0.12 * pitch - 0.16 * sharpness + (targetEmotion === "happy" ? 0.08 : 0), 0, 1);
  const arousal = clamp(0.18 + 0.34 * energy + 0.20 * energyVar + 0.16 * sharpness + 0.12 * flux, 0, 1);
  const dominance = clamp(0.24 + 0.30 * energy + 0.22 * s.voicedRatio + 0.18 * s.stability + 0.10 * brightness - 0.08 * s.pauseRatio, 0, 1);
  const scores = {
    neutral: 0.48 + s.stability * 0.36 + (1 - arousal) * 0.24,
    happy: 0.22 + valence * 0.54 + energy * 0.22 + pitch * 0.14,
    sad: 0.20 + (1 - valence) * 0.38 + (1 - energy) * 0.30 + s.pauseRatio * 0.12,
    angry: 0.18 + arousal * 0.40 + energy * 0.32 + sharpness * 0.26 + s.contrastMean / 120
  };
  if (targetEmotion && scores[targetEmotion] !== undefined) scores[targetEmotion] += 0.04;
  const probs = softmax(EMOTIONS.map((emotion) => scores[emotion]));
  const probabilities = Object.fromEntries(EMOTIONS.map((emotion, i) => [emotion, probs[i]]));
  const emotion = EMOTIONS.reduce((best, item) => probabilities[item] > probabilities[best] ? item : best, EMOTIONS[0]);
  return { emotion, probabilities, valence, arousal, dominance, confidence: probabilities[emotion], match: emotion === targetEmotion };
}

function summarizeFrames(frames) {
  const by = (key) => frames.map((f) => f[key]);
  const voiced = frames.filter((f) => f.voiced);
  const pitchValues = voiced.map((f) => f.pitch).filter((v) => Number.isFinite(v) && v > 0);
  const rmsMean = mean(by("rms"));
  return {
    rmsMean,
    rmsStd: std(by("rms")),
    zcrMean: mean(by("zcr")),
    centroidMean: mean(by("centroid")),
    bandwidthMean: mean(by("bandwidth")),
    rolloffMean: mean(by("rolloff")),
    fluxMean: mean(by("flux")),
    contrastMean: mean(by("contrast")),
    pitchMean: mean(pitchValues),
    pitchStd: std(pitchValues),
    voicedRatio: frames.length ? voiced.length / frames.length : 0,
    pauseRatio: frames.length ? 1 - voiced.length / frames.length : 1,
    stability: clamp(1 - std(by("rms")) / (rmsMean + 1e-6), 0, 1)
  };
}

function updateLiveWaveHistory(timeData, elapsed) {
  let min = 1;
  let max = -1;
  let sum = 0;
  for (const value of timeData) {
    const amp = (value - 128) / 128;
    min = Math.min(min, amp);
    max = Math.max(max, amp);
    sum += amp * amp;
  }
  state.liveWaveHistory.push({ time: elapsed, min, max, rms: Math.sqrt(sum / timeData.length) });
  state.liveWaveHistory = state.liveWaveHistory.filter((point) => point.time >= elapsed - RECORD_SECONDS);
}

function drawLiveWave(ctx, canvas, history, elapsed = 0, segments = [], fallbackText = "") {
  const duration = RECORD_SECONDS;
  drawWaveScaffold(ctx, canvas, duration, elapsed, "live");
  if (!history.length) return;
  const plot = wavePlotArea(canvas);
  ctx.strokeStyle = "rgba(0, 122, 255, 0.78)";
  ctx.lineWidth = 1.4;
  history.forEach((point) => {
    const x = plot.left + clamp(point.time / duration, 0, 1) * plot.width;
    const upper = plot.mid - clamp(point.max, -1, 1) * plot.amp;
    const lower = plot.mid - clamp(point.min, -1, 1) * plot.amp;
    ctx.beginPath();
    ctx.moveTo(x, upper);
    ctx.lineTo(x, lower);
    ctx.stroke();
  });
}

function drawProcessWave(item) {
  drawSpeechWaveform($("#processWaveCanvas"), item.samples, item.sampleRate, item.duration, item.sttSegments || [], item.duration, item.task?.script || "");
  drawSpeechWaveform($("#processWaveThumb"), item.samples, item.sampleRate, item.duration, [], item.duration, "");
}

function drawSpeechWaveform(canvas, samples, sampleRate, duration, segments = [], markerTime = null, fallbackText = "") {
  if (!canvas || !samples?.length) return;
  const ctx = canvas.getContext("2d");
  drawWaveScaffold(ctx, canvas, duration || samples.length / sampleRate, markerTime, "recorded");
  const plot = wavePlotArea(canvas);
  const columns = Math.max(260, Math.floor(plot.width / 2.2));
  const samplesPerColumn = Math.max(1, Math.floor(samples.length / columns));
  const scale = waveformScale(samples);
  const grad = ctx.createLinearGradient(plot.left, 0, plot.right, 0);
  grad.addColorStop(0, "rgba(0, 113, 200, 0.88)");
  grad.addColorStop(0.55, "rgba(0, 148, 220, 0.92)");
  grad.addColorStop(1, "rgba(0, 92, 170, 0.88)");
  ctx.strokeStyle = grad;
  ctx.lineWidth = 1.25;
  ctx.lineCap = "round";
  for (let col = 0; col < columns; col++) {
    let min = 1;
    let max = -1;
    const start = col * samplesPerColumn;
    const end = Math.min(samples.length, start + samplesPerColumn);
    for (let i = start; i < end; i++) {
      const normalized = clamp((samples[i] || 0) / scale, -1, 1);
      min = Math.min(min, normalized);
      max = Math.max(max, normalized);
    }
    const x = plot.left + (col / Math.max(1, columns - 1)) * plot.width;
    const y1 = plot.mid - max * plot.amp;
    const y2 = plot.mid - min * plot.amp;
    ctx.beginPath();
    ctx.moveTo(x, y1);
    ctx.lineTo(x, y2);
    ctx.stroke();
  }
  drawWordTimeline(ctx, canvas, segments, duration || samples.length / sampleRate, fallbackText);
}

function waveformScale(samples) {
  const stride = Math.max(1, Math.floor(samples.length / 5000));
  const values = [];
  for (let i = 0; i < samples.length; i += stride) values.push(Math.abs(samples[i] || 0));
  const p = percentile(values, 98) || 0.02;
  return Math.max(0.02, p * 1.08);
}

function drawWaveScaffold(ctx, canvas, duration = RECORD_SECONDS, markerTime = 0, label = "live") {
  const { width, height } = canvas;
  const plot = wavePlotArea(canvas);
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(99,116,145,0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 6; i++) {
    const x = plot.left + (i / 6) * plot.width;
    ctx.beginPath();
    ctx.moveTo(x, plot.top);
    ctx.lineTo(x, plot.bottom + 24);
    ctx.stroke();
    ctx.fillStyle = "#344258";
    ctx.font = "800 22px Inter";
    ctx.textAlign = "center";
    ctx.fillText(`${((duration || RECORD_SECONDS) * i / 6).toFixed(1)}s`, x, height - 10);
  }
  [-1, -0.5, 0, 0.5, 1].forEach((amp) => {
    const y = plot.mid - amp * plot.amp;
    ctx.strokeStyle = amp === 0 ? "rgba(19,32,51,0.30)" : "rgba(99,116,145,0.10)";
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
    if (amp === 1 || amp === 0 || amp === -1) {
      ctx.fillStyle = "#65738a";
      ctx.font = "800 18px Inter";
      ctx.textAlign = "right";
      ctx.fillText(amp.toFixed(0), plot.left - 8, y + 4);
    }
  });
  ctx.save();
  ctx.translate(20, plot.mid);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = "rgba(101, 115, 138, 0.62)";
  ctx.font = "500 15px Inter";
  ctx.textAlign = "center";
  ctx.fillText("Amplitude", 0, 0);
  ctx.restore();
  if (markerTime !== null && Number.isFinite(markerTime)) {
    const markerX = plot.left + clamp(markerTime / (duration || RECORD_SECONDS), 0, 1) * plot.width;
    ctx.strokeStyle = markerTime >= RECORD_SECONDS ? "rgba(238,75,98,0.80)" : "rgba(10,159,114,0.62)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(markerX, plot.top);
    ctx.lineTo(markerX, plot.bottom + 18);
    ctx.stroke();
  }
}

function drawWordTimeline(ctx, canvas, segments, duration, fallbackText = "") {
  const plot = wavePlotArea(canvas);
  const safeDuration = Math.max(0.1, duration || RECORD_SECONDS);
  const words = buildWordTimeline(segments, safeDuration, "").slice(0, 10);
  if (!words.length && fallbackText) {
    ctx.fillStyle = "rgba(101,115,138,0.72)";
    ctx.font = "800 14px Inter";
    ctx.textAlign = "left";
    ctx.fillText("No timestamped transcript yet", plot.left, plot.bottom + 32);
    return;
  }
  ctx.textAlign = "center";
  words.forEach((word) => {
    const x1 = plot.left + clamp(word.start / safeDuration, 0, 1) * plot.width;
    const x2 = plot.left + clamp(word.end / safeDuration, 0, 1) * plot.width;
    const x = (x1 + x2) / 2;
    ctx.strokeStyle = "rgba(45,108,255,0.16)";
    ctx.beginPath();
    ctx.moveTo(x, plot.bottom + 2);
    ctx.lineTo(x, plot.bottom + 15);
    ctx.stroke();
    ctx.fillStyle = "#1f2e44";
    ctx.font = "850 12px Inter";
    const text = word.text.length > 8 ? `${word.text.slice(0, 7)}...` : word.text;
    ctx.fillText(text, x, plot.bottom + 29);
  });
}

function buildWordTimeline(segments, duration, fallbackText = "") {
  const cleanSegments = (segments || []).filter((segment) => segment.text?.trim());
  const result = [];
  if (cleanSegments.length) {
    cleanSegments.forEach((segment) => {
      const words = segment.text.trim().split(/\s+/).filter(Boolean);
      const span = Math.max(0.2, segment.end - segment.start);
      words.forEach((word, index) => {
        const start = segment.start + (span * index) / words.length;
        const end = segment.start + (span * (index + 1)) / words.length;
        result.push({ text: word, start, end });
      });
    });
    return result;
  }
  const words = (fallbackText || "").split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const span = Math.max(1, duration || RECORD_SECONDS);
  return words.map((word, index) => ({
    text: word,
    start: (span * index) / words.length,
    end: (span * (index + 1)) / words.length
  }));
}

function wavePlotArea(canvas) {
  const left = 58;
  const rightPad = 20;
  const top = 24;
  const bottomPad = 62;
  const width = canvas.width - left - rightPad;
  const bottom = canvas.height - bottomPad;
  return {
    left,
    right: left + width,
    width,
    top,
    bottom,
    mid: top + (bottom - top) / 2,
    amp: (bottom - top) * 0.46
  };
}

function drawWaveBadge(ctx, canvas, label = "live", seconds = 0) {
  const text = `${label} ${Number(seconds || 0).toFixed(1)}s`;
  ctx.font = "900 22px Inter";
  const badgeWidth = ctx.measureText(text).width + 20;
  fillRound(ctx, canvas.width - badgeWidth - 14, 12, badgeWidth, 26, 999, "rgba(19,32,51,0.78)");
  ctx.fillStyle = "rgba(255,255,255,0.94)";
  ctx.textAlign = "left";
  ctx.fillText(text, canvas.width - badgeWidth - 4, 34);
}

function drawHeatmap(canvas, matrix, label, options = {}) {
  if (!canvas) return;
  if (!matrix?.length) {
    drawEmptyCanvas(canvas, "no feature");
    return;
  }
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  const rows = matrix[0]?.length || 0;
  const cols = matrix.length;
  const pad = { left: 58, right: 18, top: 36, bottom: 42 };
  const plot = {
    left: pad.left,
    right: width - pad.right,
    top: pad.top,
    bottom: height - pad.bottom,
    width: width - pad.left - pad.right,
    height: height - pad.top - pad.bottom
  };
  const flat = flatten(matrix);
  const min = percentile(flat, 2);
  const max = percentile(flat, 98);
  for (let x = 0; x < cols; x++) {
    for (let y = 0; y < rows; y++) {
      const value = normalize(matrix[x][y], min, max);
      ctx.fillStyle = heatColor(value);
      ctx.fillRect(plot.left + (x / cols) * plot.width, plot.bottom - ((y + 1) / rows) * plot.height, Math.ceil(plot.width / cols), Math.ceil(plot.height / rows));
    }
  }
  drawFeatureAxes(ctx, plot, {
    xLabel: options.xLabel || "Time frames",
    yLabel: options.yLabel || "Feature bins",
    xMax: cols,
    yMax: rows
  });
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.fillRect(10, 10, 130, 25);
  ctx.fillStyle = "#132033";
  ctx.font = "900 12px Inter";
  ctx.fillText(label, 18, 27);
}

function drawFeatureAxes(ctx, plot, { xLabel, yLabel, xMax, yMax }) {
  ctx.save();
  ctx.strokeStyle = "rgba(19,32,51,0.35)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(plot.left, plot.top);
  ctx.lineTo(plot.left, plot.bottom);
  ctx.lineTo(plot.right, plot.bottom);
  ctx.stroke();
  ctx.fillStyle = "#53647c";
  ctx.font = "800 11px Inter";
  ctx.textAlign = "center";
  ctx.fillText(xLabel, plot.left + plot.width / 2, plot.bottom + 31);
  ctx.fillText("0", plot.left, plot.bottom + 16);
  ctx.fillText(String(Math.max(0, xMax - 1)), plot.right, plot.bottom + 16);
  ctx.save();
  ctx.translate(18, plot.top + plot.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(yLabel, 0, 0);
  ctx.restore();
  ctx.textAlign = "right";
  ctx.fillText("0", plot.left - 8, plot.bottom + 4);
  ctx.fillText(String(Math.max(0, yMax - 1)), plot.left - 8, plot.top + 4);
  ctx.restore();
}

function drawLineFeatureCanvas(canvas, report, specs, title) {
  if (!canvas) return;
  const frames = report?.frames || [];
  if (!frames.length) {
    drawEmptyCanvas(canvas, "no frame data");
    return;
  }
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const pad = { left: 58, right: 18, top: 38, bottom: 40 };
  const plot = {
    left: pad.left,
    right: width - pad.right,
    top: pad.top,
    bottom: height - pad.bottom,
    width: width - pad.left - pad.right,
    height: height - pad.top - pad.bottom
  };
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  drawChartGrid(ctx, plot, frames.length, 5);
  ctx.fillStyle = "#172238";
  ctx.font = "900 15px Inter";
  ctx.textAlign = "left";
  ctx.fillText(title, plot.left, 24);

  specs.forEach((spec, specIndex) => {
    const values = frames.map((frame) => Number(frame[spec.key] || 0));
    const active = values.filter((value) => Number.isFinite(value) && value > 0);
    const min = spec.key === "rms" || spec.key === "zcr" ? 0 : (percentile(active, 5) || 0);
    let max = percentile(active.length ? active : values, 95) || 1;
    if (max <= min) max = min + 1e-5;
    ctx.strokeStyle = spec.color;
    ctx.lineWidth = 2.4;
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = plot.left + (index / Math.max(1, values.length - 1)) * plot.width;
      const y = plot.bottom - clamp((value - min) / (max - min), 0, 1) * plot.height;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = spec.color;
    ctx.font = "850 11px Inter";
    ctx.fillText(spec.label, plot.left + specIndex * 170, height - 12);
  });
  drawFeatureAxes(ctx, plot, { xLabel: "Time frames", yLabel: "Normalized value", xMax: frames.length, yMax: 1 });
}

function drawChartGrid(ctx, plot, xCount = 1, yCount = 5) {
  for (let i = 0; i <= yCount; i++) {
    const y = plot.top + (i / yCount) * plot.height;
    ctx.strokeStyle = i === yCount ? "rgba(19,32,51,0.22)" : "rgba(99,116,145,0.12)";
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
  }
  for (let i = 0; i <= 6; i++) {
    const x = plot.left + (i / 6) * plot.width;
    ctx.strokeStyle = "rgba(99,116,145,0.10)";
    ctx.beginPath();
    ctx.moveTo(x, plot.top);
    ctx.lineTo(x, plot.bottom);
    ctx.stroke();
  }
}

function drawStatsSummaryCanvas(canvas, report) {
  const stats = report?.stats || {};
  const values = [
    ["RMS", stats.rmsMean || 0, "#2d6cff"],
    ["ZCR", stats.zcrMean || 0, "#df4f73"],
    ["Pause", stats.pauseRatio || 0, "#65718c"],
    ["Flux", stats.fluxMean || 0, "#8b5cf6"]
  ];
  drawBarCanvas(canvas, values, "Statistical functionals from acoustic frames");
}

function drawEmbeddingSummaryCanvas(canvas, report) {
  const shapes = report?.modelDebug?.feature_shapes || report?.modelDebug?.featureShapes || {};
  const e2vShape = Array.isArray(shapes.X_e2v) ? shapes.X_e2v.join(" x ") : "768-d embedding";
  const values = [
    ["E2V dim", 0.82, "#8b5cf6"],
    ["Speech repr.", 0.68, "#2d6cff"],
    ["Adapter", 0.54, "#00aeca"],
    ["Fusion token", 0.46, "#0a9f72"]
  ];
  drawBarCanvas(canvas, values, `Pretrained speech branch: ${e2vShape}`);
}

function drawProbabilityCanvas(canvas, prediction) {
  const probabilities = prediction?.probabilities || {};
  drawBarCanvas(canvas, EMOTIONS.map((emotion) => [capitalize(emotion), probabilities[emotion] || 0, EMOTION_META[emotion].color]), "Emotion classifier probability output");
}

function drawTranscriptEvidenceCanvas(canvas, item, report) {
  const text = report?.modelTranscript?.text || resolvedTranscriptSegments(item, report).map((segment) => segment.text).join(" ");
  const words = text.split(/\s+/).filter(Boolean);
  const values = [
    ["Words", Math.min(1, words.length / 24), "#2d6cff"],
    ["Segments", Math.min(1, (resolvedTranscriptSegments(item, report).length || 0) / 6), "#00aeca"],
    ["Tokenizer", words.length ? 0.72 : 0.08, "#8b5cf6"],
    ["Text branch", words.length ? 0.66 : 0.08, "#0a9f72"]
  ];
  drawBarCanvas(canvas, values, `Transcript evidence: ${words.length} word(s)`);
}

function drawBarCanvas(canvas, rows, title) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const pad = { left: 120, right: 30, top: 42, bottom: 28 };
  const plot = { left: pad.left, right: width - pad.right, top: pad.top, bottom: height - pad.bottom, width: width - pad.left - pad.right, height: height - pad.top - pad.bottom };
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#172238";
  ctx.font = "900 15px Inter";
  ctx.textAlign = "left";
  ctx.fillText(title, 18, 25);
  rows.forEach(([label, value, color], index) => {
    const rowH = plot.height / rows.length;
    const y = plot.top + index * rowH + rowH * 0.22;
    ctx.fillStyle = "#53647c";
    ctx.font = "850 12px Inter";
    ctx.textAlign = "right";
    ctx.fillText(label, plot.left - 12, y + rowH * 0.28);
    fillRound(ctx, plot.left, y, plot.width, rowH * 0.44, 8, "rgba(226,234,246,0.95)");
    fillRound(ctx, plot.left, y, plot.width * clamp(Number(value || 0), 0, 1), rowH * 0.44, 8, color || "#2d6cff");
    ctx.fillStyle = "#13243d";
    ctx.textAlign = "left";
    ctx.fillText(`${Math.round(clamp(Number(value || 0), 0, 1) * 100)}%`, plot.left + plot.width + 8, y + rowH * 0.28);
  });
}

function drawFeatureTrendCanvas(canvas, report, label = "Feature trends") {
  if (!canvas) return;
  const frames = report?.frames || [];
  if (!frames.length) {
    drawEmptyCanvas(canvas, "no frame data");
    return;
  }
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const pad = { left: 92, right: 54, top: 42, bottom: 34 };
  const plot = {
    left: pad.left,
    right: width - pad.right,
    top: pad.top,
    bottom: height - pad.bottom,
    width: width - pad.left - pad.right,
    height: height - pad.top - pad.bottom
  };
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = "#172238";
  ctx.font = "900 15px Inter";
  ctx.textAlign = "left";
  ctx.fillText(label, plot.left, 23);
  ctx.fillStyle = "#65738a";
  ctx.font = "800 11px Inter";
  ctx.fillText("separate lanes, each signal uses its own scale", plot.left + 210, 23);

  const series = [
    {
      key: "rms",
      label: "RMS",
      color: "#2d6cff",
      values: frames.map((frame) => frame.rms || 0),
      min: 0,
      max: percentile(frames.map((frame) => frame.rms || 0), 98) || 0.08
    },
    {
      key: "pitch",
      label: "Pitch/F0",
      color: "#0a9f72",
      values: frames.map((frame) => Number.isFinite(frame.pitch) ? frame.pitch : 0),
      min: percentile(frames.map((frame) => frame.pitch || 0).filter((v) => v > 0), 5) || 70,
      max: percentile(frames.map((frame) => frame.pitch || 0).filter((v) => v > 0), 95) || 320
    },
    {
      key: "centroid",
      label: "Centroid",
      color: "#f59f00",
      values: frames.map((frame) => frame.centroid || 0),
      min: percentile(frames.map((frame) => frame.centroid || 0), 5) || 300,
      max: percentile(frames.map((frame) => frame.centroid || 0), 95) || 4500
    },
    {
      key: "zcr",
      label: "ZCR",
      color: "#df4f73",
      values: frames.map((frame) => frame.zcr || 0),
      min: 0,
      max: percentile(frames.map((frame) => frame.zcr || 0), 98) || 0.18
    }
  ];

  series.forEach((line, lineIndex) => {
    const values = line.values.map((value) => Number.isFinite(value) ? value : 0);
    const safeMin = Number.isFinite(line.min) ? line.min : percentile(values, 5);
    let safeMax = Number.isFinite(line.max) ? line.max : percentile(values, 95);
    if (!Number.isFinite(safeMax) || safeMax <= safeMin) safeMax = safeMin + 1e-5;
    const laneGap = 12;
    const laneHeight = (plot.height - laneGap * (series.length - 1)) / series.length;
    const laneTop = plot.top + lineIndex * (laneHeight + laneGap);
    const laneBottom = laneTop + laneHeight;
    fillRound(ctx, plot.left, laneTop, plot.width, laneHeight, 10, "rgba(248,251,255,0.96)");
    ctx.strokeStyle = "rgba(99,116,145,0.16)";
    ctx.lineWidth = 1;
    ctx.strokeRect(plot.left, laneTop, plot.width, laneHeight);
    for (let i = 1; i < 4; i++) {
      const y = laneTop + (i / 4) * laneHeight;
      ctx.strokeStyle = "rgba(99,116,145,0.09)";
      ctx.beginPath();
      ctx.moveTo(plot.left, y);
      ctx.lineTo(plot.right, y);
      ctx.stroke();
    }
    for (let i = 1; i <= 6; i++) {
      const x = plot.left + (i / 6) * plot.width;
      ctx.strokeStyle = "rgba(99,116,145,0.07)";
      ctx.beginPath();
      ctx.moveTo(x, laneTop);
      ctx.lineTo(x, laneBottom);
      ctx.stroke();
    }
    ctx.fillStyle = line.color;
    ctx.font = "900 12px Inter";
    ctx.textAlign = "right";
    ctx.fillText(line.label, plot.left - 12, laneTop + laneHeight * 0.58);
    ctx.fillStyle = "#738197";
    ctx.font = "800 9px Inter";
    const minLabel = formatFeatureAxisValue(safeMin, line.key);
    const maxLabel = formatFeatureAxisValue(safeMax, line.key);
    ctx.fillText(maxLabel, plot.left - 12, laneTop + 10);
    ctx.fillText(minLabel, plot.left - 12, laneBottom - 3);

    ctx.strokeStyle = line.color;
    ctx.lineWidth = 2.6;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = plot.left + (index / Math.max(1, values.length - 1)) * plot.width;
      const y = laneBottom - clamp(normalize(value, safeMin, safeMax), 0, 1) * laneHeight;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  });
  ctx.fillStyle = "#53647c";
  ctx.font = "800 11px Inter";
  ctx.textAlign = "center";
  ctx.fillText("Time frames", plot.left + plot.width / 2, height - 10);
  ctx.textAlign = "left";
  ctx.fillText("0", plot.left, height - 10);
  ctx.textAlign = "right";
  ctx.fillText(String(Math.max(0, frames.length - 1)), plot.right, height - 10);
}

function formatFeatureAxisValue(value, key = "") {
  const numeric = Number(value || 0);
  if (key === "pitch" || key === "centroid") return `${Math.round(numeric)}`;
  if (Math.abs(numeric) < 1) return numeric.toFixed(3);
  return numeric.toFixed(1);
}

function resolvedTranscriptSegments(item, report) {
  const fromReport = report?.recognizedTranscript || report?.modelTranscript?.segments || [];
  const fromItem = item?.sttSegments || [];
  return (fromReport.length ? fromReport : fromItem)
    .filter((segment) => segment?.text?.trim())
    .map((segment) => ({
      start: Number.isFinite(Number(segment.start)) ? Number(segment.start) : 0,
      end: Number.isFinite(Number(segment.end)) ? Number(segment.end) : Math.max(0.2, item?.duration || RECORD_SECONDS),
      text: String(segment.text || "").trim()
    }));
}

function deltaMatrix(matrix) {
  if (!matrix?.length) return [];
  return matrix.map((row, index) => {
    const previous = matrix[Math.max(0, index - 1)] || row;
    return row.map((value, col) => Number(value || 0) - Number(previous[col] || 0));
  });
}

function drawEmptyWaveformCanvas(canvas, label = "idle") {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  drawBareWaveGrid(ctx, canvas);
}

function drawBareWaveGrid(ctx, canvas) {
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "rgba(16, 24, 40, 0.075)";
  ctx.lineWidth = 1;
  const verticalCount = 8;
  const horizontalCount = 5;
  for (let i = 0; i <= verticalCount; i++) {
    const x = (width * i) / verticalCount;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let i = 0; i <= horizontalCount; i++) {
    const y = (height * i) / horizontalCount;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawEmptySpectrogramCanvas(canvas, label = "log-Mel preview") {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const bg = ctx.createLinearGradient(0, 0, width, height);
  bg.addColorStop(0, "#07142b");
  bg.addColorStop(0.55, "#101f43");
  bg.addColorStop(1, "#1b1238");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, width, height);
  for (let x = 0; x < width; x += width / 14) {
    ctx.strokeStyle = "rgba(255,255,255,0.06)";
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }
  for (let y = 0; y < height; y += height / 5) {
    ctx.strokeStyle = "rgba(255,255,255,0.07)";
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
  for (let i = 0; i < 36; i++) {
    const x = (i / 36) * width;
    const y = height * (0.18 + 0.64 * Math.abs(Math.sin(i * 0.47)));
    const heat = ctx.createRadialGradient(x, y, 0, x, y, 42);
    heat.addColorStop(0, "rgba(255,213,104,0.28)");
    heat.addColorStop(0.42, "rgba(238,75,98,0.16)");
    heat.addColorStop(1, "rgba(35,195,217,0)");
    ctx.fillStyle = heat;
    ctx.fillRect(x - 48, y - 48, 96, 96);
  }
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.font = "900 22px Inter";
  ctx.textAlign = "left";
  ctx.fillText(label, 22, 34);
  ctx.fillStyle = "rgba(255,255,255,0.58)";
  ctx.font = "800 16px Inter";
  ctx.fillText("time ->", width - 118, height - 18);
  ctx.save();
  ctx.translate(24, height - 20);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Mel frequency", 0, 0);
  ctx.restore();
}

function drawEmptyCanvas(canvas, text, dark = false) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  fillRound(ctx, 0, 0, canvas.width, canvas.height, 10, dark ? "#0b1730" : "rgba(255,255,255,0.62)");
  ctx.fillStyle = dark ? "rgba(255,255,255,0.78)" : "#65738a";
  ctx.font = "900 34px Inter";
  ctx.textAlign = "center";
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
}

function drawOrb(level, freqData) {
  const canvas = $("#orbCanvas");
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const cx = width / 2;
  const cy = height / 2;
  ctx.clearRect(0, 0, width, height);
  const count = 96;
  const base = width * 0.23 + level * 24;
  for (let i = 0; i < count; i++) {
    const idx = Math.floor((i / count) * freqData.length * 0.70);
    const value = (freqData[idx] || 0) / 255;
    const angle = (Math.PI * 2 * i) / count - Math.PI / 2;
    const inner = base;
    const outer = base + 10 + value * 58;
    ctx.strokeStyle = `hsla(${210 - value * 150}, 90%, ${56 + value * 18}%, ${0.20 + value * 0.70})`;
    ctx.lineWidth = 2 + value * 2.6;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(angle) * inner, cy + Math.sin(angle) * inner);
    ctx.lineTo(cx + Math.cos(angle) * outer, cy + Math.sin(angle) * outer);
    ctx.stroke();
  }
}

function initLedMeter() {
  $("#ledMeter").innerHTML = Array.from({ length: 28 }, (_, i) => `<span class="led-bar" data-i="${i}"></span>`).join("");
}

function updateLedMeter(level, freqData) {
  $$(".led-bar").forEach((bar, i, bars) => {
    const idx = Math.floor((i / bars.length) * freqData.length * 0.72);
    const value = Math.max(level * 0.9, (freqData[idx] || 0) / 255);
    bar.style.height = `${12 + value * 88}%`;
    bar.classList.toggle("hot", value > 0.72);
    bar.classList.toggle("live", value > 0.22 && value <= 0.72);
  });
}

function drawAmbient() {
  const canvas = $("#ambientCanvas");
  const ctx = canvas.getContext("2d");
  const particles = Array.from({ length: 46 }, (_, i) => ({
    x: Math.random(), y: Math.random(), r: 80 + Math.random() * 180,
    dx: (Math.random() - 0.5) * 0.00045, dy: (Math.random() - 0.5) * 0.00045,
    hue: [214, 186, 158, 258, 28][i % 5]
  }));
  const resize = () => { canvas.width = window.innerWidth * devicePixelRatio; canvas.height = window.innerHeight * devicePixelRatio; };
  window.addEventListener("resize", resize);
  resize();
  const render = () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach((p) => {
      p.x += p.dx; p.y += p.dy;
      if (p.x < -0.1 || p.x > 1.1) p.dx *= -1;
      if (p.y < -0.1 || p.y > 1.1) p.dy *= -1;
      const x = p.x * canvas.width, y = p.y * canvas.height;
      const gradient = ctx.createRadialGradient(x, y, 0, x, y, p.r * devicePixelRatio);
      gradient.addColorStop(0, `hsla(${p.hue}, 90%, 62%, 0.17)`);
      gradient.addColorStop(1, `hsla(${p.hue}, 90%, 62%, 0)`);
      ctx.fillStyle = gradient;
      ctx.fillRect(x - p.r * devicePixelRatio, y - p.r * devicePixelRatio, p.r * 2 * devicePixelRatio, p.r * 2 * devicePixelRatio);
    });
    requestAnimationFrame(render);
  };
  render();
}

function playTone(type = "click") {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return;
  const ctx = getSoundContext(AudioCtx);
  if (ctx.state === "suspended") ctx.resume();
  const bus = ctx.createGain();
  const compressor = ctx.createDynamicsCompressor();
  bus.gain.setValueAtTime(0.8, ctx.currentTime);
  compressor.threshold.value = -18;
  compressor.knee.value = 18;
  compressor.ratio.value = 3;
  compressor.attack.value = 0.004;
  compressor.release.value = 0.12;
  bus.connect(compressor);
  compressor.connect(ctx.destination);

  const now = ctx.currentTime + 0.006;
  const score = SOUND_SCORES[type] || SOUND_SCORES.click;
  let endAt = now;
  (score.tones || []).forEach((tone) => {
    endAt = Math.max(endAt, scheduleTone(ctx, bus, now + (tone.at || 0), tone));
  });
  (score.noise || []).forEach((noise) => {
    endAt = Math.max(endAt, scheduleNoise(ctx, bus, now + (noise.at || 0), noise));
  });
  setTimeout(() => {
    try { bus.disconnect(); compressor.disconnect(); } catch {}
  }, Math.ceil((endAt - ctx.currentTime + 0.12) * 1000));
}

function unlockSoundOnce() {
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) return;
  const ctx = getSoundContext(AudioCtx);
  if (ctx.state === "suspended") ctx.resume();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const now = ctx.currentTime;
  osc.type = "sine";
  osc.frequency.setValueAtTime(880, now);
  gain.gain.setValueAtTime(0.0001, now);
  gain.gain.exponentialRampToValueAtTime(0.055, now + 0.012);
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.055);
  osc.connect(gain);
  gain.connect(ctx.destination);
  osc.start(now);
  osc.stop(now + 0.065);
}

const SOUND_SCORES = {
  click: {
    tones: [
      { freq: 760, end: 1040, dur: 0.070, type: "sine", gain: 0.58 },
      { at: 0.024, freq: 1340, dur: 0.055, type: "triangle", gain: 0.34 }
    ]
  },
  tab: {
    tones: [
      { freq: 420, end: 680, dur: 0.090, type: "sine", gain: 0.42 },
      { at: 0.050, freq: 720, end: 980, dur: 0.096, type: "triangle", gain: 0.36 }
    ],
    noise: [{ at: 0.006, dur: 0.085, gain: 0.055, filter: 1800 }]
  },
  tick: {
    tones: [{ freq: 980, dur: 0.050, type: "sine", gain: 0.42 }],
    noise: [{ dur: 0.040, gain: 0.040, filter: 2400 }]
  },
  start: {
    tones: [
      { freq: 180, end: 260, dur: 0.055, type: "sine", gain: 0.30 },
      { at: 0.045, freq: 360, end: 620, dur: 0.085, type: "triangle", gain: 0.24 },
      { at: 0.108, freq: 720, end: 980, dur: 0.120, type: "sine", gain: 0.16 }
    ],
    noise: [{ at: 0.018, dur: 0.12, gain: 0.032, filter: 900 }]
  },
  stop: {
    tones: [
      { freq: 640, end: 360, dur: 0.075, type: "triangle", gain: 0.25 },
      { at: 0.055, freq: 240, dur: 0.085, type: "sine", gain: 0.18 }
    ]
  },
  warning: {
    tones: [
      { freq: 820, dur: 0.070, type: "square", gain: 0.16 },
      { at: 0.080, freq: 420, dur: 0.100, type: "sawtooth", gain: 0.13 }
    ],
    noise: [{ at: 0.010, dur: 0.16, gain: 0.030, filter: 1200 }]
  },
  play: {
    tones: [
      { freq: 540, end: 820, dur: 0.095, type: "sine", gain: 0.45 },
      { at: 0.055, freq: 1080, dur: 0.082, type: "triangle", gain: 0.30 }
    ]
  },
  queueReady: {
    tones: [
      { freq: 420, dur: 0.044, type: "sine", gain: 0.20 },
      { at: 0.040, freq: 640, dur: 0.052, type: "sine", gain: 0.20 },
      { at: 0.088, freq: 880, dur: 0.100, type: "triangle", gain: 0.18 }
    ],
    noise: [{ at: 0.018, dur: 0.105, gain: 0.018, filter: 2600 }]
  },
  queue: {
    tones: [
      { freq: 520, end: 760, dur: 0.060, type: "triangle", gain: 0.22 },
      { at: 0.052, freq: 980, dur: 0.080, type: "sine", gain: 0.16 }
    ],
    noise: [{ at: 0.014, dur: 0.090, gain: 0.022, filter: 2100 }]
  },
  analyze: {
    tones: [
      { freq: 220, end: 360, dur: 0.055, type: "sawtooth", gain: 0.14 },
      { at: 0.050, freq: 360, end: 520, dur: 0.065, type: "triangle", gain: 0.16 },
      { at: 0.108, freq: 520, end: 740, dur: 0.095, type: "sine", gain: 0.14 }
    ],
    noise: [{ at: 0.020, dur: 0.18, gain: 0.028, filter: 1400 }]
  },
  success: {
    tones: [
      { freq: 523.25, dur: 0.090, type: "sine", gain: 0.42 },
      { at: 0.070, freq: 659.25, dur: 0.100, type: "sine", gain: 0.42 },
      { at: 0.145, freq: 783.99, dur: 0.150, type: "triangle", gain: 0.36 },
      { at: 0.220, freq: 1046.50, dur: 0.170, type: "sine", gain: 0.24 }
    ],
    noise: [{ at: 0.090, dur: 0.16, gain: 0.034, filter: 3200 }]
  },
  "emotion-neutral": {
    tones: [
      { freq: 392, end: 440, dur: 0.105, type: "sine", gain: 0.15 },
      { at: 0.072, freq: 523.25, dur: 0.110, type: "triangle", gain: 0.12 }
    ]
  },
  "emotion-happy": {
    tones: [
      { freq: 523.25, dur: 0.052, type: "sine", gain: 0.16 },
      { at: 0.046, freq: 659.25, dur: 0.052, type: "sine", gain: 0.16 },
      { at: 0.092, freq: 880, dur: 0.120, type: "triangle", gain: 0.13 }
    ],
    noise: [{ at: 0.060, dur: 0.10, gain: 0.012, filter: 3600 }]
  },
  "emotion-sad": {
    tones: [
      { freq: 330, end: 247, dur: 0.145, type: "sine", gain: 0.14 },
      { at: 0.105, freq: 220, dur: 0.140, type: "triangle", gain: 0.10 }
    ]
  },
  "emotion-angry": {
    tones: [
      { freq: 196, end: 294, dur: 0.070, type: "sawtooth", gain: 0.12 },
      { at: 0.060, freq: 392, end: 330, dur: 0.085, type: "square", gain: 0.10 }
    ],
    noise: [{ at: 0.010, dur: 0.125, gain: 0.024, filter: 900 }]
  },
  reset: {
    tones: [{ freq: 320, end: 180, dur: 0.095, type: "sine", gain: 0.19 }],
    noise: [{ dur: 0.080, gain: 0.020, filter: 700 }]
  }
};

function getSoundContext(AudioCtx) {
  if (!playTone.ctx || playTone.ctx.state === "closed") playTone.ctx = new AudioCtx();
  return playTone.ctx;
}

function scheduleTone(ctx, destination, start, tone) {
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  const filter = ctx.createBiquadFilter();
  const dur = tone.dur || 0.05;
  const attack = Math.min(0.012, dur * 0.24);
  osc.type = tone.type || "sine";
  osc.frequency.setValueAtTime(tone.freq, start);
  if (tone.end) osc.frequency.exponentialRampToValueAtTime(Math.max(1, tone.end), start + dur);
  filter.type = "lowpass";
  filter.frequency.setValueAtTime(tone.filter || 4200, start);
  filter.Q.value = 0.6;
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(tone.gain || 0.18, start + attack);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  osc.connect(filter);
  filter.connect(gain);
  gain.connect(destination);
  osc.start(start);
  osc.stop(start + dur + 0.018);
  return start + dur;
}

function scheduleNoise(ctx, destination, start, noise) {
  const dur = noise.dur || 0.08;
  const bufferSize = Math.max(1, Math.floor(ctx.sampleRate * dur));
  const buffer = ctx.createBuffer(1, bufferSize, ctx.sampleRate);
  const data = buffer.getChannelData(0);
  for (let i = 0; i < bufferSize; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / bufferSize);
  const source = ctx.createBufferSource();
  const filter = ctx.createBiquadFilter();
  const gain = ctx.createGain();
  source.buffer = buffer;
  filter.type = "bandpass";
  filter.frequency.setValueAtTime(noise.filter || 1800, start);
  filter.Q.value = 1.2;
  gain.gain.setValueAtTime(0.0001, start);
  gain.gain.exponentialRampToValueAtTime(noise.gain || 0.02, start + 0.010);
  gain.gain.exponentialRampToValueAtTime(0.0001, start + dur);
  source.connect(filter);
  filter.connect(gain);
  gain.connect(destination);
  source.start(start);
  source.stop(start + dur + 0.012);
  return start + dur;
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timeout);
  showToast.timeout = setTimeout(() => toast.classList.remove("show"), 3300);
}

function magnitudeSpectrum(frame) {
  const n = frame.length;
  const real = new Float32Array(n);
  const imag = new Float32Array(n);
  real.set(frame);
  fft(real, imag);
  const mag = new Float32Array(n / 2);
  for (let i = 0; i < mag.length; i++) mag[i] = Math.hypot(real[i], imag[i]);
  return mag;
}

function fft(real, imag) {
  const n = real.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    while (j & bit) { j ^= bit; bit >>= 1; }
    j ^= bit;
    if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = -2 * Math.PI / len;
    const wLenR = Math.cos(angle);
    const wLenI = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wR = 1, wI = 0;
      for (let k = 0; k < len / 2; k++) {
        const uR = real[i + k], uI = imag[i + k];
        const vR = real[i + k + len / 2] * wR - imag[i + k + len / 2] * wI;
        const vI = real[i + k + len / 2] * wI + imag[i + k + len / 2] * wR;
        real[i + k] = uR + vR; imag[i + k] = uI + vI;
        real[i + k + len / 2] = uR - vR; imag[i + k + len / 2] = uI - vI;
        const nextR = wR * wLenR - wI * wLenI;
        wI = wR * wLenI + wI * wLenR; wR = nextR;
      }
    }
  }
}

function spectralFeatures(spectrum, sampleRate, prevSpectrum) {
  const nyquist = sampleRate / 2;
  const binHz = nyquist / spectrum.length;
  let sumMag = 0, weighted = 0, weighted2 = 0, cumulative = 0, rolloff = 0, flux = 0;
  for (let i = 0; i < spectrum.length; i++) { const mag = spectrum[i] + 1e-10; sumMag += mag; weighted += i * binHz * mag; }
  const centroid = weighted / Math.max(sumMag, 1e-10);
  for (let i = 0; i < spectrum.length; i++) weighted2 += (((i * binHz) - centroid) ** 2) * (spectrum[i] + 1e-10);
  const bandwidth = Math.sqrt(weighted2 / Math.max(sumMag, 1e-10));
  const target = sumMag * 0.85;
  for (let i = 0; i < spectrum.length; i++) { cumulative += spectrum[i]; if (cumulative >= target) { rolloff = i * binHz; break; } }
  if (prevSpectrum) {
    for (let i = 0; i < spectrum.length; i++) { const diff = spectrum[i] - prevSpectrum[i]; if (diff > 0) flux += diff * diff; }
    flux = Math.sqrt(flux / spectrum.length);
  }
  return { centroid, bandwidth, rolloff, flux, contrast: spectralContrast(spectrum) };
}

function spectralContrast(spectrum) {
  const bands = 6, values = [];
  for (let b = 0; b < bands; b++) {
    const start = Math.floor((b / bands) * spectrum.length);
    const end = Math.max(start + 2, Math.floor(((b + 1) / bands) * spectrum.length));
    const slice = Array.from(spectrum.slice(start, end)).sort((a, bVal) => a - bVal);
    const low = mean(slice.slice(0, Math.max(1, Math.floor(slice.length * 0.18)))) + 1e-8;
    const high = mean(slice.slice(Math.max(0, Math.floor(slice.length * 0.82)))) + 1e-8;
    values.push(20 * Math.log10(high / low));
  }
  return mean(values);
}

function buildMelFilterBank(sampleRate, fftSize, nMels) {
  const minMel = hzToMel(40), maxMel = hzToMel(sampleRate / 2);
  const points = Array.from({ length: nMels + 2 }, (_, i) => melToHz(minMel + (i / (nMels + 1)) * (maxMel - minMel)));
  const bins = points.map((hz) => Math.floor((fftSize + 1) * hz / sampleRate));
  return Array.from({ length: nMels }, (_, m0) => {
    const m = m0 + 1;
    const filter = new Float32Array(fftSize / 2);
    const left = bins[m - 1], center = bins[m], right = bins[m + 1];
    for (let k = left; k < center; k++) filter[k] = (k - left) / Math.max(1, center - left);
    for (let k = center; k < right && k < filter.length; k++) filter[k] = (right - k) / Math.max(1, right - center);
    return filter;
  });
}

function applyMelBank(spectrum, bank) {
  return bank.map((filter) => {
    let sum = 0;
    for (let i = 0; i < spectrum.length; i++) sum += spectrum[i] * filter[i];
    return sum;
  });
}

function dct(values, count) {
  const n = values.length;
  return Array.from({ length: count }, (_, k) => {
    let sum = 0;
    for (let i = 0; i < n; i++) sum += values[i] * Math.cos((Math.PI / n) * (i + 0.5) * k);
    return sum;
  });
}

function estimatePitch(samples, start, frameSize, sampleRate) {
  const minLag = Math.floor(sampleRate / 420);
  const maxLag = Math.min(Math.floor(sampleRate / 70), frameSize - 2);
  let bestLag = 0, bestCorr = 0, energy = 0;
  for (let i = 0; i < frameSize; i++) energy += (samples[start + i] || 0) ** 2;
  if (energy < 1e-4) return NaN;
  for (let lag = minLag; lag <= maxLag; lag += 2) {
    let corr = 0;
    for (let i = 0; i < frameSize - lag; i += 2) corr += (samples[start + i] || 0) * (samples[start + i + lag] || 0);
    corr /= Math.max(1, frameSize - lag);
    if (corr > bestCorr) { bestCorr = corr; bestLag = lag; }
  }
  return bestLag ? sampleRate / bestLag : NaN;
}

function nearestPitch(frames, index) {
  for (let radius = 1; radius < 6; radius++) {
    const left = frames[index - radius]?.pitch, right = frames[index + radius]?.pitch;
    if (Number.isFinite(left)) return left;
    if (Number.isFinite(right)) return right;
  }
  return 0;
}

function structuredCloneTask(task) {
  return JSON.parse(JSON.stringify(task));
}

function buildTask(emotion, promptIndex = 0) {
  if (emotion === "none") return { ...FREE_TASK, promptIndex: 0 };
  const group = TASK_GROUPS.find((item) => item.emotion === emotion) || TASK_GROUPS[0];
  const safeIndex = clamp(Math.floor(promptIndex || 0), 0, group.prompts.length - 1);
  const prompt = group.prompts[safeIndex];
  return {
    emotion: group.emotion,
    title: prompt.title,
    icon: group.icon,
    script: prompt.script,
    hints: prompt.hints,
    promptIndex: safeIndex
  };
}

function hann(n) { return Float32Array.from({ length: n }, (_, i) => 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)))); }
function hzToMel(hz) { return 2595 * Math.log10(1 + hz / 700); }
function melToHz(mel) { return 700 * (10 ** (mel / 2595) - 1); }
function mean(values) { return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0; }
function std(values) { const m = mean(values); return values.length ? Math.sqrt(mean(values.map((value) => (value - m) ** 2))) : 0; }
function percentile(values, p) { const sorted = [...values].filter(Number.isFinite).sort((a, b) => a - b); if (!sorted.length) return 0; const index = (p / 100) * (sorted.length - 1); const low = Math.floor(index), high = Math.ceil(index), weight = index - low; return sorted[low] * (1 - weight) + sorted[high] * weight; }
function flatten(matrix) { return matrix.reduce((out, row) => out.concat(row), []); }
function downsample(samples, targetLength) { if (samples.length <= targetLength) return Array.from(samples); const block = Math.floor(samples.length / targetLength); return Array.from({ length: targetLength }, (_, i) => { let sum = 0; for (let j = 0; j < block; j++) sum += samples[i * block + j] || 0; return sum / block; }); }
function softmax(values) { const max = Math.max(...values); const exp = values.map((value) => Math.exp(value - max)); const sum = exp.reduce((a, b) => a + b, 0); return exp.map((value) => value / sum); }
function normalize(value, min, max) { return clamp((value - min) / Math.max(1e-8, max - min), 0, 1); }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function computeByteRms(timeData) { let sum = 0; for (let i = 0; i < timeData.length; i++) { const value = (timeData[i] - 128) / 128; sum += value * value; } return Math.sqrt(sum / timeData.length); }
function fillRound(ctx, x, y, width, height, radius, fillStyle) { ctx.beginPath(); ctx.moveTo(x + radius, y); ctx.arcTo(x + width, y, x + width, y + height, radius); ctx.arcTo(x + width, y + height, x, y + height, radius); ctx.arcTo(x, y + height, x, y, radius); ctx.arcTo(x, y, x + width, y, radius); ctx.closePath(); ctx.fillStyle = fillStyle; ctx.fill(); }
function heatColor(value) { const v = clamp(value, 0, 1); return `hsl(${235 - v * 205}, 92%, ${15 + v * 62}%)`; }
function pct(value) { return `${Math.round(value * 100)}%`; }
function formatVadScore(value) {
  const numeric = clamp(Number(value || 0), 0, 1);
  return `${numeric.toFixed(2)} / 1.00`;
}
function formatVadIemocap(value) {
  const numeric = clamp(Number(value || 0), 0, 1);
  return `${(1 + numeric * 4).toFixed(2)} / 5`;
}
function capitalize(value) { return value.charAt(0).toUpperCase() + value.slice(1); }
function formatTaskLabel(task) { return task?.emotion === "none" ? `Free - ${task.title}` : `${capitalize(task.emotion)} - ${task.title}`; }
function splitSentences(text) {
  const parts = String(text || "").match(/[^.!?]+[.!?]+|[^.!?]+$/g) || [];
  return parts.map((part) => part.trim()).filter(Boolean);
}
function formatTimecode(seconds) { const safe = Math.max(0, seconds || 0); const mins = Math.floor(safe / 60); const secs = Math.floor(safe % 60); const tenths = Math.round((safe - Math.floor(safe)) * 10); return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${tenths}`; }
function formatDuration(seconds) { return `${Math.max(0, seconds || 0).toFixed(1)}s`; }
function formatRecordClock(seconds) {
  const safe = Math.max(0, Number(seconds || 0));
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const hundredths = Math.floor((safe - Math.floor(safe)) * 100);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(hundredths).padStart(2, "0")}`;
}
function formatClock(date) { return new Intl.DateTimeFormat("vi-VN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(date); }
function delay(ms) { return new Promise((resolve) => setTimeout(resolve, ms)); }
function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setText(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

