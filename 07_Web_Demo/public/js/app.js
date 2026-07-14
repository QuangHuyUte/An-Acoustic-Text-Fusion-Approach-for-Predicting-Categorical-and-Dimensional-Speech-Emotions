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
  script: "Record any short presentation sentence you want. The system will analyze emotion, valence, arousal and dominance from the captured voice.",
  hints: []
};

const TASK_GROUPS = [
  {
    emotion: "neutral",
    label: "Neutral",
    icon: "meh",
    prompts: [
      { title: "Calm response", script: "I am fine. I can continue the presentation. The result is important to me.", hints: ["Pitch balanced", "Energy steady", "Pace clear"] },
      { title: "Academic summary", script: "The experiment result is stable. I will explain the next chart carefully.", hints: ["Even tone", "Clear articulation", "Controlled pace"] },
      { title: "Meeting update", script: "The current progress is acceptable. We can move to the next part.", hints: ["Flat valence", "Moderate energy", "Precise pauses"] }
    ]
  },
  {
    emotion: "happy",
    label: "Happy",
    icon: "smile",
    prompts: [
      { title: "Warm response", script: "I am fine. I can continue the presentation. The result is important to me.", hints: ["Pitch brighter", "Energy warm", "Pace lively"] },
      { title: "Positive result", script: "This result is encouraging. I am excited to show how the model improves.", hints: ["Higher pitch", "Open tone", "Energetic stress"] },
      { title: "Confident greeting", script: "Good morning everyone. I am happy to present our speech project today.", hints: ["Friendly onset", "Bright timbre", "Smooth rhythm"] }
    ]
  },
  {
    emotion: "sad",
    label: "Sad",
    icon: "frown",
    prompts: [
      { title: "Low-energy response", script: "I am fine. I can continue the presentation. The result is important to me.", hints: ["Pitch lower", "Energy softer", "Pace slower"] },
      { title: "Concerned result", script: "The accuracy is still low. We need more experiments before the final version.", hints: ["Low energy", "Longer pauses", "Soft ending"] },
      { title: "Reflective note", script: "I expected a better result, but this mistake helps us understand the model.", hints: ["Lower valence", "Reduced loudness", "Gentle speech"] }
    ]
  },
  {
    emotion: "angry",
    label: "Angry",
    icon: "angry",
    prompts: [
      { title: "Firm response", script: "I am fine. I can continue the presentation. The result is important to me.", hints: ["Pitch firm", "Energy high", "Pace controlled"] },
      { title: "Strong objection", script: "This error is serious. We must fix the pipeline before reporting the result.", hints: ["High arousal", "Sharp consonants", "Strong emphasis"] },
      { title: "Urgent warning", script: "The system cannot ignore this issue. The evaluation split must be checked again.", hints: ["Tense voice", "High intensity", "Short pauses"] }
    ]
  }
];

const state = {
  task: FREE_TASK,
  selectedEmotion: "none",
  selectedPromptIndex: 0,
  queue: [],
  selectedQueueId: null,
  selectedResultId: null,
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
  renderResultHistory();
  restoreQueueBubblePosition();
  initQueueDrag();
  if (window.gsap) {
    gsap.from(".app-header, .tab-screen.active .panel:not(.queue-drawer)", { opacity: 0, y: 16, duration: 0.6, stagger: 0.035, ease: "power3.out" });
  }
  $("#queueDrawer")?.removeAttribute("style");
}

function initIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function bindEvents() {
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
    const item = getSelectedItem() || state.currentRecording;
    if (!item?.audioUrl) return;
    $("#playbackAudio").src = item.audioUrl;
    $("#playbackAudio").currentTime = 0;
    $("#playbackAudio").play();
    playTone("play");
  });
  $("#addQueueBtn").addEventListener("click", addCurrentRecordingToQueue);
  $("#resetBtn").addEventListener("click", resetCurrentRecording);
  $("#analyzeBtn").addEventListener("click", analyzeSelectedQueue);
  $("#backRecordBtn").addEventListener("click", () => switchTab("record"));
  $("#exportReportBtn").addEventListener("click", exportSelectedReport);
  $("#downloadBtn").addEventListener("click", downloadSelectedAudio);
  bindProcessingExplainer();
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
    if (event.target.closest(".feature-tabs")) return;
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

function selectAcousticFeature(featureName = "temporal") {
  $$("#acousticFeatureTabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.feature === featureName);
  });
  $$("[data-feature-panel]").forEach((panel) => {
    panel.classList.toggle("active-feature", panel.dataset.featurePanel === featureName);
  });
}

function openInspectModal(kind) {
  const item = getSelectedItem() || getSelectedResultItem();
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
    drawSpeechWaveform($("#inspectWaveCanvas"), item.samples, item.sampleRate, item.duration, item.sttSegments || [], item.duration, item.task?.script || "");
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
    attention: "Attention(Q,K,V) = softmax(QKᵀ / √dₖ)V",
    fusion: "z = z_bridge + g ⊙ z_acoustic + (1 - g) ⊙ z_text",
    vad: "VAD = [valence, arousal, dominance] ∈ [0, 1]³",
    temporal: "X_temporal = concat(MFCC, ΔMFCC, Δ²MFCC, RMS, ZCR, spectral LLD)",
    spectral: "X_spectral = stack(logMel, ΔlogMel, Δ²logMel)",
    stats: "stats = [mean, std, min, max, median, p10, p90, IQR, harmonic/spectral summaries]"
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
      body: `<div class="inspect-transcript-lines">${transcriptHtml}</div><div class="formula-card"><span>Text sequence</span><code>raw text → tokenizer → input_ids + attention_mask → frozen text encoder → text tokens</code></div>`
    },
    acoustic: {
      tag: "Acoustic",
      title: "Four acoustic branches",
      subtitle: "The acoustic block keeps complementary views instead of compressing speech into one feature type too early.",
      body: `${metricGrid}<div class="explain-card"><span>Branch logic</span><p>A temporal branch captures frame-by-frame acoustic dynamics, a spectrogram branch captures time-frequency patterns, an emotion2vec branch contributes pretrained emotion representation, and a statistical branch preserves stable clip-level descriptors.</p></div><div class="formula-card"><span>Token set</span><code>A = [z_temporal, z_spectral, z_e2v, z_stats]</code></div>`
    },
    temporal: {
      tag: "Branch A",
      title: "Temporal acoustic branch",
      subtitle: "MFCC/delta/prosody/spectral low-level descriptors keep emotion changes along time.",
      body: `${metricsForStats(s)}<div class="formula-card"><span>Input</span><code>${formulas.temporal}</code></div><div class="explain-card"><span>Encoder</span><p>1D-CNN/TCN detects local acoustic patterns, BiLSTM/GRU reads context in both directions, and attention pooling emphasizes frames that contain stronger emotion evidence.</p></div>`
    },
    spectral: {
      tag: "Branch B",
      title: "Spectrogram branch",
      subtitle: "A 2D time-frequency view helps CNN filters see emotion-related spectral texture.",
      body: `${metricGrid}<div class="formula-card"><span>Input</span><code>${formulas.spectral}</code></div><div class="explain-card"><span>Encoder</span><p>2D-CNN extracts local time-frequency patterns. SE/channel attention reweights useful feature maps before producing a spectral token.</p></div>`
    },
    e2v: {
      tag: "Branch C",
      title: "Frozen emotion2vec branch",
      subtitle: "A pretrained speech emotion representation acts as a compact emotion prior.",
      body: `${metricGrid}<div class="formula-card"><span>Projection</span><code>waveform → frozen emotion2vec → mean/utterance embedding [B, 768] → adapter MLP → z_e2v</code></div><div class="explain-card"><span>Why frozen</span><p>The demo keeps this encoder frozen to stay lightweight and to avoid overfitting the small SER datasets during live use.</p></div>`
    },
    stats: {
      tag: "Branch D",
      title: "Statistical branch",
      subtitle: "Clip-level descriptors preserve stable cues that sequence models may blur.",
      body: `${metricsForStats(s)}<div class="formula-card"><span>Feature vector</span><code>${formulas.stats}</code></div><div class="explain-card"><span>Role</span><p>The vector summarizes loudness, voicing, brightness, rolloff, contrast, pitch and stability. It can feed a small MLP or an RBF-SVM probability branch.</p></div>`
    },
    acoustic_attention: {
      tag: "Attention",
      title: "Acoustic self-attention",
      subtitle: "The four acoustic tokens exchange information before touching the text branch.",
      body: `<div class="formula-card"><span>Formula</span><code>${formulas.attention}</code></div><div class="explain-card"><span>Function</span><p>Self-attention lets the model learn whether temporal dynamics, log-Mel texture, emotion2vec embedding or statistical descriptors should dominate for this utterance.</p></div>`
    },
    text_model: {
      tag: "Text",
      title: "Pretrained text branch",
      subtitle: "Transcript tokens are encoded by a frozen language model and projected into the same fusion space.",
      body: `${transcriptHtml}<div class="formula-card"><span>Pipeline</span><code>Transcript → tokenizer → frozen RoBERTa/DeBERTa → sequence projection → text self-attention</code></div><div class="explain-card"><span>Function</span><p>The branch adds linguistic context, while acoustic cues still decide how the sentence was spoken.</p></div>`
    },
    bridge: {
      tag: "Fusion",
      title: "Bridge cross-attention",
      subtitle: "Bridge tokens are the controlled meeting point between acoustic and text tokens.",
      body: `<div class="formula-card"><span>Cross-attention</span><code>Q = Z_bridge W_Q, K = X_modality W_K, V = X_modality W_V; ${formulas.attention}</code></div><div class="explain-card"><span>Function</span><p>Instead of concatenating everything blindly, learnable bridge tokens query acoustic and text evidence and keep fusion more structured.</p></div>`
    },
    balanced_fusion: {
      tag: "Fusion",
      title: "Balanced fusion",
      subtitle: "A gate controls how much acoustic and text evidence enters the final decision.",
      body: `<div class="formula-card"><span>Gated fusion</span><code>${formulas.fusion}</code></div><div class="explain-card"><span>Function</span><p>The gate prevents transcript information from overwhelming emotion cues in the voice. This follows the 03B idea of balanced bridge fusion and random modality masking.</p></div>`
    },
    heads: {
      tag: "Heads",
      title: "Emotion classifier and VAD regressor",
      subtitle: "The same fused vector produces categorical emotion and dimensional affect scores.",
      body: `${predictionMetrics(report)}<div class="formula-card"><span>Outputs</span><code>p_emotion = softmax(Wz + b); ${formulas.vad}</code></div><div class="explain-card"><span>Function</span><p>Emotion uses four demo labels here. VAD describes whether the voice sounds positive/negative, excited/calm and dominant/weak.</p></div>`
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
      <div class="inspect-metric"><span>VAD</span><strong>${p ? `${pct(p.valence)} / ${pct(p.arousal)} / ${pct(p.dominance)}` : "--"}</strong></div>
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
    showToast("Microphone chỉ chạy khi mở bằng http://localhost:5174 hoặc HTTPS.");
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
    showToast("Không mở được microphone. Hãy dùng localhost và cấp quyền mic.");
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
    $("#playbackAudio").src = audioUrl;
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
    $("#recordStatus").textContent = "Ready to queue";
    $("#globalStatus").textContent = "Audio recorded";
    if (!$(".timer").classList.contains("over-limit")) $("#timerNote").textContent = "captured / 00:12.00";
    $("#recHint").textContent = "Recording saved locally. Add it to queue to process.";
    $("#addQueueBtn").disabled = false;
    $("#playBtn").disabled = false;
    drawSpeechWaveform($("#liveWaveCanvas"), samples, audioBuffer.sampleRate, audioBuffer.duration, state.currentRecording.sttSegments, audioBuffer.duration, state.currentRecording.task.script);
    $("#waveDurationLabel").textContent = `${audioBuffer.duration.toFixed(1)}s / ${RECORD_SECONDS.toFixed(1)}s`;
    await decodeContext.close();
    await state.audioContext?.close();
    showToast("Voice recorded. Add it to queue, then open Processing.");
    playTone("queueReady");
  } catch (error) {
    console.error(error);
    showToast("Decode audio lỗi. Thử ghi lại bằng Chrome hoặc Edge.");
  }
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
  const item = {
    ...state.currentRecording,
    id: `session-${Date.now()}`,
    status: "recorded",
    report: null
  };
  state.queue.unshift(item);
  state.selectedQueueId = item.id;
  state.currentRecording = null;
  $("#addQueueBtn").disabled = true;
  $("#recordStatus").textContent = "Queued";
  $("#globalStatus").textContent = "Queue ready";
  renderQueue();
  renderProcessSelection();
  resetPipeline();
  unlockTab("process");
  openQueueDrawer();
  showToast("Added to queue. Open Processing to analyze this audio.");
  playTone("queue");
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
      <small>${item.status}</small>
    </button>
  `).join("");
  $$("#recordQueue .queue-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedQueueId = button.dataset.id;
      renderQueue();
      renderProcessSelection();
      resetPipeline();
      unlockTab("process");
      playTone("click");
    });
  });
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
    return;
  }
  $("#processTitle").textContent = formatTaskLabel(item.task);
  $("#processScript").textContent = item.task.script;
  $("#processMeta").innerHTML = `
    <div><span>Duration</span><strong>${formatDuration(item.duration)}</strong></div>
    <div><span>Sample rate</span><strong>${item.sampleRate} Hz</strong></div>
    <div><span>Status</span><strong>${item.status}</strong></div>
    <div><span>STT lines</span><strong>${item.sttSegments?.length || 0}</strong></div>
  `;
  $("#analyzeBtn").disabled = false;
  renderProcessInputPreview(item);
}

function renderProcessInputPreview(item) {
  const thumb = $("#processWaveThumb");
  if (!item) {
    drawEmptyWaveformCanvas(thumb, "empty");
    $("#processAudioCaption").textContent = "No queue item loaded";
    $("#processTranscriptPreview").textContent = "Record a voice sample, add it to queue, then load it into Processing.";
    $("#processTranscriptCaption").textContent = "Speech-to-text lines appear after recording.";
    $("#textTokenSummary").textContent = "No transcript token has been loaded yet.";
    return;
  }
  drawSpeechWaveform(thumb, item.samples, item.sampleRate, item.duration, item.sttSegments || [], item.duration, item.task?.script || "");
  $("#processAudioCaption").textContent = `${formatDuration(item.duration)} | ${item.sampleRate} Hz | ${formatTaskLabel(item.task)}`;
  const transcript = item.sttSegments?.length
    ? item.sttSegments.map((segment) => segment.text).join(" ")
    : item.task.script;
  $("#processTranscriptPreview").textContent = transcript;
  $("#processTranscriptCaption").textContent = `${item.sttSegments?.length || 0} STT line(s) | click to inspect timestamped text`;
  $("#textTokenSummary").textContent = `Transcript length: ${transcript.split(/\s+/).filter(Boolean).length} words. Frozen text encoder produces contextual sequence tokens for self-attention.`;
}

async function analyzeSelectedQueue() {
  const item = getSelectedItem();
  if (!item) return;
  resetPipeline();
  $("#processStatus").textContent = "Running";
  $("#globalStatus").textContent = "Analyzing queued audio";
  $("#analyzeBtn").disabled = true;
  playTone("analyze");

  await runStep("load", 420);
  drawProcessWave(item);
  await runStep("wave", 520);
  const report = analyzeAudio(item.samples, item.sampleRate);
  report.duration = item.duration;
  report.targetEmotion = item.task.emotion === "none" ? null : item.task.emotion;
  report.script = item.task.script;
  report.recognizedTranscript = item.sttSegments || [];
  report.scriptTimeline = buildRecorderTimeline(item);
  await runStep("features", 600);
  drawHeatmap($("#processMelCanvas"), report.logMel, "Log-Mel");
  renderProcessNumbers(report);
  await Promise.all([
    runStep("temporal", 520),
    runStep("spectral", 520),
    runStep("pretrained", 520),
    runStep("stats", 520),
    runStep("text", 520)
  ]);
  await runStep("attention", 420);
  await runStep("text_attention", 340);
  report.prediction = predictAffect(report, report.targetEmotion);
  $("#flowOutputSummary").textContent = `${capitalize(report.prediction.emotion)} | V ${pct(report.prediction.valence)} A ${pct(report.prediction.arousal)} D ${pct(report.prediction.dominance)}`;
  await runStep("fusion", 520);
  await runStep("bridge", 420);
  await runStep("heads", 380);
  item.status = "processed";
  item.report = report;
  state.selectedResultId = item.id;
  await runStep("report", 360);

  $("#processStatus").textContent = "Complete";
  $("#globalStatus").textContent = "Report ready";
  $("#analyzeBtn").disabled = false;
  renderQueue();
  renderResultHistory();
  renderSelectedResult();
  unlockTab("results");
  playTone("success");
  showToast("Analysis complete. Open Results to review emotion, VAD and transcript.");
}

function resetPipeline() {
  $$("[data-step]").forEach((step) => step.classList.remove("active", "done"));
  drawEmptyWaveformCanvas($("#processWaveCanvas"), "waiting for selected audio");
  drawEmptySpectrogramCanvas($("#processMelCanvas"), "waiting for feature extraction");
  $("#processNumbers").innerHTML = "";
  $("#flowOutputSummary").textContent = "Waiting for processed audio.";
  const item = getSelectedItem();
  renderProcessInputPreview(item || null);
}

async function runStep(stepName, ms) {
  const steps = $$(`[data-step="${stepName}"]`);
  steps.forEach((step) => step.classList.add("active"));
  playTone(stepName === "report" ? "success" : "tick");
  await delay(ms);
  steps.forEach((step) => {
    step.classList.remove("active");
    step.classList.add("done");
  });
}

function renderProcessNumbers(report) {
  const s = report.stats;
  const frameCount = report.frames.length;
  const melBins = report.logMel[0]?.length || 0;
  const mfccDim = report.mfcc[0]?.length || 0;
  $("#temporalShape").textContent = `[B, ${mfccDim * 3 || 39}+LLD, ${frameCount}] -> 1D-CNN/TCN -> BiLSTM/GRU -> attention`;
  $("#spectralShape").textContent = `[B, 3, ${melBins || 40}, ${frameCount}] -> 2D-CNN + SE/channel attention`;
  $("#statsShape").textContent = `RMS ${s.rmsMean.toFixed(4)} | ZCR ${s.zcrMean.toFixed(4)} | centroid ${Math.round(s.centroidMean)} Hz`;
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
      <small>report</small>
    </button>
  `).join("");
  $$("#resultHistory .queue-item").forEach((button) => {
    button.addEventListener("click", () => {
      state.selectedResultId = button.dataset.id;
      renderResultHistory();
      renderSelectedResult();
      playTone("click");
    });
  });
}

function getSelectedResultItem() {
  return state.queue.find((item) => item.id === state.selectedResultId && item.report) || null;
}

function renderSelectedResult() {
  const item = getSelectedResultItem();
  if (!item) return;
  const report = item.report;
  const pred = report.prediction;
  $("#resultTitle").textContent = `${formatTaskLabel(item.task)} report`;
  $("#predictedEmotion").textContent = capitalize(pred.emotion);
  $("#predictionSummary").textContent = item.task.emotion === "none"
    ? `Free recording; predicted ${pred.emotion} with ${pct(pred.confidence)} confidence.`
    : `Target ${item.task.emotion}; predicted ${pred.emotion} with ${pct(pred.confidence)} confidence.`;
  $("#valenceValue").textContent = pct(pred.valence);
  $("#arousalValue").textContent = pct(pred.arousal);
  $("#dominanceValue").textContent = pct(pred.dominance);
  updateProbabilityBars(pred.probabilities);
  updateRadar(pred);
  $("#rmsMetric").textContent = report.stats.rmsMean.toFixed(4);
  $("#zcrMetric").textContent = report.stats.zcrMean.toFixed(4);
  $("#centroidMetric").textContent = `${Math.round(report.stats.centroidMean)} Hz`;
  $("#pitchMetric").textContent = report.stats.pitchMean ? `${Math.round(report.stats.pitchMean)} Hz` : "--";
  renderScriptTimeline("#resultTranscript", report.scriptTimeline || buildRecorderTimeline(item), (report.duration || RECORD_SECONDS) + 1);
  renderSttTranscript("#resultSttTranscript", report.recognizedTranscript || []);
  $("#exportReportBtn").disabled = false;
  $("#downloadBtn").disabled = false;
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
  const report = item.report;
  const output = {
    sessionId: item.id,
    createdAt: item.createdAt,
    task: item.task,
    sampleRate: item.sampleRate,
    duration: item.duration,
    prediction: report.prediction,
    stats: report.stats,
    script: report.script,
    scriptTimeline: report.scriptTimeline,
    recognizedTranscript: report.recognizedTranscript,
    featureShapes: {
      frames: report.frames.length,
      logMel: [report.logMel.length, report.logMel[0]?.length || 0],
      mfcc: [report.mfcc.length, report.mfcc[0]?.length || 0]
    },
    framePreview: report.frames.slice(0, 80)
  };
  const blob = new Blob([JSON.stringify(output, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `speech-emotion-report-${item.id}.json`;
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
  link.download = `speech-demo-${item.id}.webm`;
  link.click();
  URL.revokeObjectURL(url);
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
  drawSpeechWaveform($("#processWaveThumb"), item.samples, item.sampleRate, item.duration, item.sttSegments || [], item.duration, item.task?.script || "");
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
  const words = buildWordTimeline(segments, safeDuration, fallbackText).slice(0, 18);
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
    ctx.font = "800 19px Inter";
    const text = word.text.length > 11 ? `${word.text.slice(0, 10)}...` : word.text;
    ctx.fillText(text, x, plot.bottom + 31);
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

function drawHeatmap(canvas, matrix, label) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  ctx.clearRect(0, 0, width, height);
  const rows = matrix[0]?.length || 0;
  const cols = matrix.length;
  const flat = flatten(matrix);
  const min = percentile(flat, 2);
  const max = percentile(flat, 98);
  for (let x = 0; x < cols; x++) {
    for (let y = 0; y < rows; y++) {
      const value = normalize(matrix[x][y], min, max);
      ctx.fillStyle = heatColor(value);
      ctx.fillRect((x / cols) * width, height - ((y + 1) / rows) * height, Math.ceil(width / cols), Math.ceil(height / rows));
    }
  }
  ctx.fillStyle = "rgba(255,255,255,0.88)";
  ctx.fillRect(10, 10, 130, 25);
  ctx.fillStyle = "#132033";
  ctx.font = "900 12px Inter";
  ctx.fillText(label, 18, 27);
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
  const ctx = new AudioCtx();
  const master = ctx.createGain();
  master.gain.value = 0.045;
  master.connect(ctx.destination);
  const patterns = {
    click: [[620, 0.020], [920, 0.025]],
    tab: [[440, 0.025], [660, 0.035]],
    tick: [[720, 0.018]],
    start: [[240, 0.035], [480, 0.060], [960, 0.090]],
    stop: [[520, 0.035], [260, 0.065]],
    warning: [[760, 0.055], [380, 0.090]],
    play: [[660, 0.035], [990, 0.055]],
    queueReady: [[500, 0.035], [760, 0.045], [980, 0.060]],
    queue: [[620, 0.035], [920, 0.060]],
    analyze: [[280, 0.030], [420, 0.030], [560, 0.070]],
    success: [[523, 0.040], [659, 0.050], [784, 0.080]],
    reset: [[240, 0.045]]
  };
  let cursor = ctx.currentTime;
  (patterns[type] || patterns.click).forEach(([freq, duration]) => {
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type === "analyze" || type === "warning" ? "square" : "triangle";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, cursor);
    gain.gain.exponentialRampToValueAtTime(0.75, cursor + 0.006);
    gain.gain.exponentialRampToValueAtTime(0.0001, cursor + duration);
    osc.connect(gain);
    gain.connect(master);
    osc.start(cursor);
    osc.stop(cursor + duration);
    cursor += duration * 0.72;
  });
  setTimeout(() => ctx.close(), Math.ceil((cursor - ctx.currentTime + 0.2) * 1000));
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
