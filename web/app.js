const $ = (id) => document.getElementById(id);

const state = {
  jobId: null,
  job: null,
  selected: "B",
  poll: null,
};

const STEP_COPY = {
  queued: "排队",
  ingest: "正在读取底片",
  analyze: "正在做技术分析",
  critique: "正在诊断不足",
  render: "正在渲染三套配方",
  qa: "正在自检成片",
  ready: "可以挑选了",
  error: "处理失败",
};

init();

async function init() {
  bind();
  refreshHealth();
  for (const key of ["technical", "style", "restraint"]) {
    const input = $(`score-${key}`);
    const label = document.querySelector(`[data-score="${key}"]`);
    const sync = () => {
      label.textContent = input.value;
    };
    input.addEventListener("input", sync);
    sync();
  }
}

function bind() {
  const drop = $("drop");
  const file = $("file");
  $("pick").addEventListener("click", () => file.click());
  drop.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    file.click();
  });
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") file.click();
  });
  ["dragenter", "dragover"].forEach((ev) => {
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    });
  });
  ["dragleave", "drop"].forEach((ev) => {
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("over");
    });
  });
  drop.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) upload(f);
  });
  file.addEventListener("change", () => {
    if (file.files[0]) upload(file.files[0]);
  });
  $("again").addEventListener("click", reset);
  $("error-reset").addEventListener("click", reset);
  $("submit").addEventListener("click", submitFeedback);
  bindHoldOrig();
  document.addEventListener("keydown", (e) => {
    if (!$("result-stage") || $("result-stage").classList.contains("hidden")) return;
    if (e.key === "1") selectSlot("A");
    if (e.key === "2") selectSlot("B");
    if (e.key === "3") selectSlot("C");
    if (e.key === " " || e.code === "Space") {
      e.preventDefault();
      showHeroOriginal();
    }
  });
  document.addEventListener("keyup", (e) => {
    if (e.key === " " || e.code === "Space") showHeroEdit();
  });
}

function bindHoldOrig() {
  const btn = $("hold-orig");
  const img = $("hero-edit");
  const down = (e) => {
    e.preventDefault();
    btn.classList.add("down");
    showHeroOriginal();
  };
  const up = () => {
    btn.classList.remove("down");
    showHeroEdit();
  };
  for (const el of [btn, img]) {
    el.addEventListener("pointerdown", down);
    el.addEventListener("pointerup", up);
    el.addEventListener("pointerleave", up);
    el.addEventListener("pointercancel", up);
  }
}

function showHeroOriginal() {
  if (!state.job) return;
  $("hero-edit").src = state.job.source_preview;
}

function showHeroEdit() {
  const v = (state.job?.variants || []).find((x) => x.slot === state.selected);
  if (v) $("hero-edit").src = v.url;
}

async function refreshHealth() {
  try {
    const h = await api("/api/health");
    const bits = [];
    const ollama = h.ollama || {};
    if (h.critic === "ollama") {
      bits.push(`本地诊断 ${ollama.resolved_model || h.critic_model}`);
    } else if (ollama.ok && !ollama.ready) {
      bits.push(`Ollama 已连接，正在准备 ${ollama.model || "视觉模型"}`);
    } else if (h.critic === "grok") {
      bits.push(`Grok 诊断（${h.critic_model || h.model}）`);
    } else {
      bits.push("本地规则诊断");
    }
    bits.push(h.rawpy ? "RAW 可用" : "仅 JPEG/TIFF/PNG");
    if (h.darktable_cli) bits.push("darktable-cli 已发现");
    $("health").textContent = bits.join(" · ");
    $("taste").textContent = h.taste || "";
  } catch {
    $("health").textContent = "后端未启动";
  }
}

async function upload(file) {
  show("work-stage");
  $("progress-file").textContent = `${file.name} · ${prettySize(file.size)}`;
  $("progress-title").textContent = "正在上传";
  markSteps("ingest");
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/api/jobs", { method: "POST", body });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "upload failed");
    }
    const data = await res.json();
    state.jobId = data.id;
    poll();
  } catch (err) {
    fail(String(err.message || err));
  }
}

async function poll() {
  clearInterval(state.poll);
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${state.jobId}`);
      state.job = job;
      $("progress-title").textContent = STEP_COPY[job.step] || job.step;
      markSteps(job.step);
      if (job.status === "ready") {
        clearInterval(state.poll);
        renderResult(job);
      } else if (job.status === "error") {
        clearInterval(state.poll);
        fail(job.error || "unknown error");
      }
    } catch (err) {
      clearInterval(state.poll);
      fail(String(err.message || err));
    }
  };
  await tick();
  state.poll = setInterval(tick, 900);
}

function renderResult(job) {
  show("result-stage");
  $("orig-img").src = job.source_preview;
  $("hero-orig").src = job.source_preview;
  $("orig-name").textContent = job.original_name || "";
  const c = job.critique || {};
  $("summary").textContent = c.summary || "";
  const sourceLabel = {
    ollama: "本地 Qwen 诊断",
    grok: "Grok 诊断",
    local: "规则诊断",
  }[c.source] || "诊断";
  $("genre").textContent = `${c.genre || "general"} · ${sourceLabel}`;
  const issues = $("issues");
  issues.innerHTML = "";
  for (const issue of c.issues || []) {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${issue.dim}</strong><div class="bar"><span style="width:${Math.round((issue.severity || 0) * 100)}%"></span></div>${escapeHtml(issue.note || "")}`;
    issues.appendChild(li);
  }
  const chips = $("tech-chips");
  chips.innerHTML = "";
  const t = job.technical || {};
  const chip = (label) => {
    const s = document.createElement("span");
    s.textContent = label;
    chips.appendChild(s);
  };
  if (t.mean_luma != null) chip(`亮度 ${t.mean_luma.toFixed(2)}`);
  if (t.suggested_exposure_ev != null) chip(`建议 ${t.suggested_exposure_ev >= 0 ? "+" : ""}${t.suggested_exposure_ev.toFixed(2)} EV`);
  if (t.highlight_clip != null) chip(`高光裁切 ${(t.highlight_clip * 100).toFixed(1)}%`);
  if (t.shadow_clip != null) chip(`阴影裁切 ${(t.shadow_clip * 100).toFixed(1)}%`);
  if (t.width && t.height) chip(`${t.width}×${t.height}`);
  if (t.source_kind) chip(t.source_kind.toUpperCase());
  if (t.horizon_tilt_deg != null && Math.abs(t.horizon_tilt_deg) > 0.4) {
    chip(`地平线 ${t.horizon_tilt_deg > 0 ? "+" : ""}${t.horizon_tilt_deg.toFixed(1)}°`);
  }
  if (c.lut_mix != null) chip(`LUT 用量建议 ${Math.round(c.lut_mix * 100)}%`);
  const iso = t.exif && (t.exif.ISOSpeedRatings || t.exif.PhotographicSensitivity);
  if (iso) chip(`ISO ${iso}`);

  const donot = $("donot");
  donot.innerHTML = (c.do_not || []).map((x) => `不要：${escapeHtml(x)}`).join("<br/>");

  const box = $("variants");
  box.innerHTML = "";
  for (const v of job.variants || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "variant";
    btn.dataset.slot = v.slot;
    btn.innerHTML = `
      <img src="${v.url}" alt="${escapeHtml(v.title)}" />
      <div class="meta">
        <div class="slot">方案 ${v.slot}${qaBadge(v.recipe)}</div>
        <h3>${escapeHtml(v.title)}</h3>
        <p>${escapeHtml(v.why || "")}</p>
        <details>
          <summary>配方</summary>
          <ul>${(v.moves || []).map((m) => `<li>${escapeHtml(m)}</li>`).join("")}</ul>
          <pre>${escapeHtml(summarizeRecipe(v.recipe))}</pre>
        </details>
      </div>`;
    btn.addEventListener("click", () => selectSlot(v.slot));
    box.appendChild(btn);
  }
  const defaultSlot = (job.variants.find((v) => v.slot === "B") || job.variants[0] || {}).slot;
  if (defaultSlot) selectSlot(defaultSlot);
  $("judge").classList.remove("hidden");
  $("thanks").classList.add("hidden");
  $("submit").disabled = false;
  refreshHealth();
}

function selectSlot(slot) {
  state.selected = slot;
  document.querySelectorAll(".variant").forEach((el) => {
    el.classList.toggle("selected", el.dataset.slot === slot);
  });
  const v = (state.job?.variants || []).find((x) => x.slot === slot);
  if (v) {
    $("hero-edit").src = v.url;
    $("hero-label").textContent = `方案 ${v.slot} · ${v.title}`;
  }
}

async function submitFeedback() {
  if (!state.jobId || !state.selected) return;
  $("submit").disabled = true;
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        chosen_slot: state.selected,
        technical_score: Number($("score-technical").value),
        style_score: Number($("score-style").value),
        restraint_score: Number($("score-restraint").value),
        notes: $("notes").value,
      }),
    });
    if (!res.ok) throw new Error("feedback failed");
    const data = await res.json();
    $("thanks").textContent = data.summary || "已记下。";
    $("thanks").classList.remove("hidden");
    $("taste").textContent = data.summary || "";
  } catch (err) {
    $("submit").disabled = false;
    $("thanks").textContent = String(err.message || err);
    $("thanks").classList.remove("hidden");
  }
}

function reset() {
  clearInterval(state.poll);
  state.jobId = null;
  state.job = null;
  $("file").value = "";
  $("notes").value = "";
  show("drop-stage");
  refreshHealth();
}

function fail(message) {
  show("error-stage");
  $("error-body").textContent = message;
}

function show(id) {
  for (const stage of ["drop-stage", "work-stage", "result-stage", "error-stage"]) {
    $(stage).classList.toggle("hidden", stage !== id);
  }
}

function qaBadge(recipe) {
  const n = recipe && recipe.qa_score;
  if (n == null || Number.isNaN(Number(n))) return "";
  const score = Math.round(Number(n));
  let cls = "mid";
  if (score >= 88) cls = "high";
  else if (score < 70) cls = "low";
  return `<span class="qa-score ${cls}">自检 ${score}</span>`;
}

function markSteps(step) {
  const order = ["ingest", "analyze", "critique", "render", "qa"];
  const idx = order.indexOf(step);
  document.querySelectorAll("#steps li").forEach((li) => {
    const i = order.indexOf(li.dataset.step);
    li.classList.remove("on", "done");
    if (step === "ready" || (idx >= 0 && i < idx)) li.classList.add("done");
    else if (i === idx) li.classList.add("on");
  });
}

function summarizeRecipe(recipe) {
  if (!recipe) return "";
  const lines = [];
  if (recipe.lut_mix != null && recipe.lut_mix < 0.98) {
    lines.push(`LUT 用量: ${Math.round(Number(recipe.lut_mix) * 100)}%`);
  }
  const g = recipe.geometry;
  if (g && g.apply) {
    if (Math.abs(g.rotate_deg || 0) > 0.1) lines.push(`旋转: ${Number(g.rotate_deg).toFixed(1)}°`);
    const trim =
      (g.crop_left || 0) + (g.crop_top || 0) + (1 - (g.crop_right ?? 1)) + (1 - (g.crop_bottom ?? 1));
    if (trim > 0.01) lines.push("构图: 有裁切");
  }
  const keys = [
    "exposure_ev",
    "temperature",
    "highlights",
    "shadows",
    "contrast",
    "saturation",
    "clarity",
    "dehaze",
    "vignette",
  ];
  for (const k of keys) {
    if (Math.abs(recipe[k] || 0) > 0.01) lines.push(`${k}: ${Number(recipe[k]).toFixed(2)}`);
  }
  const loc = recipe.local;
  if (loc) {
    const localKeys = [
      ["color_clean", "颜色干净"],
      ["subject_sharpen", "主体锐化"],
      ["subject_clarity", "主体清晰"],
      ["background_desat", "背景降饱和"],
      ["background_darken", "背景压暗"],
      ["clutter_suppress", "压干扰"],
    ];
    for (const [k, label] of localKeys) {
      if (Math.abs(loc[k] || 0) > 0.02) lines.push(`${label}: ${Number(loc[k]).toFixed(2)}`);
    }
  }
  return lines.join("\n");
}

function prettySize(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function api(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}
