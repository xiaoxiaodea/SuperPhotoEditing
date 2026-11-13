const $ = (id) => document.getElementById(id);

const state = {
  jobId: null,
  job: null,
  selected: "A",
  poll: null,
  refinementPoll: null,
  refinementPosting: false,
  refinementRequest: null,
  versions: {},
  drafts: {},
};

const STEP_COPY = {
  queued: "排队",
  connect: "正在连接 Lightroom，必要时会自动打开",
  ingest: "正在读取底片",
  analyze: "正在做技术分析",
  restore: "正在安全清理画质",
  critique: "正在分析缺陷与读图卡",
  render: "正在选择路线并生成三套配方",
  qa: "正在自检配方",
  lightroom: "Lightroom 正在调色并检查成片",
  ready: "Lightroom 成片已完成，可以挑选了",
  error: "处理失败",
};

init();

async function init() {
  bind();
  refreshHealth();
  await restoreJob();
}

function rememberJob(id) {
  state.jobId = id || null;
  try { localStorage.setItem("superphoto.active-job", id || ""); } catch {}
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("job", id);
  else url.searchParams.delete("job");
  history.replaceState(null, "", url);
}

async function restoreJob() {
  let id = new URL(window.location.href).searchParams.get("job");
  if (id === null) {
    try { id = localStorage.getItem("superphoto.active-job"); } catch {}
  }
  // Older pages did not remember the active job. Recover their saved result
  // once; an explicit reset stores an empty string and stays on the upload UI.
  if (id === null) {
    try { id = (await api("/api/jobs/latest")).id; } catch { return; }
  }
  if (!id) return;
  rememberJob(id);
  show("work-stage");
  $("progress-title").textContent = "正在读取已保存的处理结果";
  await poll();
}

function bind() {
  // Build the new panel here so a pre-upgrade document can use this script too.
  $("judge").innerHTML = `
    <p class="kicker">保留喜欢的方向，再调到合心意</p>
    <h3 id="refine-title">微调这张成片</h3>
    <label class="version-label" for="refine-version">查看版本</label>
    <select id="refine-version"></select>
    <a id="version-download" class="version-download" download>下载当前成片</a>
    <p class="feedback-hint" id="refine-base"></p>
    <label for="notes">想改哪里？直接说就好</label>
    <textarea id="notes" rows="3" maxlength="1000" placeholder="例如：整体亮一点，保留现在的暖色。或：主体亮一点，背景暗一点。"></textarea>
    <p class="feedback-hint">Lightroom 会在当前版本上继续调整。完成后可对照微调前后，也可随时切回之前的版本。</p>
    <div class="judge-actions">
      <button type="button" class="btn primary" id="refine-submit">让 Lightroom 微调这张</button>
      <button type="button" class="btn ghost" id="again">再来一张</button>
    </div>
    <p id="refine-status" role="status" aria-live="polite"></p>
    <div id="refine-history"></div>`;
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
  $("error-retry").addEventListener("click", retryJob);
  $("refine-submit").addEventListener("click", submitRefinement);
  $("notes").addEventListener("input", () => {
    state.drafts[state.selected] = $("notes").value;
    updateRefinementButton();
  });
  $("refine-version").addEventListener("change", () => {
    state.versions[state.selected] = $("refine-version").value || null;
    selectSlot(state.selected);
  });
  bindHoldOrig();
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, textarea, select, button, [contenteditable='true']")) return;
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
    if (e.target.closest("input, textarea, select, button, [contenteditable='true']")) return;
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
  $("hero-edit").src = comparisonReference();
}

function showHeroEdit() {
  const v = (state.job?.variants || []).find((x) => x.slot === state.selected);
  if (v) $("hero-edit").src = selectedVersion()?.url || v.url;
}

function selectedVersion() {
  return (state.job?.refinements || []).find(r => r.id === state.versions[state.selected] && r.status === "ready");
}

function comparisonReference() {
  const version = selectedVersion();
  if (!version) return state.job?.source_preview;
  const parent = (state.job.refinements || []).find(r => r.id === version.parent_id && r.status === "ready");
  return parent?.url || state.job.variants.find(v => v.slot === state.selected)?.url;
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
    const semantic = h.semantic || {};
    if (semantic.available) bits.push("AI 主体分区可用");
    else if (semantic.enabled) bits.push("智能分区使用基础模式");
    if (h.darktable_cli) bits.push("darktable-cli 已发现");
    const lr = h.lightroom || {};
    if (lr.connected) bits.push("Lightroom 已连接");
    else if (lr.available) bits.push("处理时自动连接 Lightroom");
    $("health").textContent = bits.join(" · ");
    $("taste").textContent = h.taste || "";
  } catch {
    $("health").textContent = "后端未启动";
  }
}

async function upload(file) {
  clearTimeout(state.poll);
  clearTimeout(state.refinementPoll);
  state.versions = {};
  state.drafts = {};
  rememberJob(null);
  state.job = null;
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
    rememberJob(data.id);
    poll();
  } catch (err) {
    fail(String(err.message || err));
  }
}

async function retryJob() {
  if (!state.jobId) return;
  if (state.job?.status === "ready") {
    const url = new URL(window.location.href);
    url.searchParams.set("job", state.jobId);
    url.searchParams.set("ui", String(Date.now()));
    window.location.replace(url.href);
    return;
  }
  const button = $("error-retry");
  button.disabled = true;
  try {
    const res = await fetch(`/api/jobs/${state.jobId}/retry`, { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "重试失败");
    rememberJob(data.id);
    show("work-stage");
    $("progress-title").textContent = "正在重新处理这张照片";
    markSteps("connect");
    poll();
  } catch (err) {
    fail(String(err.message || err));
  } finally {
    button.disabled = false;
  }
}

async function poll() {
  clearInterval(state.poll);
  const jobId = state.jobId;
  const tick = async () => {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      if (state.jobId !== jobId) return;
      state.job = job;
      $("progress-title").textContent = STEP_COPY[job.step] || job.step;
      markSteps(job.step);
      if (job.status === "ready") {
        renderResult(job);
        return;
      } else if (job.status === "error") {
        fail(job.error || "unknown error");
        return;
      }
      state.poll = setTimeout(tick, 900);
    } catch (err) {
      if (state.jobId !== jobId) return;
      fail(String(err.message || err));
    }
  };
  await tick();
}

function renderResult(job) {
  state.job = job;
  show("result-stage");
  // A cached pre-upgrade document can load the current script. Upgrade the
  // two added elements in place before displaying already completed photos.
  const count = $("result-count") || document.querySelector(".studio-head .kicker");
  if (count) {
    count.id = "result-count";
    count.textContent = `${job.variants.length} 套后期成片`;
  }
  const omitted = job.delivery?.omitted || [];
  let deliveryNote = $("delivery-note");
  if (!deliveryNote) {
    deliveryNote = document.createElement("p");
    deliveryNote.id = "delivery-note";
    deliveryNote.className = "compare-hint hidden";
    $("variants").after(deliveryNote);
  }
  deliveryNote.classList.toggle("hidden", omitted.length === 0);
  deliveryNote.textContent = omitted.length
    ? `已保留 ${job.variants.length} 套成片。方案 ${omitted.map(v => v.slot).join("、")} 未达到要求，已撤下。`
    : "";
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
  if (c.intent && c.intent.natural_color) chip("保留自然光色");
  const iso = t.exif && (t.exif.ISOSpeedRatings || t.exif.PhotographicSensitivity);
  if (iso) chip(`ISO ${iso}`);

  const donot = $("donot");
  donot.innerHTML = (c.do_not || []).map((x) => `不要：${escapeHtml(x)}`).join("<br/>");

  const decision = job.style_decision || {};
  $("tone-card").textContent = decision.tone || "";
  $("color-card").textContent = decision.color || "";
  $("purpose-card").textContent = decision.purpose || "";
  $("route-title").textContent = decision.route_title || "";
  $("route-reason").textContent = decision.route_reason || "";
  $("manual-gaps").replaceChildren();

  const box = $("variants");
  box.innerHTML = "";
  for (const v of job.variants || []) {
    const btn = document.createElement("article");
    btn.className = "variant";
    btn.dataset.slot = v.slot;
    const sw = v.recipe?.style_workflow || {};
    const label = sw.route_label || ({ A: "自然精修", B: "光影层次", C: "不同氛围" }[v.slot] || "候选");
    const match = sw.mode !== "art-directed" && sw.match_score != null ? ` · 适配 ${Math.round(Number(sw.match_score) * 100)}%` : "";
    btn.innerHTML = `
      <button type="button" class="variant-preview" aria-label="查看方案 ${v.slot} ${escapeHtml(v.title)}">
        <img src="${v.url}" alt="${escapeHtml(v.title)}" />
      </button>
      <div class="meta">
        <div class="slot">方案 ${v.slot} · ${escapeHtml(label)}${match}${qaBadge(v.recipe)}</div>
        <h3>${escapeHtml(v.title)}</h3>
        <div class="ratings" aria-label="评价方案 ${v.slot}">
          <button type="button" class="rating" data-verdict="like" aria-pressed="false">满意</button>
          <button type="button" class="rating" data-verdict="dislike" aria-pressed="false">不满意</button>
        </div>
        <p class="rating-status" role="status" aria-live="polite"></p>
        <button type="button" class="variant-refine">微调这张成片</button>
        <p>${escapeHtml(v.why || "")}</p>
        ${workflowSummary(v.recipe?.workflow)}
        ${styleWorkflowSummary(sw)}
        <details>
          <summary>查看调整细节</summary>
          <ul>${(v.moves || []).map((m) => `<li>${escapeHtml(m)}</li>`).join("")}</ul>
          <pre>${escapeHtml(summarizeRecipe(v.recipe))}</pre>
        </details>
      </div>`;
    btn.querySelector(".variant-preview").addEventListener("click", () => selectSlot(v.slot));
    btn.querySelector(".variant-refine").addEventListener("click", () => {
      selectSlot(v.slot);
      $("judge").scrollIntoView({behavior: "smooth", block: "center"});
      $("notes").focus({preventScroll: true});
    });
    btn.querySelectorAll(".rating").forEach(button => {
      button.addEventListener("click", () => rateVariant(v.slot, button.dataset.verdict));
    });
    box.appendChild(btn);
    updateRating(v.slot);
  }
  for (const v of job.variants) {
    if (!(v.slot in state.versions)) {
      state.versions[v.slot] = (job.refinements || []).filter(r => r.slot === v.slot && r.status === "ready").at(-1)?.id || null;
    }
  }
  const defaultSlot = (job.variants.find((v) => v.slot === state.selected) || job.variants[0] || {}).slot;
  if (defaultSlot) selectSlot(defaultSlot);
  $("judge").classList.remove("hidden");
  watchRefinements();
  refreshHealth();
}

function selectSlot(slot) {
  const v = (state.job?.variants || []).find((x) => x.slot === slot);
  if (!v) return;
  state.selected = slot;
  document.querySelectorAll(".variant").forEach((el) => {
    el.classList.toggle("selected", el.dataset.slot === slot);
  });
  const version = selectedVersion();
  $("hero-edit").src = version?.url || v.url;
  $("hero-orig").src = comparisonReference();
  $("hero-label").textContent = `方案 ${v.slot} · ${v.title}${version ? " · 微调版" : ""}`;
  if ($("hero-before-label")) $("hero-before-label").textContent = version ? "微调前" : "原片";
  $("hold-orig").textContent = version ? "按住看微调前" : "按住看原片";
  $("notes").value = state.drafts[slot] || "";
  renderRefinementPanel();
}

function updateRating(slot) {
  const card = document.querySelector(`.variant[data-slot="${slot}"]`);
  if (!card) return;
  const verdict = state.job?.ratings?.[slot];
  card.querySelectorAll(".rating").forEach(button => {
    button.classList.toggle("selected", button.dataset.verdict === verdict);
    button.setAttribute("aria-pressed", String(button.dataset.verdict === verdict));
  });
}

async function rateVariant(slot, requested) {
  const jobId = state.jobId;
  const card = document.querySelector(`.variant[data-slot="${slot}"]`);
  if (!jobId || !card) return;
  const verdict = state.job.ratings?.[slot] === requested ? null : requested;
  card.querySelectorAll(".rating").forEach(button => button.disabled = true);
  const status = card.querySelector(".rating-status");
  status.textContent = "正在保存…";
  try {
    const res = await fetch(`/api/jobs/${jobId}/ratings/${slot}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({verdict}),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "评价保存失败，请重试");
    if (state.jobId !== jobId) return;
    state.job.ratings = {...state.job.ratings, [slot]: data.ratings[slot]};
    updateRating(slot);
    status.textContent = verdict === null ? "已撤销评价" : !data.learnable ? "已记下评价；这张旧成片缺少推荐所需的场景记录。"
      : verdict === "dislike" ? "已记下，类似画面会更少推荐这个方向。" : "已记下，类似画面会优先参考这个方向。";
    $("taste").textContent = data.summary || "";
  } catch (err) {
    status.textContent = String(err.message || err);
  } finally {
    card.querySelectorAll(".rating").forEach(button => button.disabled = false);
  }
}

function renderRefinementPanel() {
  const v = state.job?.variants.find(v => v.slot === state.selected);
  if (!v) return;
  const records = (state.job.refinements || []).filter(r => r.slot === v.slot);
  const versions = records.filter(r => r.status === "ready");
  $("refine-title").textContent = `微调方案 ${v.slot} · ${v.title}`;
  $("refine-version").innerHTML = `<option value="">最初成片</option>` + versions.map((r, i) =>
    `<option value="${escapeHtml(r.id)}">微调版 ${i + 1} · ${escapeHtml(r.summary || r.prompt)}</option>`).join("");
  $("refine-version").value = selectedVersion()?.id || "";
  const version = selectedVersion();
  $("refine-base").textContent = `接下来基于：${version ? `微调版 ${versions.findIndex(r => r.id === version.id) + 1}` : "最初成片"}。满意度评价记录的是上方卡片中的最初成片。`;
  $("version-download").href = version?.url || v.url;
  $("version-download").download = `SuperPhoto-${v.slot}${version ? "-微调" : ""}.jpg`;
  const active = records.find(r => ["queued", "running"].includes(r.status));
  const last = records.at(-1);
  const copy = {queued: "微调已排队，正在等待 Lightroom", connect: "正在连接 Lightroom", interpret: "正在理解你的微调意见",
    lightroom: "Lightroom 正在调整并导出", qa: "正在检查微调成片", ready: "微调版已完成，可以对照前后效果", error: "微调未完成"};
  $("refine-status").textContent = active ? (copy[active.step] || "正在微调") : last?.status === "error"
    ? `${last.error || "微调失败"} 调整前版本仍可查看和下载。` : last?.status === "ready" ? copy.ready : "";
  $("refine-history").innerHTML = records.length ? `<details><summary>查看微调记录（${records.length}）</summary><ol>${records.map(r =>
    `<li><p>${escapeHtml(r.prompt)}</p><span>${escapeHtml(r.status === "error" ? r.error : r.summary || copy[r.step] || r.status)}</span></li>`).join("")}</ol></details>` : "";
  updateRefinementButton();
}

function updateRefinementButton() {
  const active = (state.job?.refinements || []).some(r => r.slot === state.selected && ["queued", "running"].includes(r.status));
  $("refine-submit").disabled = state.refinementPosting || active || $("notes").value.trim().length < 2;
  $("refine-submit").textContent = active ? "正在微调这张…" : "让 Lightroom 微调这张";
}

async function submitRefinement() {
  if (state.refinementPosting || $("refine-submit").disabled) return;
  const jobId = state.jobId;
  const slot = state.selected;
  const prompt = $("notes").value.trim();
  const parentId = selectedVersion()?.id || null;
  const key = JSON.stringify([jobId, slot, parentId, prompt]);
  if (state.refinementRequest?.key !== key) state.refinementRequest = {key, id: crypto.randomUUID()};
  state.refinementPosting = true;
  updateRefinementButton();
  $("refine-status").textContent = "正在提交微调意见…";
  try {
    const res = await fetch(`/api/jobs/${jobId}/refinements`, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({slot, prompt, parent_id: parentId, request_id: state.refinementRequest.id})});
    const record = await res.json();
    if (!res.ok) throw new Error(typeof record.detail === "string" ? record.detail : "提交失败，请稍后重试");
    if (state.jobId !== jobId) return;
    state.job.refinements = [...(state.job.refinements || []).filter(r => r.id !== record.id), record];
    if (["ready", "error"].includes(record.status)) state.refinementRequest = null;
    renderRefinementPanel();
    watchRefinements();
  } catch (err) {
    if (state.jobId === jobId) $("refine-status").textContent = String(err.message || err);
  } finally {
    state.refinementPosting = false;
    updateRefinementButton();
  }
}

function watchRefinements() {
  clearTimeout(state.refinementPoll);
  const jobId = state.jobId;
  const active = (state.job?.refinements || []).filter(r => ["queued", "running"].includes(r.status));
  if (!active.length) return;
  state.refinementPoll = setTimeout(async () => {
    try {
      const updates = await Promise.all(active.map(r => api(`/api/jobs/${jobId}/refinements/${r.id}`)));
      if (state.jobId !== jobId) return;
      for (const record of updates) {
        state.job.refinements = state.job.refinements.map(r => r.id === record.id ? record : r);
        if (record.status === "ready") state.versions[record.slot] = record.id;
        if (["ready", "error"].includes(record.status)) state.refinementRequest = null;
      }
      selectSlot(state.selected);
    } catch (err) {
      if (state.jobId !== jobId) return;
      $("refine-status").textContent = "暂时读不到微调进度，正在重新连接。成片会保留。";
    }
    watchRefinements();
  }, 1500);
}

function reset() {
  clearInterval(state.poll);
  clearTimeout(state.refinementPoll);
  rememberJob(null);
  state.job = null;
  state.selected = "A";
  state.versions = {};
  state.drafts = {};
  state.refinementRequest = null;
  $("file").value = "";
  $("notes").value = "";
  show("drop-stage");
  refreshHealth();
}

function fail(message) {
  clearTimeout(state.poll);
  const completed = state.job?.status === "ready";
  $("error-title").textContent = completed ? "成片已完成，页面未能显示" : "处理失败";
  const kicker = document.querySelector("#error-stage .kicker");
  if (kicker) kicker.textContent = completed ? "成片已保存" : "没有跑通";
  const button = $("error-retry");
  button.textContent = completed ? "重新打开成片" : "重试这张照片";
  button.classList.toggle("hidden", !state.jobId || !(completed || state.job?.status === "error"));
  if (completed) {
    console.error("Result page failed to display:", message);
    show("error-stage");
    $("error-body").textContent = "成片已保存。点击“重新打开成片”即可再次查看，无需重新处理照片。";
    return;
  }
  let detail = String(message).split("\n")[0];
  if (detail.includes("Lightroom plugin not connected")) {
    detail = "Lightroom 尚未连接。请完成 Lightroom 的启动或关闭弹出的对话框，再点“重试这张照片”。";
  }
  show("error-stage");
  $("error-body").textContent = detail;
}

function show(id) {
  for (const stage of ["drop-stage", "work-stage", "result-stage", "error-stage"]) {
    $(stage).classList.toggle("hidden", stage !== id);
  }
}

function qaBadge(recipe) {
  if (recipe && recipe.lightroom_pixel_qa) {
    const review = recipe.visual_review || {};
    if (review.decision === "better" && Number(review.confidence) >= 0.65) {
      return '<span class="qa-score high">视觉复核有改善</span>';
    }
    return '<span class="qa-score mid">技术检查通过</span>';
  }
  if (recipe && recipe.preview_only_qa) return "";
  const n = recipe && recipe.qa_score;
  if (n == null || Number.isNaN(Number(n))) return "";
  const score = Math.round(Number(n));
  let cls = "mid";
  if (score >= 88) cls = "high";
  else if (score < 70) cls = "low";
  return `<span class="qa-score ${cls}">自检 ${score}</span>`;
}

function markSteps(step) {
  const order = ["connect", "ingest", "analyze", "restore", "critique", "render", "qa", "lightroom"];
  const idx = order.indexOf(step);
  document.querySelectorAll("#steps li").forEach((li) => {
    const i = order.indexOf(li.dataset.step);
    li.classList.remove("on", "done");
    if (step === "ready" || (idx >= 0 && i < idx)) li.classList.add("done");
    else if (i === idx) li.classList.add("on");
  });
}

function workflowSummary(workflow) {
  if (!workflow) return "";
  const row = (label, values) => {
    const items = (values || []).map(escapeHtml).join("、");
    return items ? `<div><strong>${label}</strong>${items}</div>` : "";
  };
  const ethicsLabel = {
    documentary: "纪实：不改内容",
    landscape: "风光：自然色保护",
    portrait: "人像：肤色保护",
    product: "产品：颜色准确",
  }[workflow.ethics];
  const ethics = ethicsLabel ? `<div><strong>制度：</strong>${ethicsLabel}</div>` : "";
  return `<div class="workflow-summary">
    ${row("修复：", workflow.repaired)}
    ${row("美化：", workflow.beautified)}
    ${row("保护：", workflow.protected)}
    ${ethics}
  </div>`;
}

function styleWorkflowSummary(style) {
  if (!style) return "";
  const notes = (style.adaptation_notes || []).map((x) => `<li>${escapeHtml(x)}</li>`).join("");
  const reason = style.selection_reason ? `<p>${escapeHtml(style.selection_reason)}</p>` : "";
  if (!notes && !reason) return "";
  return `<details class="style-workflow">
    <summary>为什么这样调整</summary>
    ${reason}
    ${notes ? `<strong>按本图调整</strong><ul>${notes}</ul>` : ""}
  </details>`;
}

function summarizeRecipe(recipe) {
  if (!recipe) return "";
  const lines = [];
  if (recipe.lut_mix > 0 && recipe.lut_mix < 0.98) {
    lines.push(`风格参考强度: ${Math.round(Number(recipe.lut_mix) * 100)}%`);
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
  if (recipe.curve?.rgb?.length) lines.push(`影调曲线: ${recipe.curve.rgb.length} 个控制点`);
  const wheels = recipe.color_grading || {};
  for (const [key, label] of [["shadow", "阴影"], ["midtone", "中间调"], ["highlight", "高光"]]) {
    if (Number(wheels[`${key}_sat`]) > 0) {
      lines.push(`${label}色轮: ${Math.round(wheels[`${key}_hue`] || 0)}° / 饱和度 ${Number(wheels[`${key}_sat`]).toFixed(0)}`);
    }
  }
  const targets = { subject: "主体", sky: "天空", background: "背景" };
  for (const mask of recipe.masks || []) {
    const moves = [];
    if (Math.abs(mask.exposure_ev || 0) > .01) moves.push(`曝光 ${Number(mask.exposure_ev).toFixed(2)} EV`);
    if (Math.abs(mask.highlights || 0) > .01) moves.push(`高光 ${Math.round(mask.highlights * 100)}`);
    if (Math.abs(mask.shadows || 0) > .01) moves.push(`阴影 ${Math.round(mask.shadows * 100)}`);
    if (moves.length) lines.push(`${targets[mask.target] || mask.target}蒙版: ${moves.join(" / ")}`);
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
