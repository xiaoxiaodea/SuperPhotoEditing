const $ = (id) => document.getElementById(id);

const state = {
  jobId: null,
  job: null,
  selected: "A",
  order: [],          // slots in display order; number keys follow it
  shownJob: null,     // the job whose cards are on screen
  poll: null,
  refinementPoll: null,
  refinementPosting: false,
  refinementRequest: null,
  versions: {},
  drafts: {},
  lightcraftEnabled: null,
  previewUrl: null,
};

const STEP_COPY = {
  queued: "排队",
  connect: "正在准备 LightCraft",
  ingest: "正在读取底片",
  analyze: "正在做技术分析",
  restore: "正在安全清理画质",
  critique: "正在分析缺陷与读图卡",
  render: "正在选择路线并生成三套配方",
  qa: "正在自检配方",
  lightroom: "LightCraft 正在调色并检查成片",
  ready: "成片已完成，可以挑选了",
  error: "处理失败",
};

init();

async function init() {
  bind();
  try {
    await ensureAuthenticated();
    await refreshHealth();
    await restoreJob();
  } catch (err) {
    $("health").textContent = err.message || "无法连接服务";
  }
}

async function ensureAuthenticated() {
  const response = await fetch("/api/auth/status", {cache: "no-store"});
  if (!response.ok) throw new Error("无法检查登录状态");
  const status = await response.json();
  if (!status.required || status.authenticated) return;
  document.querySelector("main").classList.add("hidden");
  $("auth-stage").classList.remove("hidden");
  $("auth-token").focus();
  await new Promise((resolve) => {
    $("auth-form").onsubmit = async (event) => {
      event.preventDefault();
      const button = event.target.querySelector("button");
      button.disabled = true;
      $("auth-error").textContent = "";
      try {
        const login = await fetch("/api/auth/login", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({token: $("auth-token").value}),
        });
        if (!login.ok) throw new Error("令牌无效，请重新输入");
        $("auth-token").value = "";
        $("auth-stage").classList.add("hidden");
        document.querySelector("main").classList.remove("hidden");
        $("auth-form").onsubmit = null;
        resolve();
      } catch (error) {
        $("auth-error").textContent = error.message;
      } finally { button.disabled = false; }
    };
  });
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
  $("judge").innerHTML = `
    <p class="kicker">当前方案</p>
    <h2 id="refine-title">微调这张成片</h2>
    <div class="version-row"><label class="version-label" for="refine-version">成片版本</label><span class="version-label">可随时切换</span></div>
    <select id="refine-version"></select>
    <a id="version-download" class="btn primary version-download" download>下载当前成片 <span aria-hidden="true">↓</span></a>
    <div class="full-export" id="full-export">
      <span class="version-label">原尺寸导出</span>
      <button type="button" data-full="jpeg" title="原尺寸 JPEG，适合分享和冲印">JPEG</button>
      <button type="button" data-full="tiff" title="原尺寸 16 位 TIFF，适合继续精修；文件很大">16 位 TIFF</button>
    </div>
    <p class="feedback-hint" id="full-status" role="status" aria-live="polite"></p>
    <div class="refine-section">
      <label for="notes">还想做一点调整？</label>
      <textarea id="notes" rows="4" maxlength="1000" placeholder="说说你想要的效果，例如：主体亮一点，保留现在的暖色。"></textarea>
      <div class="prompt-suggestions" aria-label="微调建议"><button type="button" data-prompt="整体亮一点，保留现在的暖色">整体亮一点</button><button type="button" data-prompt="主体亮一点，背景暗一点">突出主体</button><button type="button" data-prompt="饱和度低一点">降低饱和度</button></div>
      <p class="feedback-hint" id="refine-base"></p>
      <button type="button" class="btn ghost" id="refine-submit">开始微调 <span aria-hidden="true">↗</span></button>
      <p id="refine-status" role="status" aria-live="polite"></p>
      <div id="refine-history"></div>
    </div>
    <button type="button" class="btn ghost" id="again">导入下一张照片</button>`;
  $("new-photo").addEventListener("click", reset);
  $("vision-open").addEventListener("click", openVisionDialog);
  $("vision-close").addEventListener("click", () => $("vision-dialog").close());
  $("vision-dialog").addEventListener("close", () => { $("vision-body").innerHTML = ""; refreshHealth(); });
  $("view-single").addEventListener("click", () => setCompareView(false));
  $("view-compare").addEventListener("click", () => setCompareView(true));
  ["edit", "info"].forEach(name => {
    const tab = $(`tab-${name}`);
    tab.addEventListener("click", () => selectInspector(name));
    tab.addEventListener("keydown", event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
      event.preventDefault();
      const next = event.key === "Home" ? "edit" : event.key === "End" ? "info" : name === "edit" ? "info" : "edit";
      selectInspector(next);
      $(`tab-${next}`).focus();
    });
  });
  document.querySelectorAll("[data-full]").forEach(button => button.addEventListener("click", () => exportFullSize(button.dataset.full)));
  document.querySelectorAll("[data-prompt]").forEach(button => {
    button.addEventListener("click", () => {
      $("notes").value = button.dataset.prompt;
      $("notes").dispatchEvent(new Event("input"));
      $("notes").focus();
    });
  });
  const drop = $("drop");
  const file = $("file");
  $("pick").addEventListener("click", () => file.click());
  drop.addEventListener("submit", event => event.preventDefault());
  drop.addEventListener("click", (e) => {
    if (e.target.closest("button, input")) return;
    file.click();
  });
  drop.addEventListener("keydown", (e) => {
    if (e.target === drop && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      file.click();
    }
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
    if (e.target.closest("input, textarea, select, [contenteditable='true']")) return;
    if (!$("result-stage") || $("result-stage").classList.contains("hidden")) return;
    // Number keys pick the cards in the order they are shown (best-matching first), not by letter.
    if (/^[1-9]$/.test(e.key) && state.order[Number(e.key) - 1]) selectSlot(state.order[Number(e.key) - 1]);
    if (e.target.closest("button")) return;
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

function setCompareView(compare) {
  $("hero-compare").classList.toggle("compare", compare);
  $("view-single").classList.toggle("active", !compare);
  $("view-compare").classList.toggle("active", compare);
  $("view-single").setAttribute("aria-pressed", String(!compare));
  $("view-compare").setAttribute("aria-pressed", String(compare));
}

function selectInspector(name) {
  for (const tab of ["edit", "info"]) {
    const active = tab === name;
    $(`tab-${tab}`).setAttribute("aria-selected", String(active));
    $(`tab-${tab}`).tabIndex = active ? 0 : -1;
    $(`${tab}-panel`).classList.toggle("hidden", !active);
  }
}

function bindHoldOrig() {
  const btn = $("hold-orig");
  const img = $("hero-edit");
  const down = (e) => {
    e.preventDefault();
    if (e.currentTarget === img) $("hero-compare").focus({preventScroll: true});
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
    el.addEventListener("contextmenu", (event) => event.preventDefault());
  }
  btn.addEventListener("keydown", event => {
    if (event.key === " " || event.key === "Enter") down(event);
  });
  btn.addEventListener("keyup", event => {
    if (event.key === " " || event.key === "Enter") up();
  });
  btn.addEventListener("blur", up);
  window.addEventListener("blur", up);
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
    const vision = h.vision || {};
    const vault = vision.vault || {};
    if (vision.ready && vision.provider === "vault") {
      bits.push(`视觉模型已解锁（启用 ${vault.count || 0} 个，当前首选 ${vision.model}）`);
    } else if (vision.ready && vision.provider !== "local") {
      bits.push(`${vision.provider === "ollama" ? "本地视觉" : "云端视觉"} ${vision.model || h.critic_model || ""}`);
      if (vision.provider !== "ollama" && !vision.cloud_connectivity_checked) bits.push("API 已配置，调用时验证");
    } else if (h.critic === "ollama") {
      bits.push(`本地视觉 ${ollama.resolved_model || h.critic_model}`);
    } else {
      bits.push("本地规则诊断");
      if (vision.requested_provider && !["auto", "local"].includes(vision.requested_provider)) bits.push("视觉模型配置不可用");
    }
    if (vault.exists && !vault.unlocked) bits.push("视觉模型保险箱已锁定，点「视觉模型」输入口令");
    $("vision-open").dataset.state = vault.exists ? (vault.unlocked ? "unlocked" : "locked") : "none";
    bits.push(h.rawpy ? "RAW 可用" : "仅 JPEG/TIFF/PNG");
    const semantic = h.semantic || {};
    if (semantic.available) bits.push("AI 主体分区可用");
    else if (semantic.enabled) bits.push("智能分区使用基础模式");
    if (h.darktable_cli) bits.push("darktable-cli 已发现");
    const lr = h.lightroom || {};
    state.lightcraftEnabled = Boolean(lr.enabled);
    $("drop").querySelector(".lede").textContent = state.lightcraftEnabled
      ? "先分析照片，再生成三套修图方案。完成后可以选择喜欢的方向，并继续微调。"
      : "先分析照片，再生成三套基础修图方案。当前模式支持下载和评价，暂不支持文字微调。";
    if (lr.connected) bits.push("LightCraft 已就绪");
    else if (lr.enabled && lr.available) bits.push("LightCraft 暂不可用");
    else if (lr.enabled) bits.push("尚未安装 LightCraft");
    else bits.push("基础渲染模式");
    $("health").textContent = bits.join(" · ");
    document.querySelector(".engine-status").dataset.state = lr.enabled && !lr.connected ? "error" : "ready";
    $("taste").textContent = h.taste || "";
    updateRefinementButton();
  } catch {
    document.querySelector(".engine-status").dataset.state = "error";
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
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  $("processing-preview").src = state.previewUrl;
  $("processing-preview").classList.remove("hidden");
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
      if (!state.previewUrl && job.source_preview) {
        $("processing-preview").src = job.source_preview;
        $("processing-preview").classList.remove("hidden");
      }
      $("progress-file").textContent = job.original_name || $("progress-file").textContent;
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
  const similar = job.delivery?.similar_slots || [];
  const deliveryMessages = [];
  if (omitted.length) {
    deliveryMessages.push(`已保留 ${job.variants.length} 套成片。方案 ${omitted.map(v => v.slot).join("、")} 未达到要求，已撤下。`);
  }
  if (similar.length) {
    deliveryMessages.push(`方案 ${similar.join("、")} 与其他方案较接近，已保留有效成片供比较。`);
  }
  deliveryNote.classList.toggle("hidden", deliveryMessages.length === 0);
  deliveryNote.textContent = deliveryMessages.join(" ");
  $("orig-img").src = job.source_preview;
  $("hero-orig").src = job.source_preview;
  $("orig-name").textContent = job.original_name || "";
  const c = job.critique || {};
  $("summary").textContent = c.summary || "";
  const sourceLabel = {
    ollama: "本地 Qwen 诊断",
    grok: "Grok 诊断（临时替代）",
    xai: "Grok 视觉诊断（临时替代）",
    vault: "视觉模型诊断",
    openai: "云端视觉诊断",
    local: "规则诊断",
  }[c.source] || "诊断";
  $("genre").textContent = `${c.genre || "general"} · ${sourceLabel}`;
  const issues = $("issues");
  issues.innerHTML = "";
  for (const issue of c.issues || []) {
    const li = document.createElement("li");
    const dimension = {exposure: "曝光", color: "色彩", contrast: "反差", detail: "细节", composition: "构图", noise: "噪点", white_balance: "白平衡"}[issue.dim] || issue.dim;
    const severity = issue.severity >= .7 ? "需要关注" : issue.severity >= .35 ? "建议调整" : "轻微";
    li.innerHTML = `<div class="issue-heading"><strong>${escapeHtml(dimension)}</strong><span>${severity}</span></div>${escapeHtml(issue.note || "")}`;
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
  const ordered = versionOrder(job.variants || []);
  state.order = ordered.map(v => v.slot);
  const many = ordered.length > 3;
  box.classList.toggle("many", many);
  box.style.setProperty("--count", String(ordered.length <= 6 ? ordered.length : Math.ceil(ordered.length / 2)));
  document.body.classList.toggle("many-versions", many);
  for (const v of ordered) {
    const btn = document.createElement("article");
    btn.className = "variant";
    btn.dataset.slot = v.slot;
    const direction = v.recipe?.style_workflow?.direction_title || "";
    btn.innerHTML = `
      <div class="variant-main">
        <button type="button" class="variant-preview" aria-label="查看方案 ${v.slot} ${escapeHtml(v.title)}" aria-pressed="false">
          <img src="${v.url}" alt="${escapeHtml(v.title)}" /><span class="slot-letter">${escapeHtml(v.slot)}</span>
        </button>
        <div class="meta">
          <p class="slot" title="${escapeHtml(direction)}">方案 ${escapeHtml(v.slot)}${direction ? ` · ${escapeHtml(direction)}` : ""}</p>
          <button type="button" class="variant-title" title="${escapeHtml(v.title)}" aria-label="选择方案 ${escapeHtml(v.slot)} ${escapeHtml(v.title)}">${escapeHtml(v.title)}</button>
          ${qaBadge(v.recipe)}
        </div>
      </div>
      <div class="ratings" aria-label="评价方案 ${v.slot}">
        <button type="button" class="rating" data-verdict="pass" aria-pressed="false" title="这种处理不适合这张">pass</button>
        <button type="button" class="rating" data-verdict="keep" aria-pressed="false">留下</button>
        <button type="button" class="rating" data-verdict="favorite" aria-pressed="false" title="每张照片只有一个最爱">最爱</button>
      </div>
      <p class="rating-status" role="status" aria-live="polite"></p>`;
    btn.querySelector(".variant-preview").addEventListener("click", () => selectSlot(v.slot));
    btn.querySelector(".variant-title").addEventListener("click", () => selectSlot(v.slot));
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
  // A photo opens on its first card (the version this user is likeliest to keep); a refresh of the same photo
  // keeps the version being looked at.
  const same = state.shownJob === job.id && job.variants.find((v) => v.slot === state.selected);
  state.shownJob = job.id;
  const defaultSlot = (same || ordered[0] || {}).slot;
  if (defaultSlot) selectSlot(defaultSlot);
  $("judge").classList.remove("hidden");
  watchRefinements();
  refreshHealth();
}

// The server ranks the versions by what this user kept on similar pictures (sidecar experience.rank, A first on a
// tie); versions made before that keep their letter order.
function versionOrder(variants) {
  const rank = (v) => Number.isInteger(v.recipe?.experience?.rank) ? v.recipe.experience.rank : 99;
  return [...variants].sort((a, b) => rank(a) - rank(b) || a.slot.localeCompare(b.slot));
}

function selectSlot(slot) {
  const v = (state.job?.variants || []).find((x) => x.slot === slot);
  if (!v) return;
  state.selected = slot;
  document.querySelectorAll(".variant").forEach((el) => {
    const selected = el.dataset.slot === slot;
    el.classList.toggle("selected", selected);
    el.querySelector(".variant-preview").setAttribute("aria-pressed", String(selected));
  });
  $("selected-details").innerHTML = `<p>${escapeHtml(v.why || "")}</p>
    ${workflowSummary(v.recipe?.workflow)}${styleWorkflowSummary(v.recipe?.style_workflow)}
    <details><summary>查看配方参数</summary><ul>${(v.moves || []).map(m => `<li>${escapeHtml(m)}</li>`).join("")}</ul><pre>${escapeHtml(summarizeRecipe(v.recipe))}</pre></details>`;
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
    // A new favourite turns the previous one into a kept version, so refresh every card.
    state.job.ratings = {...data.ratings};
    (state.job.variants || []).forEach(v => updateRating(v.slot));
    status.textContent = verdict === null ? "已撤销评价" : !data.learnable ? "已记下评价；这张旧成片缺少推荐所需的场景记录。"
      : verdict === "pass" ? "已记下：类似画面会少用这个方向，但不会完全排除。"
      : verdict === "favorite" ? "已设为最爱，类似画面会优先参考这个方向。" : "已留下，类似画面会参考这个方向。";
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
  $("refine-title").textContent = `${v.slot} / ${v.title}`;
  $("refine-version").innerHTML = `<option value="">最初成片</option>` + versions.map((r, i) =>
    `<option value="${escapeHtml(r.id)}">微调版 ${i + 1} · ${escapeHtml(r.summary || r.prompt)}</option>`).join("");
  $("refine-version").value = selectedVersion()?.id || "";
  const version = selectedVersion();
  $("refine-base").textContent = `基于${version ? `微调版 ${versions.findIndex(r => r.id === version.id) + 1}` : "最初成片"}继续调整。方案评价针对最初成片。`;
  $("version-download").href = version?.url || v.url;
  $("version-download").download = `SuperPhoto-${v.slot}${version ? "-微调" : ""}.jpg`;
  const active = records.find(r => ["queued", "running"].includes(r.status));
  const last = records.at(-1);
  const copy = {queued: "微调已排队，正在等待 LightCraft", connect: "正在准备 LightCraft", interpret: "正在理解你的微调意见",
    lightroom: "LightCraft 正在调整并导出", qa: "正在检查微调成片", ready: "微调版已完成，可以对照前后效果", error: "微调未完成"};
  $("refine-status").textContent = active ? (copy[active.step] || "正在微调") : last?.status === "error"
    ? `${last.error || "微调失败"} 调整前版本仍可查看和下载。` : last?.status === "ready" ? copy.ready : "";
  $("refine-history").innerHTML = records.length ? `<details><summary>查看微调记录（${records.length}）</summary><ol>${records.map(r =>
    `<li><p>${escapeHtml(r.prompt)}</p><span>${escapeHtml(r.status === "error" ? r.error : r.summary || copy[r.step] || r.status)}</span></li>`).join("")}</ol></details>` : "";
  updateRefinementButton();
}

function updateRefinementButton() {
  const active = (state.job?.refinements || []).some(r => r.slot === state.selected && ["queued", "running"].includes(r.status));
  const unavailable = state.lightcraftEnabled === false;
  $("notes").disabled = unavailable;
  $("refine-submit").disabled = unavailable || state.refinementPosting || active || $("notes").value.trim().length < 2;
  $("refine-submit").textContent = unavailable ? "当前模式暂不支持微调" : state.refinementPosting ? "正在提交…" : active ? "正在微调这张…" : "开始微调 ↗";
  document.querySelectorAll("[data-prompt]").forEach(button => button.disabled = unavailable || state.refinementPosting || active);
  if (unavailable) $("refine-status").textContent = "基础模式的成片可以下载和评价；文字微调需要启用完整出片引擎。";
  updateFullExport();
}

function fullExportKey() {
  return JSON.stringify([state.jobId, state.selected, selectedVersion()?.id || null]);
}

function updateFullExport() {
  // The base mode keeps no LightCraft develop record, so there is nothing to render at full size.
  const unavailable = state.lightcraftEnabled === false;
  $("full-export").hidden = unavailable;
  const busy = state.fullExport && !state.fullExport.done;
  document.querySelectorAll("[data-full]").forEach(button => button.disabled = Boolean(busy));
  if (unavailable) { $("full-status").textContent = ""; return; }
  if (!state.fullExport || state.fullExport.key === fullExportKey()) $("full-status").textContent = state.fullExport?.text || "";
  else if (state.fullExport.done) $("full-status").textContent = "";
}

async function exportFullSize(format) {
  if (state.fullExport && !state.fullExport.done) return;
  const jobId = state.jobId;
  const slot = state.selected;
  const version = selectedVersion()?.id || null;
  const label = format === "tiff" ? "16 位 TIFF" : "JPEG";
  const job = state.fullExport = {key: fullExportKey(), done: false,
    text: `正在导出原尺寸 ${label}，大图约需 10–20 秒…`};
  updateFullExport();
  try {
    const res = await fetch(`/api/jobs/${jobId}/full`, {method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({slot, version, format})});
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "原尺寸导出失败，请稍后重试");
    const link = document.createElement("a");
    link.href = data.url;
    link.download = data.filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    const size = data.width && data.height ? `${data.width}×${data.height}，` : "";
    job.text = `已导出原尺寸 ${label}（${size}${(data.bytes / 1048576).toFixed(1)} MB）。`;
  } catch (err) {
    job.text = String(err.message || err);
  } finally {
    job.done = true;
    if (state.fullExport === job) updateFullExport();
  }
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
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  $("processing-preview").removeAttribute("src");
  $("processing-preview").classList.add("hidden");
  setCompareView(false);
  selectInspector("edit");
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
    detail = "LightCraft 还不能出片。请运行 install-lightcraft，再点“重试这张照片”。";
  }
  show("error-stage");
  $("error-body").textContent = detail;
}

function show(id) {
  $("new-photo").classList.toggle("hidden", !["result-stage", "error-stage"].includes(id));
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
    if (review.decision === "checked") {   // the batched damage check of a many-version job
      return (review.damage || []).length ? '<span class="qa-score mid">有轻微瑕疵</span>'
        : '<span class="qa-score high">未见损坏</span>';
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

function summarizeDirected(recipe) {
  const develop = recipe.plan?.develop || {};
  const names = {
    exposure: "曝光", contrast: "对比", highlights: "高光", shadows: "阴影", whites: "白色", blacks: "黑色",
    temp: "色温", tint: "色调", saturation: "饱和度", vibrance: "自然饱和度", clarity: "清晰度",
    texture: "纹理", dehaze: "去朦胧",
  };
  const fmt = (key, value) => key === "exposure"
    ? `${names[key]} ${Number(value) >= 0 ? "+" : ""}${Number(value).toFixed(2)} EV`
    : `${names[key] || key} ${Number(value) >= 0 ? "+" : ""}${Math.round(Number(value))}`;
  const moves = (values) => Object.entries(values || {})
    .filter(([key, value]) => names[key] && Math.abs(Number(value) || 0) > 1e-3)
    .map(([key, value]) => fmt(key, value));
  const lines = [];
  const global = [...moves(develop.light), ...moves(develop.color), ...moves(develop.effects)];
  if (global.length) lines.push(`全局：${global.join(" / ")}`);
  if (develop.wb && (develop.wb.temp || develop.wb.tint)) {
    lines.push(`白平衡：色温 ${Math.round(develop.wb.temp || 0)}K / 色调 ${Math.round(develop.wb.tint || 0)}`);
  }
  for (const [key, label] of [["shadows", "阴影"], ["midtones", "中间调"], ["highlights", "高光"]]) {
    const wheel = develop.grading?.[key];
    if (wheel && Number(wheel.sat) > 0) lines.push(`${label}色轮：${Math.round(wheel.hue || 0)}° / 饱和度 ${Math.round(wheel.sat)}`);
  }
  for (const mask of recipe.plan?.masks || []) {
    const kind = mask.shape?.kind === "linear" ? "线性渐变" : "径向";
    const local = moves(mask.adjust);
    lines.push(`${mask.name}（${kind}）：${local.join(" / ") || "局部调整"}`);
  }
  const directed = recipe.directed || {};
  if (directed.rounds > 1) lines.push(`复核修订：第 ${directed.selected_round + 1} 版 / 共 ${directed.rounds} 版`);
  return lines.join("\n");
}

function summarizeRecipe(recipe) {
  if (!recipe) return "";
  if (recipe.engine === "lightcraft-directed") return summarizeDirected(recipe);
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

// ---- Vision-model vault -------------------------------------------------
// The user picks a provider and pastes a key. The server fills in the address
// and protocol and picks a model that can read images; the key goes straight
// into the encrypted vault and is never shown again.

let VISION_PRESETS = [];

async function loadVisionPresets() {
  if (!VISION_PRESETS.length) VISION_PRESETS = await api("/api/vision/presets");
  return VISION_PRESETS;
}

async function visionRequest(url, body) {
  const res = await fetch(url, {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {}), credentials: "same-origin"});
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const error = new Error(typeof data.detail === "string" ? data.detail : "操作失败");
    error.status = res.status;
    throw error;
  }
  return data;
}

async function openVisionDialog() {
  const dialog = $("vision-dialog");
  if (!dialog.open) dialog.showModal();
  try {
    await loadVisionPresets();
    renderVision(await api("/api/vision/vault"));
  } catch (err) {
    $("vision-error").textContent = err.message;
  }
}

async function visionFail(err) {
  if (err.status === 403) {
    try { renderVision(await api("/api/vision/vault")); } catch {}
  }
  $("vision-error").textContent = err.message || String(err);
}

function renderVision(v) {
  $("vision-error").textContent = "";
  const body = $("vision-body");
  body.innerHTML = "";
  if (!v.exists) renderVaultCreate(body, v);
  else if (!v.manage) renderVaultUnlock(body, v);
  else renderVaultManage(body, v);
}

function visionForm(html) {
  const form = document.createElement("form");
  form.className = "vision-form";
  form.autocomplete = "off";
  form.innerHTML = html;
  return form;
}

function renderVaultCreate(body, v) {
  const form = visionForm(`
    <h3>先设置保险箱口令</h3>
    <p class="vision-hint">口令至少 ${v.min_passphrase} 个字符。它只用来加密，不会保存在任何地方；忘记后只能清空重来。</p>
    <label>口令<input type="password" name="p1" autocomplete="new-password" minlength="${v.min_passphrase}" required></label>
    <label>再输入一次<input type="password" name="p2" autocomplete="new-password" required></label>
    <div class="vision-row"><button class="btn primary" type="submit">创建并解锁</button></div>`);
  form.onsubmit = async (event) => {
    event.preventDefault();
    const p1 = form.elements.p1.value, p2 = form.elements.p2.value;
    if (p1 !== p2) { $("vision-error").textContent = "两次输入的口令不一致"; return; }
    form.querySelector("button").disabled = true;
    try {
      const data = await visionRequest("/api/vision/vault/create", {passphrase: p1});
      form.reset();
      renderVision(data);
    } catch (err) { visionFail(err); } finally { form.querySelector("button").disabled = false; }
  };
  body.appendChild(form);
}

function renderVaultUnlock(body, v) {
  const status = v.unlocked
    ? `保险箱已解锁，正在使用 ${v.count || 0} 个模型。输入口令可以查看和修改配置。`
    : "保险箱已锁定。客户端每次启动后都要输入口令，模型才能使用。";
  const form = visionForm(`
    <p class="vision-status">${status}</p>
    <label>口令<input type="password" name="passphrase" autocomplete="current-password" required></label>
    <div class="vision-row"><button class="btn primary" type="submit">解锁</button>
      ${v.unlocked ? '<button class="btn ghost" type="button" data-act="lock">立即锁定</button>' : ""}</div>
    <details class="vision-reset"><summary>忘记口令</summary>
      <p class="vision-hint">只能删除全部模型配置后重新设置。输入「重置」确认。</p>
      <div class="vision-row"><input name="confirm" placeholder="重置"><button class="btn ghost" type="button" data-act="reset">删除全部配置</button></div>
    </details>`);
  form.onsubmit = async (event) => {
    event.preventDefault();
    const button = form.querySelector("button[type=submit]");
    button.disabled = true;
    try {
      const data = await visionRequest("/api/vision/vault/unlock", {passphrase: form.elements.passphrase.value});
      form.reset();
      renderVision(data);
    } catch (err) { form.elements.passphrase.value = ""; visionFail(err); } finally { button.disabled = false; }
  };
  form.querySelector("[data-act=reset]").onclick = async () => {
    try { renderVision(await visionRequest("/api/vision/vault/reset", {confirm: form.elements.confirm.value.trim()})); }
    catch (err) { visionFail(err); }
  };
  const lockButton = form.querySelector("[data-act=lock]");
  if (lockButton) lockButton.onclick = lockVault;
  body.appendChild(form);
}

async function lockVault() {
  try { renderVision(await visionRequest("/api/vision/vault/lock")); } catch (err) { visionFail(err); }
}

function hostOf(url) {
  try { return new URL(url).host; } catch { return url; }
}

function renderVaultManage(body, v) {
  const profiles = v.profiles || [];
  const head = document.createElement("div");
  head.className = "vision-row vision-toolbar";
  const bridge = v.bridge ? " · 本机评测通道已开启（本机程序可借用这里的模型，只拿到脱敏回答，看不到 Key 和模型名）" : "";
  head.innerHTML = `<p class="vision-status">已解锁 · 启用 ${v.count || 0} 个 · 按列表顺序优先使用，前一个失败时自动换下一个${bridge}</p>
    <button class="btn ghost" type="button" data-act="passphrase">修改口令</button>
    <button class="btn ghost" type="button" data-act="lock">锁定</button>`;
  head.querySelector("[data-act=lock]").onclick = lockVault;
  head.querySelector("[data-act=passphrase]").onclick = () => renderPassphraseForm(body);
  body.appendChild(head);

  const list = document.createElement("ol");
  list.className = "vision-list";
  if (!profiles.length) list.innerHTML = `<li class="vision-empty">还没有模型。点「添加模型」开始。</li>`;
  profiles.forEach((p, index) => {
    const item = document.createElement("li");
    item.className = "vision-item" + (p.enabled ? "" : " disabled");
    const preset = VISION_PRESETS.find(x => x.id === p.preset);
    // The address is worth showing only where the user typed it.
    const where = !preset || preset.custom_url || !preset.needs_key || preset.base_url !== p.base_url
      ? ` · ${escapeHtml(hostOf(p.base_url))}` : "";
    item.innerHTML = `
      <div class="vision-meta"><strong>${escapeHtml(p.slot)}</strong>${p.label ? `<span class="vision-label">${escapeHtml(p.label)}</span>` : ""}
        ${p.enabled ? "" : '<span class="vision-off">已停用</span>'}
        <span class="vision-detail">${escapeHtml(p.provider_name || "")} · ${escapeHtml(p.model)}${where} · ${p.has_key ? "Key 已保存" : "无 Key"}</span></div>
      <div class="vision-actions">
        <button type="button" data-act="up" ${index === 0 ? "disabled" : ""} aria-label="上移">↑</button>
        <button type="button" data-act="down" ${index === profiles.length - 1 ? "disabled" : ""} aria-label="下移">↓</button>
        <button type="button" data-act="test">测试</button>
        <button type="button" data-act="edit">编辑</button>
        <button type="button" data-act="toggle">${p.enabled ? "停用" : "启用"}</button>
        <button type="button" data-act="delete">删除</button>
      </div>
      <p class="vision-test"></p>`;
    const ids = profiles.map(x => x.id);
    const move = async (delta) => {
      const next = ids.slice();
      [next[index], next[index + delta]] = [next[index + delta], next[index]];
      try { renderVision(await visionRequest("/api/vision/profiles/order", {ids: next})); } catch (err) { visionFail(err); }
    };
    item.querySelector("[data-act=up]").onclick = () => move(-1);
    item.querySelector("[data-act=down]").onclick = () => move(1);
    item.querySelector("[data-act=edit]").onclick = () => renderProfileForm(body, p);
    item.querySelector("[data-act=toggle]").onclick = async () => {
      try {
        renderVision(await visionRequest("/api/vision/profiles", {id: p.id, preset: p.preset, label: p.label,
          base_url: p.base_url, model: p.model, protocol: p.protocol, enabled: !p.enabled}));
      } catch (err) { visionFail(err); }
    };
    item.querySelector("[data-act=delete]").onclick = async () => {
      if (!confirm(`删除${p.slot}？保存的 API Key 会一起删除。`)) return;
      try { renderVision(await visionRequest(`/api/vision/profiles/${encodeURIComponent(p.id)}/delete`)); }
      catch (err) { visionFail(err); }
    };
    item.querySelector("[data-act=test]").onclick = async (event) => {
      const out = item.querySelector(".vision-test");
      event.target.disabled = true;
      out.textContent = "正在发送一张 64×64 的纯色测试图…";
      try {
        out.textContent = describeVisionTest(await visionRequest(`/api/vision/profiles/${encodeURIComponent(p.id)}/test`));
      } catch (err) { out.textContent = ""; visionFail(err); } finally { event.target.disabled = false; }
    };
    list.appendChild(item);
  });
  body.appendChild(list);

  const add = document.createElement("div");
  add.className = "vision-row";
  add.innerHTML = `<button class="btn primary" type="button" ${profiles.length >= 12 ? "disabled" : ""}>添加模型</button>`;
  add.querySelector("button").onclick = () => renderProfileForm(body, null);
  body.appendChild(add);
}

function describeVisionTest(result) {
  if (!result.ok) {
    const hints = {401: "API Key 无效", 403: "没有权限使用这个模型", 404: "模型名或服务地址不对", 429: "额度不足或请求过于频繁"};
    const named = {APIConnectionError: "连不上服务地址", APITimeoutError: "请求超时", ConnectError: "连不上服务地址",
      ReadTimeout: "请求超时", AuthenticationError: "API Key 无效", NotFoundError: "模型名或服务地址不对"};
    const reason = hints[result.status] || named[result.error] || result.error;
    return `失败：${reason}${result.status ? `（HTTP ${result.status}）` : ""} · ${result.latency_s}s`;
  }
  const seen = result.sees_image === true ? "能看到图片" : result.sees_image === false
    ? "有回答，但没认出图片颜色，可能不支持图片输入" : "有回答，但不是 JSON 格式";
  return `连接成功 · ${result.latency_s}s · ${seen}`;
}

function renderProfileForm(body, profile) {
  body.querySelector(".vision-profile-form")?.remove();
  body.querySelector(".vision-passphrase-form")?.remove();
  const editing = Boolean(profile);
  const current = profile || {preset: VISION_PRESETS[0].id, label: "", base_url: "", model: "", protocol: "", enabled: true};
  const options = VISION_PRESETS.map(p => `<option value="${p.id}">${escapeHtml(p.name)}</option>`).join("");
  const form = visionForm(`
    <h3>${editing ? `编辑${escapeHtml(profile.slot)}` : "添加模型"}</h3>
    <label>服务商<select name="preset">${options}</select></label>
    <label data-for="key">API Key<input type="password" name="api_key" autocomplete="off" spellcheck="false"></label>
    <label data-for="url">服务地址<input name="base_url" spellcheck="false"></label>
    <details class="vision-advanced"><summary>高级设置（一般不用填）</summary>
      <label>模型名<input name="model" spellcheck="false" placeholder="留空：自动挑选能看图的模型"></label>
      <label data-for="cloud-url">服务地址<input name="base_url_override" spellcheck="false" placeholder="留空：使用服务商的官方地址"></label>
      <label data-for="openai">接口协议<select name="protocol"><option value="chat/completions">chat/completions</option><option value="responses">responses</option></select></label>
      <label>备注名（只在这个窗口显示）<input name="label" maxlength="40"></label>
    </details>
    <label class="vision-check"><input type="checkbox" name="enabled"> 启用</label>
    <p class="vision-hint" data-role="hint"></p>
    <div class="vision-row"><button class="btn primary" type="submit">保存</button><button class="btn ghost" type="button" data-act="cancel">取消</button>
      <span class="vision-progress" aria-live="polite"></span></div>`);
  form.classList.add("vision-profile-form");
  const el = form.elements;
  const presetOf = () => VISION_PRESETS.find(p => p.id === el.preset.value) || VISION_PRESETS[VISION_PRESETS.length - 1];
  el.preset.value = current.preset || "custom";
  el.label.value = current.label || "";
  el.model.value = current.model || "";
  el.enabled.checked = current.enabled !== false;
  // Typed address: required for custom services, editable for local ones, an override for cloud presets.
  const fillAddress = (preset, url) => {
    const typed = preset.custom_url || !preset.needs_key;
    el.base_url.value = typed ? (url || preset.base_url) : "";
    el.base_url_override.value = !typed && url && url !== preset.base_url ? url : "";
    el.protocol.value = (editing && current.protocol && preset.id === current.preset ? current.protocol : preset.protocol) || "chat/completions";
  };
  const sync = () => {
    const preset = presetOf();
    const typed = preset.custom_url || !preset.needs_key;
    form.querySelector("[data-for=key]").classList.toggle("hidden", !preset.needs_key);
    form.querySelector("[data-for=url]").classList.toggle("hidden", !typed);
    form.querySelector("[data-for=cloud-url]").classList.toggle("hidden", typed);
    form.querySelector("[data-for=openai]").classList.toggle("hidden", preset.kind !== "openai");
    el.base_url.required = Boolean(preset.custom_url);
    el.api_key.placeholder = editing && profile.has_key && preset.id === current.preset
      ? "已保存。留空表示不修改" : preset.needs_key ? "粘贴这个服务商的 API Key" : "本机服务不需要";
    form.querySelector("[data-role=hint]").textContent = preset.needs_key
      ? "保存时会用这个 Key 查询模型列表，并向排在前面的几个模型发送一张 64×64 的纯色测试图，选出第一个能看懂图片的模型。换服务商后需要重新填写 Key，已保存的 Key 不会发往新地址。"
      : "保存时会查询这台服务上已经装好的模型，挑出能看懂图片的那个。";
  };
  fillAddress(presetOf(), current.base_url);
  sync();
  el.preset.onchange = () => {
    fillAddress(presetOf(), "");
    el.model.value = "";  // A model id from another provider would not exist here.
    sync();
  };
  form.querySelector("[data-act=cancel]").onclick = () => form.remove();
  form.onsubmit = async (event) => {
    event.preventDefault();
    const preset = presetOf();
    const typed = preset.custom_url || !preset.needs_key;
    const payload = {preset: preset.id, label: el.label.value.trim(), model: el.model.value.trim(),
      base_url: (typed ? el.base_url.value : el.base_url_override.value).trim(),
      protocol: preset.kind === "openai" ? el.protocol.value : null, enabled: el.enabled.checked};
    if (editing) payload.id = profile.id;
    if (preset.needs_key && el.api_key.value.trim()) payload.api_key = el.api_key.value.trim();
    const button = form.querySelector("button[type=submit]");
    const progress = form.querySelector(".vision-progress");
    button.disabled = true;
    progress.textContent = payload.model ? "正在保存…" : "正在挑选能看图的模型，最多需要一两分钟…";
    try {
      const data = await visionRequest("/api/vision/profiles", payload);
      el.api_key.value = "";
      renderVision(data);
      if (data.discovery) {
        $("vision-error").textContent = "";
        const note = document.createElement("p");
        note.className = "vision-test vision-picked";
        note.textContent = `已自动选择 ${data.discovery.model}，测试图识别正确 · ${data.discovery.latency_s}s`;
        $("vision-body").querySelector(".vision-list")?.after(note);
      }
    } catch (err) { visionFail(err); } finally { button.disabled = false; progress.textContent = ""; }
  };
  body.appendChild(form);
  (presetOf().needs_key ? el.api_key : el.preset).focus();
}

function renderPassphraseForm(body) {
  body.querySelector(".vision-profile-form")?.remove();
  body.querySelector(".vision-passphrase-form")?.remove();
  const form = visionForm(`
    <h3>修改口令</h3>
    <label>当前口令<input type="password" name="old" autocomplete="current-password" required></label>
    <label>新口令<input type="password" name="p1" autocomplete="new-password" required></label>
    <label>再输入一次<input type="password" name="p2" autocomplete="new-password" required></label>
    <div class="vision-row"><button class="btn primary" type="submit">修改</button><button class="btn ghost" type="button" data-act="cancel">取消</button></div>`);
  form.classList.add("vision-passphrase-form");
  form.querySelector("[data-act=cancel]").onclick = () => form.remove();
  form.onsubmit = async (event) => {
    event.preventDefault();
    if (form.elements.p1.value !== form.elements.p2.value) { $("vision-error").textContent = "两次输入的新口令不一致"; return; }
    try {
      const data = await visionRequest("/api/vision/vault/passphrase", {old: form.elements.old.value, new: form.elements.p1.value});
      form.reset();
      renderVision(data);
      $("vision-error").textContent = "口令已修改";
    } catch (err) { visionFail(err); }
  };
  body.appendChild(form);
}
