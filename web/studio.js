/* EXIF Watermark & Auto-Collage / WeChat Moments 9-Grid Studio */
(() => {
  const studio = {
    metaLoaded: false,
    styles: [],
    layouts: [],
    brands: {},
    mode: "wm",              // "wm" | "collage"
    wmAssets: [],            // uploaded / imported assets for watermark tab
    wmActiveId: null,
    wmStyle: "leica_bar",
    wmPreviewUrl: null,
    wmSeq: 0,
    wmTimer: null,
    wmCheckerBg: false,

    colAssets: [],           // ordered list of assets in collage / Moments pool
    colLayout: "grid_9",
    colBg: "white",
    colView: "moments",      // "moments" | "canvas"
    colPreviewUrl: null,
    colSeq: 0,
    colTimer: null,
    momentsTiles: [],        // rendered tile data URLs for lightbox & grid
    momentsIsSlice: false,
    momentsSuggestedCaption: "",
    lightboxIdx: -1,
    dragIndex: null,
  };

  function el(id) {
    return document.getElementById(id);
  }

  function showError(msg) {
    const node = el("st-error");
    if (node) node.textContent = msg ? String(msg) : "";
  }

  function esc(s) {
    return String(s ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function apiJson(url, opts = {}) {
    const res = await fetch(url, opts);
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(typeof data.detail === "string" ? data.detail : `请求失败 (${res.status})`);
    }
    return data;
  }

  async function ensureMeta() {
    if (studio.metaLoaded) return;
    const data = await apiJson("/api/studio/meta");
    studio.styles = data.watermark_styles || [];
    studio.layouts = data.collage_layouts || [];
    studio.brands = data.brands || {};
    studio.metaLoaded = true;
    renderStyleCards();
    renderBrandOptions();
    renderLayoutCards();
  }

  function styleIconSvg(id) {
    const icons = {
      leica_bar: `<svg viewBox="0 0 44 32" fill="none"><rect x="2" y="2" width="40" height="28" rx="3" fill="#fff" stroke="#555"/><rect x="3" y="3" width="38" height="19" rx="1.5" fill="#3a3d42"/><rect x="5" y="24.5" width="12" height="2" rx="1" fill="#222"/><circle cx="25" cy="25.5" r="2" fill="#e20612"/><rect x="29" y="24.5" width="10" height="2" rx="1" fill="#222"/></svg>`,
      dark_bar: `<svg viewBox="0 0 44 32" fill="none"><rect x="2" y="2" width="40" height="28" rx="3" fill="#141518" stroke="#555"/><rect x="3" y="3" width="38" height="19" rx="1.5" fill="#3a3d42"/><rect x="5" y="24.5" width="12" height="2" rx="1" fill="#eee"/><circle cx="25" cy="25.5" r="2" fill="#f2c500"/><rect x="29" y="24.5" width="10" height="2" rx="1" fill="#eee"/></svg>`,
      gallery_frame: `<svg viewBox="0 0 44 32" fill="none"><rect x="2" y="2" width="40" height="28" rx="3" fill="#fff" stroke="#555"/><rect x="6" y="5" width="32" height="16" rx="1" fill="#3a3d42"/><rect x="14" y="23.5" width="16" height="1.8" rx=".9" fill="#222"/><rect x="17" y="26.5" width="10" height="1.4" rx=".7" fill="#999"/></svg>`,
      blur_card: `<svg viewBox="0 0 44 32" fill="none"><rect x="2" y="2" width="40" height="28" rx="3" fill="#2c333b" stroke="#555"/><rect x="7" y="5" width="30" height="16" rx="2.5" fill="#637080" stroke="#ffffff40"/><rect x="13" y="24" width="18" height="1.8" rx=".9" fill="#eee"/><rect x="17" y="27" width="10" height="1.3" rx=".6" fill="#aaa"/></svg>`,
      film_polaroid: `<svg viewBox="0 0 44 32" fill="none"><rect x="2" y="2" width="40" height="28" rx="3" fill="#f6f3ec" stroke="#555"/><rect x="6" y="5" width="32" height="15" rx="1" fill="#3a3d42"/><rect x="6" y="22.5" width="14" height="1.8" rx=".9" fill="#333"/><rect x="31" y="22.5" width="7" height="1.8" rx=".9" fill="#d07a32"/><rect x="6" y="26" width="22" height="1.4" rx=".7" fill="#888"/></svg>`,
      corner_overlay: `<svg viewBox="0 0 44 32" fill="none"><rect x="2" y="2" width="40" height="28" rx="3" fill="#3a3d42" stroke="#555"/><rect x="2" y="21" width="40" height="9" rx="2" fill="#00000080"/><rect x="5" y="24.5" width="12" height="1.8" rx=".9" fill="#fff"/><rect x="26" y="24.5" width="13" height="1.8" rx=".9" fill="#fff"/></svg>`,
    };
    return icons[id] || icons.leica_bar;
  }

  function renderStyleCards() {
    const grid = el("wm-style-grid");
    if (!grid) return;
    grid.innerHTML = studio.styles.map((s) => `
      <button type="button" class="st-style-card ${s.id === studio.wmStyle ? "active" : ""}" data-wm-style="${esc(s.id)}">
        <span class="st-style-icon" aria-hidden="true">${styleIconSvg(s.id)}</span>
        <span class="st-style-name">${esc(s.name)}</span>
        <span class="st-style-sub">${esc(s.subtitle)}</span>
      </button>`).join("");
    grid.querySelectorAll("[data-wm-style]").forEach((btn) => {
      btn.addEventListener("click", () => {
        studio.wmStyle = btn.dataset.wmStyle;
        grid.querySelectorAll("[data-wm-style]").forEach((b) => b.classList.toggle("active", b === btn));
        scheduleWmPreview(true);
      });
    });
  }

  function renderBrandOptions() {
    const sel = el("wm-brand");
    if (!sel) return;
    const entries = Object.entries(studio.brands);
    sel.innerHTML = entries.map(([key, spec]) =>
      `<option value="${esc(key)}">${esc(spec.label)}</option>`
    ).join("");
  }

  function renderLayoutCards() {
    const grid = el("col-layout-grid");
    if (!grid) return;
    grid.innerHTML = studio.layouts.map((l) => `
      <button type="button" class="st-layout-card ${l.id === studio.colLayout ? "active" : ""}" data-col-layout="${esc(l.id)}">
        <span class="st-layout-head">
          <strong>${esc(l.name)}</strong>
          ${l.slice ? '<span class="st-tag">切图</span>' : ""}
        </span>
        <span class="st-layout-sub">${esc(l.subtitle)}</span>
      </button>`).join("");
    grid.querySelectorAll("[data-col-layout]").forEach((btn) => {
      btn.addEventListener("click", () => {
        studio.colLayout = btn.dataset.colLayout;
        grid.querySelectorAll("[data-col-layout]").forEach((b) => b.classList.toggle("active", b === btn));
        syncSliceControls();
        scheduleCollagePreview(true);
      });
    });
    syncSliceControls();
  }

  function isSliceLayout(id = studio.colLayout) {
    return id === "slice_9" || id === "slice_4" || id === "slice_6";
  }

  function syncSliceControls() {
    const slice = isSliceLayout();
    el("col-slice-group")?.classList.toggle("hidden", !slice);
    const tileModeWrap = el("col-tile-mode")?.closest("label");
    if (tileModeWrap) tileModeWrap.classList.toggle("hidden", slice);
  }

  // ---- Dialog open / mode switching ----------------------------------------

  async function openStudio(mode = "wm", opts = {}) {
    showError("");
    const dlg = el("studio-dialog");
    if (!dlg) return;
    if (!dlg.open) dlg.showModal();
    try {
      await ensureMeta();
    } catch (err) {
      showError(err.message);
      return;
    }

    const appState = window.SuperPhotoState;
    const hasJob = Boolean(appState?.jobId && appState?.job?.status === "ready");
    el("wm-load-current")?.classList.toggle("hidden", !hasJob);
    el("col-load-current")?.classList.toggle("hidden", !hasJob);

    setStudioMode(mode);

    if (opts.batchId) {
      await importBatchIntoCollage(opts.batchId);
      return;
    }
    if (opts.importCurrent && hasJob) {
      if (mode === "wm") await importCurrentJobToWm();
      else await importCurrentJobToCollage();
      return;
    }
    if (mode === "wm" && !studio.wmActiveId && hasJob) {
      await importCurrentJobToWm();
    } else if (mode === "collage" && studio.colAssets.length === 0 && hasJob) {
      await importCurrentJobToCollage();
    }
  }

  function setStudioMode(mode) {
    studio.mode = mode === "collage" ? "collage" : "wm";
    const isWm = studio.mode === "wm";
    el("st-tab-wm").classList.toggle("active", isWm);
    el("st-tab-wm").setAttribute("aria-selected", String(isWm));
    el("st-tab-collage").classList.toggle("active", !isWm);
    el("st-tab-collage").setAttribute("aria-selected", String(!isWm));
    el("st-pane-wm").classList.toggle("hidden", !isWm);
    el("st-pane-collage").classList.toggle("hidden", isWm);
    if (!isWm && studio.colAssets.length) scheduleCollagePreview(true);
    if (isWm && studio.wmActiveId) scheduleWmPreview(true);
  }

  // ---- Asset import & upload helpers ---------------------------------------

  function currentJobRef() {
    const appState = window.SuperPhotoState;
    if (!appState?.jobId || appState?.job?.status !== "ready") return null;
    const slot = appState.selected || "A";
    const version = (appState.job.refinements || []).find(
      (r) => r.id === appState.versions?.[slot] && r.status === "ready"
    )?.id || null;
    return { job_id: appState.jobId, slot, version };
  }

  async function importJobAsset(ref) {
    const data = await apiJson("/api/studio/import-job", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(ref),
    });
    return data.asset;
  }

  async function uploadFilesToStudio(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length) return [];
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const data = await apiJson("/api/studio/upload", { method: "POST", body: form });
    return data.assets || [];
  }

  function findSameRef(list, ref) {
    if (!ref) return null;
    return list.find((a) =>
      a.source_ref &&
      a.source_ref.job_id === ref.job_id &&
      a.source_ref.slot === ref.slot &&
      (a.source_ref.version || null) === (ref.version || null)
    );
  }

  async function importCurrentJobToWm() {
    const ref = currentJobRef();
    if (!ref) return;
    showError("");
    const existing = findSameRef(studio.wmAssets, ref);
    if (existing) {
      selectWmAsset(existing.id);
      return;
    }
    el("wm-loading").classList.remove("hidden");
    try {
      const asset = await importJobAsset(ref);
      studio.wmAssets.unshift(asset);
      renderWmStrip();
      selectWmAsset(asset.id);
    } catch (err) {
      showError(err.message);
    } finally {
      el("wm-loading").classList.add("hidden");
    }
  }

  async function importCurrentJobToCollage() {
    const ref = currentJobRef();
    if (!ref) return;
    showError("");
    if (findSameRef(studio.colAssets, ref)) {
      renderCollageTray();
      scheduleCollagePreview(true);
      return;
    }
    try {
      const asset = await importJobAsset(ref);
      studio.colAssets.push(asset);
      renderCollageTray();
      scheduleCollagePreview(true);
    } catch (err) {
      showError(err.message);
    }
  }

  async function importBatchIntoCollage(batchId) {
    showError("");
    el("moments-loading")?.classList.remove("hidden");
    try {
      const data = await apiJson("/api/studio/import-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ batch_id: batchId, include_anchor: true, limit: 9 }),
      });
      const incoming = data.assets || [];
      if (incoming.length) {
        studio.colAssets = incoming;
        if (incoming.length === 4) studio.colLayout = "grid_4";
        else if (incoming.length === 6) studio.colLayout = "grid_6";
        else studio.colLayout = "grid_9";
        renderLayoutCards();
        renderCollageTray();
        setCollageView("moments");
        el("col-arrange-note").textContent = `已一键导入「${data.batch_title || "同风格组图"}」共 ${incoming.length} 张（已自动提取各自 EXIF）`;
        await refreshCollagePreviewNow();
        if (!el("moments-caption").value.trim() && studio.momentsSuggestedCaption) {
          el("moments-caption").value = studio.momentsSuggestedCaption;
        }
      }
    } catch (err) {
      showError(err.message);
    } finally {
      el("moments-loading")?.classList.add("hidden");
    }
  }

  // ---- Library drawer (pick from SuperPhoto jobs & style batches) ----------

  async function openLibraryDrawer(target) {
    const drawer = el(target === "wm" ? "wm-library-drawer" : "col-library-drawer");
    const list = el(target === "wm" ? "wm-library-list" : "col-library-list");
    if (!drawer || !list) return;
    drawer.classList.remove("hidden");
    list.innerHTML = `<span class="st-hint-inline">正在读取本地照片库…</span>`;
    try {
      const data = await apiJson("/api/studio/sources?limit=24");
      const batches = data.batches || [];
      const jobs = data.jobs || [];
      if (!batches.length && !jobs.length) {
        list.innerHTML = `<span class="st-hint-inline">还没有已保存的修图作品，可以直接上传本地照片。</span>`;
        return;
      }
      const batchHtml = target === "collage" && batches.length ? `
        <div class="st-lib-section">
          <div class="st-lib-title">同风格批量组图（一键导入多张）</div>
          <div class="st-lib-pills">
            ${batches.map((b) => `
              <button type="button" class="st-lib-batch" data-import-batch="${esc(b.batch_id)}">
                <strong>${esc(b.title)}</strong>
                <span>${b.done_count + 1} 张（含基准图）</span>
              </button>`).join("")}
          </div>
        </div>` : "";
      const jobHtml = jobs.length ? `
        <div class="st-lib-section">
          <div class="st-lib-title">最近已完成的修图作品</div>
          <div class="st-lib-grid">
            ${jobs.map((j) => {
              const m = j.metadata || {};
              const param = [m.focal_length, m.aperture, m.shutter, m.iso].filter(Boolean).join(" · ");
              return `
                <button type="button" class="st-lib-item" data-import-job="${esc(j.job_id)}" data-slot="${esc(j.slot)}" data-version="${esc(j.version || "")}">
                  <img src="${esc(j.preview_url)}" alt="${esc(j.original_name)}" loading="lazy" />
                  <span class="st-lib-name" title="${esc(j.original_name)}">${esc(j.original_name)}</span>
                  <span class="st-lib-meta">${esc(param || m.camera || j.title || "无 EXIF")}</span>
                </button>`;
            }).join("")}
          </div>
        </div>` : "";
      list.innerHTML = batchHtml + jobHtml;

      list.querySelectorAll("[data-import-batch]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          drawer.classList.add("hidden");
          await importBatchIntoCollage(btn.dataset.importBatch);
        });
      });
      list.querySelectorAll("[data-import-job]").forEach((btn) => {
        btn.addEventListener("click", async () => {
          const ref = {
            job_id: btn.dataset.importJob,
            slot: btn.dataset.slot || "A",
            version: btn.dataset.version || null,
          };
          btn.disabled = true;
          try {
            const asset = await importJobAsset(ref);
            if (target === "wm") {
              studio.wmAssets.unshift(asset);
              renderWmStrip();
              selectWmAsset(asset.id);
              drawer.classList.add("hidden");
            } else {
              studio.colAssets.push(asset);
              renderCollageTray();
              scheduleCollagePreview(true);
            }
          } catch (err) {
            showError(err.message);
          } finally {
            btn.disabled = false;
          }
        });
      });
    } catch (err) {
      list.innerHTML = `<span class="vision-error">${esc(err.message)}</span>`;
    }
  }

  // ---- Watermark Tab Logic -------------------------------------------------

  function activeWmAsset() {
    return studio.wmAssets.find((a) => a.id === studio.wmActiveId) || null;
  }

  function populateWmFieldsFromAsset(asset) {
    const m = asset?.metadata || {};
    el("wm-camera").value = m.camera || "";
    el("wm-brand").value = m.brand && studio.brands[m.brand] ? m.brand : "generic";
    el("wm-lens").value = m.lens || "";
    el("wm-focal").value = m.focal_length || "";
    el("wm-aperture").value = m.aperture || "";
    el("wm-shutter").value = m.shutter || "";
    el("wm-iso").value = m.iso || "";
    el("wm-date").value = m.datetime || "";
    el("wm-location").value = m.gps || "";

    const badge = el("wm-exif-badge");
    if (!asset) {
      badge.textContent = "等待选择照片";
      badge.className = "st-pill";
    } else if (m.has_exif) {
      const summary = [m.camera, m.focal_length, m.aperture, m.shutter, m.iso].filter(Boolean).join(" · ");
      badge.textContent = `已自动提取 EXIF：${summary}`;
      badge.className = "st-pill ok";
    } else {
      badge.textContent = "原片未含 EXIF，可手动填写参数";
      badge.className = "st-pill warn";
    }
  }

  function readWmConfig(forBatch = false) {
    const base = {
      style: studio.wmStyle,
      show_brand_mark: el("wm-show-brand").checked,
      show_lens: el("wm-show-lens").checked,
      show_date: el("wm-show-date").checked,
      show_location: el("wm-show-loc").checked,
      show_ev: el("wm-show-ev").checked,
      border_scale: Number(el("wm-scale").value) / 100,
      signature: el("wm-signature").value.trim() || null,
    };
    if (forBatch) {
      // In batch ZIP mode, each photo uses its own auto-extracted EXIF (only global signature/location override applies)
      const loc = el("wm-location").value.trim();
      if (loc) base.location = loc;
      return base;
    }
    return {
      ...base,
      camera: el("wm-camera").value.trim(),
      brand: el("wm-brand").value,
      lens: el("wm-lens").value.trim(),
      focal_length: el("wm-focal").value.trim(),
      aperture: el("wm-aperture").value.trim(),
      shutter: el("wm-shutter").value.trim(),
      iso: el("wm-iso").value.trim(),
      datetime_text: el("wm-date").value.trim(),
      location: el("wm-location").value.trim(),
    };
  }

  function selectWmAsset(assetId) {
    studio.wmActiveId = assetId;
    const asset = activeWmAsset();
    populateWmFieldsFromAsset(asset);
    renderWmStrip();
    const has = Boolean(asset);
    el("wm-export-btn").disabled = !has;
    el("wm-send-collage").disabled = !has;
    el("wm-empty").classList.toggle("hidden", has);
    el("wm-preview-img").classList.toggle("hidden", !has);
    if (has) scheduleWmPreview(true);
  }

  function renderWmStrip() {
    const strip = el("wm-strip");
    const batchBtn = el("wm-batch-zip-btn");
    if (!strip) return;
    const multi = studio.wmAssets.length > 1;
    strip.classList.toggle("hidden", !multi);
    batchBtn?.classList.toggle("hidden", !multi);
    if (batchBtn && multi) {
      batchBtn.textContent = `批量导出全部水印 (${studio.wmAssets.length} 张 .zip)`;
    }
    if (!multi) {
      strip.innerHTML = "";
      return;
    }
    strip.innerHTML = studio.wmAssets.map((a) => `
      <button type="button" class="st-wm-thumb ${a.id === studio.wmActiveId ? "active" : ""}" data-wm-aid="${esc(a.id)}" title="${esc(a.name)}">
        <img src="${esc(a.thumb_url)}" alt="${esc(a.name)}" />
        <span>${esc(a.metadata?.iso || a.metadata?.camera || a.name)}</span>
      </button>`).join("");
    strip.querySelectorAll("[data-wm-aid]").forEach((btn) => {
      btn.addEventListener("click", () => selectWmAsset(btn.dataset.wmAid));
    });
  }

  function scheduleWmPreview(immediate = false) {
    clearTimeout(studio.wmTimer);
    if (!studio.wmActiveId) return;
    studio.wmTimer = setTimeout(refreshWmPreviewNow, immediate ? 20 : 180);
  }

  async function refreshWmPreviewNow() {
    const aid = studio.wmActiveId;
    if (!aid) return;
    const seq = ++studio.wmSeq;
    el("wm-loading").classList.remove("hidden");
    showError("");
    try {
      const res = await fetch("/api/studio/watermark/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_id: aid,
          config: readWmConfig(false),
          max_edge: 1200,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "水印预览生成失败");
      }
      const blob = await res.blob();
      if (seq !== studio.wmSeq) return;
      if (studio.wmPreviewUrl) URL.revokeObjectURL(studio.wmPreviewUrl);
      studio.wmPreviewUrl = URL.createObjectURL(blob);
      el("wm-preview-img").src = studio.wmPreviewUrl;
      el("wm-preview-img").classList.remove("hidden");
      el("wm-empty").classList.add("hidden");
    } catch (err) {
      if (seq === studio.wmSeq) showError(err.message);
    } finally {
      if (seq === studio.wmSeq) el("wm-loading").classList.add("hidden");
    }
  }

  async function downloadBlobResponse(res, fallbackName) {
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "导出失败");
    }
    const cd = res.headers.get("Content-Disposition") || "";
    const mStar = cd.match(/filename\*=UTF-8''([^;]+)/i);
    const mPlain = cd.match(/filename="([^"]+)"/i);
    const filename = mStar ? decodeURIComponent(mStar[1]) : (mPlain ? mPlain[1] : fallbackName);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  }

  async function exportWmSingle() {
    const aid = studio.wmActiveId;
    if (!aid) return;
    const btn = el("wm-export-btn");
    const orig = btn.innerHTML;
    btn.disabled = true;
    btn.textContent = "正在生成高清带水印成片…";
    showError("");
    try {
      const res = await fetch("/api/studio/watermark/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_id: aid,
          config: readWmConfig(false),
          max_edge: 0,
        }),
      });
      await downloadBlobResponse(res, `SuperPhoto-水印-${studio.wmStyle}.jpg`);
    } catch (err) {
      showError(err.message);
    } finally {
      btn.innerHTML = orig;
      btn.disabled = !studio.wmActiveId;
    }
  }

  async function exportWmBatchZip() {
    if (studio.wmAssets.length < 2) return;
    const btn = el("wm-batch-zip-btn");
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "正在批量生成水印打包…";
    showError("");
    try {
      const res = await fetch("/api/studio/watermark/batch-zip", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_ids: studio.wmAssets.map((a) => a.id),
          config: readWmConfig(true),
          max_edge: 0,
        }),
      });
      await downloadBlobResponse(res, `SuperPhoto-批量水印-${studio.wmAssets.length}张.zip`);
    } catch (err) {
      showError(err.message);
    } finally {
      btn.textContent = orig;
      btn.disabled = false;
    }
  }

  // ---- Collage & WeChat Moments 9-Grid Logic -------------------------------

  function readCollageConfig() {
    return {
      layout: studio.colLayout,
      background: studio.colBg,
      gap: Number(el("col-gap").value),
      padding: Number(el("col-pad").value),
      radius: Number(el("col-rad").value),
      tile_frame_mode: el("col-tile-mode").value,
      slice_offset_x: Number(el("col-slice-x").value) / 100,
      slice_offset_y: Number(el("col-slice-y").value) / 100,
      show_footer: el("col-show-footer").checked,
      footer_title: el("col-footer-title").value.trim(),
      footer_subtitle: el("col-footer-sub").value.trim(),
      tile_watermark: el("col-tile-wm").checked,
      watermark_config: {
        style: el("col-wm-style").value,
        show_brand_mark: true,
        show_lens: true,
        show_date: true,
        show_location: false,
      },
    };
  }

  function setCollageView(view) {
    studio.colView = view === "canvas" ? "canvas" : "moments";
    const isMoments = studio.colView === "moments";
    el("col-view-moments").classList.toggle("active", isMoments);
    el("col-view-moments").setAttribute("aria-pressed", String(isMoments));
    el("col-view-canvas").classList.toggle("active", !isMoments);
    el("col-view-canvas").setAttribute("aria-pressed", String(!isMoments));
    syncCollageEmptyState();
    if (studio.colAssets.length) scheduleCollagePreview(true);
  }

  function syncCollageEmptyState() {
    const has = studio.colAssets.length > 0;
    el("col-empty").classList.toggle("hidden", has);
    el("moments-stage").classList.toggle("hidden", !has || studio.colView !== "moments");
    el("col-canvas").classList.toggle("hidden", !has || studio.colView !== "canvas");
    el("col-export-zip").disabled = !has;
    el("col-export-jpg").disabled = !has;
    el("col-count-badge").textContent = `${studio.colAssets.length} 张照片`;
  }

  function moveCollageItem(fromIdx, toIdx) {
    if (fromIdx === toIdx || fromIdx < 0 || toIdx < 0 || fromIdx >= studio.colAssets.length || toIdx >= studio.colAssets.length) {
      return;
    }
    const [item] = studio.colAssets.splice(fromIdx, 1);
    studio.colAssets.splice(toIdx, 0, item);
    renderCollageTray();
    scheduleCollagePreview(true);
  }

  function renderCollageTray() {
    const tray = el("col-tray");
    if (!tray) return;
    syncCollageEmptyState();
    if (!studio.colAssets.length) {
      tray.innerHTML = `<span class="st-tray-empty">暂无照片 · 可一次多选 9 张自动排布九宫格，或选 1 张大图体验「单图切九宫格」</span>`;
      return;
    }
    tray.innerHTML = studio.colAssets.map((a, idx) => {
      const m = a.metadata || {};
      const exifMini = [m.focal_length, m.aperture, m.shutter, m.iso].filter(Boolean).slice(0, 2).join(" ") || a.name;
      return `
        <div class="st-tray-item" draggable="true" data-idx="${idx}" title="${esc(a.name)}">
          <img src="${esc(a.thumb_url)}" alt="${esc(a.name)}" />
          <span class="st-tray-num">${String(idx + 1).padStart(2, "0")}</span>
          <button type="button" class="st-tray-del" data-del="${idx}" aria-label="移除">×</button>
          <div class="st-tray-foot">
            <button type="button" data-move-l="${idx}" ${idx === 0 ? "disabled" : ""} aria-label="前移">‹</button>
            <span class="st-tray-exif">${esc(exifMini)}</span>
            <button type="button" data-move-r="${idx}" ${idx === studio.colAssets.length - 1 ? "disabled" : ""} aria-label="后移">›</button>
          </div>
        </div>`;
    }).join("");

    tray.querySelectorAll("[data-del]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        studio.colAssets.splice(Number(btn.dataset.del), 1);
        renderCollageTray();
        scheduleCollagePreview(true);
      });
    });
    tray.querySelectorAll("[data-move-l]").forEach((btn) => {
      btn.addEventListener("click", () => moveCollageItem(Number(btn.dataset.moveL), Number(btn.dataset.moveL) - 1));
    });
    tray.querySelectorAll("[data-move-r]").forEach((btn) => {
      btn.addEventListener("click", () => moveCollageItem(Number(btn.dataset.moveR), Number(btn.dataset.moveR) + 1));
    });
    tray.querySelectorAll(".st-tray-item").forEach((card) => {
      const idx = Number(card.dataset.idx);
      card.addEventListener("dragstart", (e) => {
        studio.dragIndex = idx;
        card.classList.add("dragging");
        e.dataTransfer.effectAllowed = "move";
      });
      card.addEventListener("dragend", () => {
        studio.dragIndex = null;
        card.classList.remove("dragging");
      });
      card.addEventListener("dragover", (e) => {
        e.preventDefault();
        card.classList.add("drag-over");
      });
      card.addEventListener("dragleave", () => card.classList.remove("drag-over"));
      card.addEventListener("drop", (e) => {
        e.preventDefault();
        card.classList.remove("drag-over");
        if (studio.dragIndex !== null && studio.dragIndex !== idx) {
          moveCollageItem(studio.dragIndex, idx);
        }
      });
    });
  }

  async function runAutoArrange(mode) {
    if (studio.colAssets.length < 2) {
      el("col-arrange-note").textContent = "至少需要 2 张照片才能自动排布。";
      return;
    }
    showError("");
    try {
      const data = await apiJson("/api/studio/collage/auto-arrange", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_ids: studio.colAssets.map((a) => a.id),
          mode,
        }),
      });
      const byId = new Map(studio.colAssets.map((a) => [a.id, a]));
      studio.colAssets = (data.ordered_ids || []).map((id) => byId.get(id)).filter(Boolean);
      el("col-arrange-note").textContent = data.reason || "已完成智能排版";
      renderCollageTray();
      scheduleCollagePreview(true);
    } catch (err) {
      showError(err.message);
    }
  }

  function scheduleCollagePreview(immediate = false) {
    clearTimeout(studio.colTimer);
    if (!studio.colAssets.length) {
      syncCollageEmptyState();
      return;
    }
    studio.colTimer = setTimeout(refreshCollagePreviewNow, immediate ? 25 : 180);
  }

  async function refreshCollagePreviewNow() {
    if (!studio.colAssets.length) return;
    const seq = ++studio.colSeq;
    const payload = {
      asset_ids: studio.colAssets.map((a) => a.id),
      config: readCollageConfig(),
      target_width: 1500,
    };
    showError("");

    if (studio.colView === "moments") {
      el("moments-loading")?.classList.remove("hidden");
      try {
        const data = await apiJson("/api/studio/collage/moments-preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (seq !== studio.colSeq) return;
        studio.momentsTiles = data.tiles || [];
        studio.momentsIsSlice = Boolean(data.is_slice);
        studio.momentsSuggestedCaption = data.suggested_caption || "";
        if (!el("moments-loc").value.trim() && data.suggested_location) {
          el("moments-loc").value = data.suggested_location;
        }
        renderMomentsGrid(data.cols || 3);
      } catch (err) {
        if (seq === studio.colSeq) showError(err.message);
      } finally {
        if (seq === studio.colSeq) el("moments-loading")?.classList.add("hidden");
      }
    } else {
      el("col-loading")?.classList.remove("hidden");
      try {
        const res = await fetch("/api/studio/collage/preview", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "拼图预览生成失败");
        }
        const blob = await res.blob();
        if (seq !== studio.colSeq) return;
        if (studio.colPreviewUrl) URL.revokeObjectURL(studio.colPreviewUrl);
        studio.colPreviewUrl = URL.createObjectURL(blob);
        el("col-preview-img").src = studio.colPreviewUrl;
        el("col-preview-img").classList.remove("hidden");
      } catch (err) {
        if (seq === studio.colSeq) showError(err.message);
      } finally {
        if (seq === studio.colSeq) el("col-loading")?.classList.add("hidden");
      }
    }
  }

  function renderMomentsGrid(cols) {
    const grid = el("moments-grid");
    if (!grid) return;
    const showOrder = el("moments-show-order").checked;
    grid.className = `moments-grid cols-${cols}`;
    grid.innerHTML = studio.momentsTiles.map((src, idx) => `
      <div class="moments-tile" ${!studio.momentsIsSlice ? 'draggable="true"' : ""} data-midx="${idx}" role="button" tabindex="0" aria-label="预览第 ${idx + 1} 张">
        <img src="${src}" alt="朋友圈图片 ${idx + 1}" />
        ${showOrder ? `<span class="moments-order">${String(idx + 1).padStart(2, "0")}</span>` : ""}
      </div>`).join("");

    grid.querySelectorAll(".moments-tile").forEach((tile) => {
      const idx = Number(tile.dataset.midx);
      tile.addEventListener("click", () => openLightbox(idx));
      tile.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openLightbox(idx);
        }
      });
      if (!studio.momentsIsSlice) {
        tile.addEventListener("dragstart", (e) => {
          studio.dragIndex = idx;
          tile.classList.add("dragging");
          e.dataTransfer.effectAllowed = "move";
        });
        tile.addEventListener("dragend", () => {
          studio.dragIndex = null;
          tile.classList.remove("dragging");
        });
        tile.addEventListener("dragover", (e) => {
          e.preventDefault();
          tile.classList.add("drag-over");
        });
        tile.addEventListener("dragleave", () => tile.classList.remove("drag-over"));
        tile.addEventListener("drop", (e) => {
          e.preventDefault();
          tile.classList.remove("drag-over");
          if (studio.dragIndex !== null && studio.dragIndex !== idx) {
            moveCollageItem(studio.dragIndex, idx);
          }
        });
      }
    });
  }

  // ---- Moments Lightbox ----------------------------------------------------

  function openLightbox(idx) {
    if (!studio.momentsTiles.length) return;
    studio.lightboxIdx = Math.max(0, Math.min(studio.momentsTiles.length - 1, idx));
    const lb = el("st-lightbox");
    lb.classList.remove("hidden");
    updateLightbox();
  }

  function updateLightbox() {
    const idx = studio.lightboxIdx;
    if (idx < 0 || idx >= studio.momentsTiles.length) return;
    const asset = !studio.momentsIsSlice ? studio.colAssets[idx] : studio.colAssets[0];
    // Show full-resolution image if not watermarked/sliced, else the rendered Moments tile
    const cfg = readCollageConfig();
    const useRaw = !studio.momentsIsSlice && !cfg.tile_watermark && cfg.tile_frame_mode === "original" && asset?.url;
    el("st-lb-img").src = useRaw ? asset.url : studio.momentsTiles[idx];
    const m = asset?.metadata || {};
    const exif = [m.camera, m.lens, m.focal_length, m.aperture, m.shutter, m.iso].filter(Boolean).join(" · ");
    el("st-lb-cap").textContent = `${idx + 1} / ${studio.momentsTiles.length}${exif ? `  ·  ${exif}` : ""}`;
    el("st-lb-prev").disabled = idx === 0;
    el("st-lb-next").disabled = idx === studio.momentsTiles.length - 1;
  }

  function closeLightbox() {
    el("st-lightbox")?.classList.add("hidden");
    studio.lightboxIdx = -1;
  }

  async function exportCollage(kind) {
    if (!studio.colAssets.length) return;
    const btn = el(kind === "zip" ? "col-export-zip" : "col-export-jpg");
    const orig = btn.innerHTML;
    btn.disabled = true;
    btn.textContent = kind === "zip" ? "正在打包朋友圈高清切图…" : "正在渲染高清合并拼图…";
    showError("");
    try {
      const endpoint = kind === "zip" ? "/api/studio/collage/export-moments-zip" : "/api/studio/collage/export";
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          asset_ids: studio.colAssets.map((a) => a.id),
          config: readCollageConfig(),
          target_width: 2600,
          tile_edge: 1600,
        }),
      });
      await downloadBlobResponse(
        res,
        kind === "zip" ? `SuperPhoto-朋友圈九宫格-${studio.colLayout}.zip` : `SuperPhoto-拼图-${studio.colLayout}.jpg`
      );
    } catch (err) {
      showError(err.message);
    } finally {
      btn.innerHTML = orig;
      btn.disabled = !studio.colAssets.length;
    }
  }

  // ---- Wire up DOM events --------------------------------------------------

  function bindDropTarget(node, onFiles) {
    if (!node) return;
    ["dragenter", "dragover"].forEach((ev) => {
      node.addEventListener(ev, (e) => {
        if (!e.dataTransfer?.types?.includes("Files")) return;
        e.preventDefault();
        node.classList.add("over");
      });
    });
    ["dragleave", "drop"].forEach((ev) => {
      node.addEventListener(ev, (e) => {
        if (!e.dataTransfer?.types?.includes("Files")) return;
        e.preventDefault();
        node.classList.remove("over");
      });
    });
    node.addEventListener("drop", (e) => {
      if (!e.dataTransfer?.files?.length) return;
      onFiles(e.dataTransfer.files);
    });
  }

  function initStudio() {
    el("studio-open")?.addEventListener("click", () => openStudio("wm"));
    el("studio-close")?.addEventListener("click", () => {
      closeLightbox();
      el("studio-dialog").close();
    });
    el("wm-open-current")?.addEventListener("click", () => openStudio("wm", { importCurrent: true }));
    el("collage-open-current")?.addEventListener("click", () => openStudio("collage", { importCurrent: true }));
    el("sync-to-collage")?.addEventListener("click", () => {
      const bid = el("sync-to-collage").dataset.batchId;
      el("sync-dialog")?.close();
      openStudio("collage", { batchId: bid });
    });

    el("st-tab-wm")?.addEventListener("click", () => setStudioMode("wm"));
    el("st-tab-collage")?.addEventListener("click", () => setStudioMode("collage"));

    // Watermark source actions
    const wmInput = el("wm-upload-input");
    el("wm-upload-btn")?.addEventListener("click", () => wmInput.click());
    const handleWmFiles = async (files) => {
      showError("");
      el("wm-loading").classList.remove("hidden");
      try {
        const added = await uploadFilesToStudio(files);
        if (added.length) {
          studio.wmAssets.unshift(...added);
          renderWmStrip();
          selectWmAsset(added[0].id);
        }
      } catch (err) {
        showError(err.message);
      } finally {
        el("wm-loading").classList.add("hidden");
        wmInput.value = "";
      }
    };
    wmInput?.addEventListener("change", () => handleWmFiles(wmInput.files));
    bindDropTarget(el("wm-canvas"), handleWmFiles);

    el("wm-load-current")?.addEventListener("click", importCurrentJobToWm);
    el("wm-pick-library")?.addEventListener("click", () => openLibraryDrawer("wm"));
    el("wm-library-close")?.addEventListener("click", () => el("wm-library-drawer").classList.add("hidden"));
    el("wm-bg-toggle")?.addEventListener("click", () => {
      studio.wmCheckerBg = !studio.wmCheckerBg;
      el("wm-canvas").classList.toggle("checker", studio.wmCheckerBg);
      el("wm-bg-toggle").textContent = studio.wmCheckerBg ? "深色背景" : "浅格背景";
    });
    el("wm-reset-exif")?.addEventListener("click", () => {
      const asset = activeWmAsset();
      if (!asset) return;
      populateWmFieldsFromAsset(asset);
      el("wm-signature").value = "";
      scheduleWmPreview(true);
    });

    // Watermark reactive inputs
    [
      "wm-camera", "wm-lens", "wm-focal", "wm-aperture", "wm-shutter", "wm-iso",
      "wm-date", "wm-location", "wm-signature",
    ].forEach((id) => el(id)?.addEventListener("input", () => scheduleWmPreview(false)));
    ["wm-brand", "wm-show-brand", "wm-show-lens", "wm-show-date", "wm-show-loc", "wm-show-ev"].forEach((id) => {
      el(id)?.addEventListener("change", () => scheduleWmPreview(true));
    });
    el("wm-scale")?.addEventListener("input", () => {
      el("wm-scale-val").textContent = `${el("wm-scale").value}%`;
      scheduleWmPreview(false);
    });
    el("wm-export-btn")?.addEventListener("click", exportWmSingle);
    el("wm-batch-zip-btn")?.addEventListener("click", exportWmBatchZip);
    el("wm-send-collage")?.addEventListener("click", () => {
      const list = studio.wmAssets.length > 1 ? studio.wmAssets : (activeWmAsset() ? [activeWmAsset()] : []);
      for (const a of list) {
        if (!studio.colAssets.some((x) => x.id === a.id)) studio.colAssets.push(a);
      }
      renderCollageTray();
      setStudioMode("collage");
      scheduleCollagePreview(true);
    });

    // Collage source actions
    const colInput = el("col-upload-input");
    el("col-upload-btn")?.addEventListener("click", () => colInput.click());
    const handleColFiles = async (files) => {
      showError("");
      el("moments-loading")?.classList.remove("hidden");
      el("col-loading")?.classList.remove("hidden");
      try {
        const added = await uploadFilesToStudio(files);
        if (added.length) {
          studio.colAssets.push(...added);
          renderCollageTray();
          scheduleCollagePreview(true);
        }
      } catch (err) {
        showError(err.message);
      } finally {
        el("moments-loading")?.classList.add("hidden");
        el("col-loading")?.classList.add("hidden");
        colInput.value = "";
      }
    };
    colInput?.addEventListener("change", () => handleColFiles(colInput.files));
    bindDropTarget(el("col-preview-area"), handleColFiles);

    el("col-load-current")?.addEventListener("click", importCurrentJobToCollage);
    el("col-pick-library")?.addEventListener("click", () => openLibraryDrawer("collage"));
    el("col-library-close")?.addEventListener("click", () => el("col-library-drawer").classList.add("hidden"));
    el("col-clear-btn")?.addEventListener("click", () => {
      studio.colAssets = [];
      studio.momentsTiles = [];
      el("col-arrange-note").textContent = "";
      renderCollageTray();
    });

    document.querySelectorAll("[data-arrange]").forEach((btn) => {
      btn.addEventListener("click", () => runAutoArrange(btn.dataset.arrange));
    });

    el("col-view-moments")?.addEventListener("click", () => setCollageView("moments"));
    el("col-view-canvas")?.addEventListener("click", () => setCollageView("canvas"));

    // Moments mockup controls
    el("moments-light")?.addEventListener("click", () => {
      el("phone-frame").dataset.theme = "light";
      el("moments-light").classList.add("active");
      el("moments-dark").classList.remove("active");
    });
    el("moments-dark")?.addEventListener("click", () => {
      el("phone-frame").dataset.theme = "dark";
      el("moments-dark").classList.add("active");
      el("moments-light").classList.remove("active");
    });
    el("moments-show-order")?.addEventListener("change", () => {
      if (studio.momentsTiles.length) {
        const cols = el("moments-grid").classList.contains("cols-2") ? 2 : el("moments-grid").classList.contains("cols-1") ? 1 : 3;
        renderMomentsGrid(cols);
      }
    });
    el("moments-gen-caption")?.addEventListener("click", () => {
      if (studio.momentsSuggestedCaption) {
        el("moments-caption").value = studio.momentsSuggestedCaption;
      }
    });

    // Collage appearance & watermark controls
    el("col-bg-swatches")?.querySelectorAll("[data-bg]").forEach((btn) => {
      btn.addEventListener("click", () => {
        studio.colBg = btn.dataset.bg;
        el("col-bg-swatches").querySelectorAll("[data-bg]").forEach((b) => b.classList.toggle("active", b === btn));
        scheduleCollagePreview(true);
      });
    });
    el("col-tile-mode")?.addEventListener("change", () => scheduleCollagePreview(true));
    [
      ["col-gap", "col-gap-val", "px"],
      ["col-pad", "col-pad-val", "px"],
      ["col-rad", "col-rad-val", "px"],
      ["col-slice-x", "col-slice-x-val", "%"],
      ["col-slice-y", "col-slice-y-val", "%"],
    ].forEach(([id, valId, unit]) => {
      el(id)?.addEventListener("input", () => {
        el(valId).textContent = `${el(id).value}${unit}`;
        scheduleCollagePreview(false);
      });
    });
    el("col-slice-reset")?.addEventListener("click", () => {
      el("col-slice-x").value = "0";
      el("col-slice-y").value = "0";
      el("col-slice-x-val").textContent = "0%";
      el("col-slice-y-val").textContent = "0%";
      scheduleCollagePreview(true);
    });
    el("col-tile-wm")?.addEventListener("change", () => {
      el("col-wm-style-wrap").classList.toggle("hidden", !el("col-tile-wm").checked);
      scheduleCollagePreview(true);
    });
    el("col-wm-style")?.addEventListener("change", () => scheduleCollagePreview(true));
    el("col-show-footer")?.addEventListener("change", () => {
      el("col-footer-fields").classList.toggle("hidden", !el("col-show-footer").checked);
      scheduleCollagePreview(true);
    });
    ["col-footer-title", "col-footer-sub"].forEach((id) => {
      el(id)?.addEventListener("input", () => scheduleCollagePreview(false));
    });

    el("col-export-zip")?.addEventListener("click", () => exportCollage("zip"));
    el("col-export-jpg")?.addEventListener("click", () => exportCollage("jpg"));

    // Lightbox events
    el("st-lb-close")?.addEventListener("click", closeLightbox);
    el("st-lb-prev")?.addEventListener("click", () => {
      if (studio.lightboxIdx > 0) { studio.lightboxIdx -= 1; updateLightbox(); }
    });
    el("st-lb-next")?.addEventListener("click", () => {
      if (studio.lightboxIdx < studio.momentsTiles.length - 1) { studio.lightboxIdx += 1; updateLightbox(); }
    });
    el("st-lightbox")?.addEventListener("click", (e) => {
      if (e.target === el("st-lightbox")) closeLightbox();
    });
    el("studio-dialog")?.addEventListener("keydown", (e) => {
      if (el("st-lightbox")?.classList.contains("hidden")) return;
      if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); closeLightbox(); }
      else if (e.key === "ArrowLeft" && studio.lightboxIdx > 0) { studio.lightboxIdx -= 1; updateLightbox(); }
      else if (e.key === "ArrowRight" && studio.lightboxIdx < studio.momentsTiles.length - 1) { studio.lightboxIdx += 1; updateLightbox(); }
    });

    renderCollageTray();
  }

  // Expose helper for debugging / integration
  window.SuperPhotoStudio = { open: openStudio };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initStudio);
  else initStudio();
})();
