(function () {
  const dropZone = document.getElementById("drop-zone");
  const fileInput = document.getElementById("file-input");
  const btnAnalyze = document.getElementById("btn-analyze");
  const loader = document.getElementById("loader");
  const errorBanner = document.getElementById("error-banner");
  const results = document.getElementById("results");
  const messagesEl = document.getElementById("messages");
  const detailsEl = document.getElementById("details");
  const imagesEl = document.getElementById("images");

  let selectedFile = null;

  function showError(msg) {
    errorBanner.textContent = msg;
    errorBanner.classList.add("active");
  }

  function clearError() {
    errorBanner.classList.remove("active");
    errorBanner.textContent = "";
  }

  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
  });

  fileInput.addEventListener("change", () => {
    const f = fileInput.files[0];
    if (f) setFile(f);
  });

  function setFile(f) {
    selectedFile = f;
    dropZone.querySelector("p").textContent = f.name;
    clearError();
  }

  function opt(id) {
    return document.getElementById(id).checked;
  }

  btnAnalyze.addEventListener("click", async () => {
    clearError();
    results.hidden = true;
    if (!selectedFile) {
      showError("اختر صورة أولاً.");
      return;
    }

    const fd = new FormData();
    fd.append("file", selectedFile);

    const params = new URLSearchParams({
      skip_qwen: opt("opt-skip-qwen"),
      skip_brand: opt("opt-skip-brand"),
      skip_color: opt("opt-skip-color"),
      compare_db: opt("opt-compare-db"),
    });

    btnAnalyze.disabled = true;
    loader.classList.add("active");

    try {
      const res = await fetch("/api/analyze?" + params.toString(), {
        method: "POST",
        body: fd,
      });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        let err = res.statusText || "فشل الطلب";
        const d = data.detail;
        if (typeof d === "string") err = d;
        else if (Array.isArray(d)) err = d.map((x) => x.msg || JSON.stringify(x)).join("; ");
        else if (d && typeof d === "object") err = d.msg || JSON.stringify(d);
        showError(err);
        return;
      }
      render(data);
      results.hidden = false;
    } catch (e) {
      showError("تعذر الاتصال بالخادم. تأكد أن uvicorn يعمل.");
    } finally {
      btnAnalyze.disabled = false;
      loader.classList.remove("active");
    }
  });

  function render(data) {
    messagesEl.innerHTML = "";
    (data.messages || []).forEach((m) => {
      const div = document.createElement("div");
      div.className = "msg " + (m.level || "info");
      div.textContent = m.text;
      messagesEl.appendChild(div);
    });

    const s = data.summary || {};
    const pr = s.plate_read || {};
    const bm = s.brand_model || {};
    const vc = s.vehicle_color || {};
    const comp = data.db_comparison;

    let compHtml = "";
    if (comp) {
      compHtml = `
        <div class="kv" style="margin-top:0.75rem;padding-top:0.75rem;border-top:1px solid var(--border)">
          <div><span class="key">تطابق السجل</span><span>${comp.registry_consistent ? "نعم" : "لا"}</span></div>
          <div><span class="key">لوحة</span><span>${comp.plate_final_match ? "✓" : "✗"}</span></div>
          <div><span class="key">ماركة</span><span>${comp.brand_match === null ? "—" : comp.brand_match ? "✓" : "✗"}</span></div>
          <div><span class="key">موديل</span><span>${comp.model_match === null ? "—" : comp.model_match ? "✓" : "✗"}</span></div>
          <div><span class="key">لون</span><span>${comp.color_match === null ? "—" : comp.color_match ? "✓" : "✗"}</span></div>
        </div>`;
    }

    detailsEl.innerHTML = `
      <div class="kv">
        <div><span class="key">مركبة</span><span>${s.vehicle ? "مكتشفة" : "لا"}</span></div>
        <div><span class="key">لوحة (كشف)</span><span>${s.plate ? "مكتشفة" : "لا"}</span></div>
        <div><span class="key">اللوحة (قراءة)</span><span>${pr.final || "—"}</span></div>
        <div><span class="key">ماركة (تقدير)</span><span>${bm.brand || "—"}</span></div>
        <div><span class="key">موديل (تقدير)</span><span>${bm.model || "—"}</span></div>
        <div><span class="key">لون (تقدير)</span><span>${vc.color || "—"}</span></div>
      </div>
      ${compHtml}
    `;

    imagesEl.innerHTML = "";
    const order = [
      ["00_overview.jpg", "ملخص الصورة"],
      ["04_plate_crop.jpg", "قص اللوحة"],
      ["05_plate_enhanced.png", "لوحة محسّنة"],
    ];
    const imgs = data.images || {};
    order.forEach(([name, title]) => {
      const url = imgs[name];
      if (!url) return;
      const wrap = document.createElement("div");
      wrap.innerHTML = `<p style="margin:0 0 0.5rem;color:var(--muted);font-size:0.85rem">${title}</p>`;
      const img = document.createElement("img");
      img.className = "result-img";
      img.src = url;
      img.alt = title;
      wrap.appendChild(img);
      imagesEl.appendChild(wrap);
    });
  }
})();
