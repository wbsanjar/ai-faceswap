(() => {
  "use strict";

  const slots = { source: null, target: null };

  const $ = (sel) => document.querySelector(sel);
  const swapBtn = $("#swap-btn");
  const statusEl = $("#status");
  const resultSection = $("#result");
  const resultImg = $("#result-img");
  const downloadBtn = $("#download-btn");

  /* ---------- IndexedDB session persistence ----------
     Images and the last result survive page reloads and
     browser restarts, so you never re-upload or re-swap. */

  const DB_NAME = "faceswap-studio";
  const STORE = "session";
  let _dbP = null;

  function openDB() {
    if (_dbP) return _dbP;
    _dbP = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => req.result.createObjectStore(STORE);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    return _dbP;
  }

  async function dbSet(key, value) {
    try {
      const db = await openDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).put(value, key);
        tx.oncomplete = resolve;
        tx.onerror = () => reject(tx.error);
      });
    } catch (_) {
      /* storage unavailable — app still works in-memory */
    }
  }

  async function dbGet(key) {
    try {
      const db = await openDB();
      return new Promise((resolve, reject) => {
        const req = db.transaction(STORE).objectStore(STORE).get(key);
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    } catch (_) {
      return undefined;
    }
  }

  async function dbDel(key) {
    try {
      const db = await openDB();
      return new Promise((resolve, reject) => {
        const tx = db.transaction(STORE, "readwrite");
        tx.objectStore(STORE).delete(key);
        tx.oncomplete = resolve;
        tx.onerror = () => reject(tx.error);
      });
    } catch (_) {}
  }

  /* ---------- file selection ---------- */

  const MAX_DIM = 1600;

  function downscale(file) {
    return new Promise((resolve) => {
      if (file.type === "image/svg+xml") return resolve(file);
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, MAX_DIM / Math.max(img.width, img.height));
        if (scale >= 1 && file.size <= 10 * 1024 * 1024) {
          URL.revokeObjectURL(url);
          return resolve(file);
        }
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(img.width * scale));
        canvas.height = Math.max(1, Math.round(img.height * scale));
        canvas.getContext("2d").drawImage(img, 0, 0, canvas.width, canvas.height);
        const mime = file.type === "image/png" ? "image/png" : "image/jpeg";
        canvas.toBlob((blob) => {
          URL.revokeObjectURL(url);
          if (!blob) return resolve(file);
          resolve(new File([blob], file.name || "image", { type: blob.type }));
        }, mime, 0.92);
      };
      img.onerror = () => resolve(file);
      img.src = url;
    });
  }

  function setFile(slot, file) {
    slots[slot] = file;
    dbSet(slot, file);
    const preview = $(`[data-preview="${slot}"]`);
    const drop = preview.closest(".drop");
    const card = drop.closest(".card");

    if (file) {
      preview.src = URL.createObjectURL(file);
      preview.hidden = false;
      drop.classList.add("has-image");
      card.classList.add("has-image");
    }
    updateReady();
  }

  function clearSlot(slot) {
    slots[slot] = null;
    dbDel(slot);
    const preview = $(`[data-preview="${slot}"]`);
    const drop = preview.closest(".drop");
    const card = drop.closest(".card");
    preview.hidden = true;
    preview.src = "";
    drop.classList.remove("has-image", "dragover");
    card.classList.remove("has-image");
    const input = $("#file-" + slot);
    input.value = "";
    statusEl.textContent = "";
    statusEl.className = "status";
    updateReady();
  }

  /* ---------- restore the previous session ---------- */

  async function restoreSession() {
    const [src, tgt, resultData, resultMime] = await Promise.all([
      dbGet("source"),
      dbGet("target"),
      dbGet("result"),
      dbGet("resultMime"),
    ]);
    if (src) setFile("source", src);
    if (tgt) setFile("target", tgt);
    if (resultData) {
      resultImg.src = resultData;
      downloadBtn.href = resultData;
      resultSection.hidden = false;
      resultImg.onload = () => resultSection.scrollIntoView({ behavior: "smooth" });
    }
  }

  /* ---------- wire up dropzones ---------- */

  document.querySelectorAll("[data-drop]").forEach((drop) => {
    const slot = drop.dataset.drop;
    const input = $("#file-" + slot);

    drop.addEventListener("click", () => input.click());

    input.addEventListener("change", async () => {
      if (input.files[0]) setFile(slot, await downscale(input.files[0]));
    });

    ["dragenter", "dragover"].forEach((evt) =>
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.classList.add("dragover");
      })
    );
    ["dragleave", "drop"].forEach((evt) =>
      drop.addEventListener(evt, (e) => {
        e.preventDefault();
        drop.classList.remove("dragover");
      })
    );
    drop.addEventListener("drop", async (e) => {
      const file = e.dataTransfer.files[0];
      if (file && file.type.startsWith("image/")) setFile(slot, await downscale(file));
      else statusEl.textContent = "Please drop an image file.";
    });
  });

  document.querySelectorAll("[data-clear]").forEach((btn) =>
    btn.addEventListener("click", () => clearSlot(btn.dataset.clear))
  );

  $("#swap-direction").addEventListener("click", () => {
    const a = slots.source;
    const b = slots.target;
    const aImg = $('[data-preview="source"]');
    const bImg = $('[data-preview="target"]');
    if (a) setFile("target", a);
    if (b) setFile("source", b);
    if (!a) clearSlot("target");
    if (!b) clearSlot("source");
    aImg && aImg.closest(".drop").classList.add("dragover");
    bImg && bImg.closest(".drop").classList.add("dragover");
  });

  /* ---------- state / buttons ---------- */

  function updateReady() {
    swapBtn.disabled = !(slots.source && slots.target);
  }

  $("#again-btn").addEventListener("click", () => {
    clearSlot("source");
    clearSlot("target");
    Promise.all([dbDel("result"), dbDel("resultMime")]);
    resultSection.hidden = true;
    $("#studio").scrollIntoView({ behavior: "smooth" });
  });

  /* ---------- the swap ---------- */

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  const HEALTH_POLL_MS = 2000;
  const MAX_WARM_WAIT_MS = 300000; // 5 min of model-warm waiting
  const FETCH_TIMEOUT_MS = 120000; // generous cap so the button never hangs
  const MAX_TRANSPORT_RETRIES = 8;

  function friendlyError(err) {
    if (err && err.code === "ABORT") {
      return "Request took too long (cold start). Retrying automatically… wait for the next attempt.";
    }
    if (err instanceof TypeError || String(err && err.message).toLowerCase().includes("failed to fetch")) {
      return "Server se connect nahi ho paya (network / cold-start). Agar ye first launch hai to models download ho rahe hain — 2-5 min baad phir try karo. Dobaara koshish ho rahi hai.";
    }
    return (err && err.message) || "Swap failed. Please try again.";
  }

  async function swapOnce() {
    const form = new FormData();
    form.append("source", slots.source);
    form.append("target", slots.target);

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
    try {
      const res = await fetch("/api/swap", {
        method: "POST",
        body: form,
        signal: controller.signal,
      });
      const data = await res.json().catch(() => ({}));

      if (res.status === 202) {
        const boom = new Error(data.message || "Models loading…");
        boom.code = "WARMING";
        throw boom;
      }
      if (res.status === 503) {
        const boom = new Error(data.error || "AI models could not be loaded.");
        boom.code = "MODELS_FAILED";
        throw boom;
      }
      if (!res.ok || !data.image) {
        throw new Error(data.error || "Swap failed. Please try again.");
      }

      const dataUrl = "data:" + (data.mime || "image/png") + ";base64," + data.image;
      resultImg.src = dataUrl;
      downloadBtn.href = dataUrl;
      resultSection.hidden = false;
      statusEl.className = "status";
      statusEl.textContent = "Done! Enjoy your swap.";
      resultSection.scrollIntoView({ behavior: "smooth" });
      dbSet("result", dataUrl);
      dbSet("resultMime", data.mime || "image/png");
      return true;
    } catch (err) {
      if (err && err.name === "AbortError") {
        const abort = new Error("Request took too long.");
        abort.code = "ABORT";
        throw abort;
      }
      throw err;
    } finally {
      clearTimeout(timer);
    }
  }

  /* Resolves to null while the models are still warming, or throws with the
     real reason when the warm-up failed / the wait ran out. */
  async function waitForHealth(deadline) {
    for (;;) {
      let health = null;
      try {
        const res = await fetch("/api/health", { cache: "no-store" });
        health = await res.json().catch(() => null);
      } catch (_) {
        /* server not up yet — keep polling */
      }
      if (health && health.ok) {
        if (health.ai) return;
        if (health.ai_status === "failed") {
          const boom = new Error(
            "AI models load nahi ho paye: " + (health.ai_error || "unknown error")
          );
          boom.code = "MODELS_FAILED";
          throw boom;
        }
      }
      if (Date.now() >= deadline) {
        const boom = new Error(
          "Models load hone me bahut time lag raha hai. Page refresh kar ke " +
          "dobaara try karo, ya thodi der baad."
        );
        boom.code = "WARM_TIMEOUT";
        throw boom;
      }
      const left = Math.max(0, Math.ceil((deadline - Date.now()) / 1000));
      statusEl.textContent =
        "AI models load ho rahe hain… " + left + "s mein ruk jayenge agar load na ho. " +
        "Pehli baar 700 MB+ download hota hai.";
      await sleep(HEALTH_POLL_MS);
    }
  }

  swapBtn.addEventListener("click", async () => {
    if (!slots.source || !slots.target) return;

    swapBtn.disabled = true;
    resultSection.hidden = true;
    statusEl.className = "status loading";
    statusEl.textContent = "Analyzing faces and warping meshes…";
    const started = Date.now();
    let transportRetries = 0;

    try {
      for (;;) {
        try {
          if (await swapOnce()) break;
        } catch (err) {
          if (err && err.code === "WARMING") {
            statusEl.textContent = "✓ Images ready. " + (err.message || "");
            await waitForHealth(started + MAX_WARM_WAIT_MS);
            continue; // models ready → retry the swap
          }
          if (err && err.code === "MODELS_FAILED") throw err;
          if (Date.now() - started < 60000 && transportRetries < MAX_TRANSPORT_RETRIES) {
            transportRetries += 1;
            statusEl.textContent =
              "Connectivity issue / cold start — dobara koshish ho rahi hai… (" +
              friendlyError(err) + ")";
            await sleep(Math.min(15000, 2000 * transportRetries));
            continue;
          }
          throw err;
        }
      }
    } catch (err) {
      statusEl.className = "status error";
      statusEl.textContent = friendlyError(err);
    } finally {
      swapBtn.disabled = false;
    }
  });

  restoreSession();
})();