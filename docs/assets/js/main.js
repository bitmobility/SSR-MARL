(() => {
  "use strict";
  const config = window.PROJECT_CONFIG || {};
  const safeURL = (value) => {
    if (!value || typeof value !== "string") return "";
    try {
      const url = new URL(value, document.baseURI);
      return ["http:", "https:", "file:"].includes(url.protocol) ? url.href : "";
    } catch { return ""; }
  };
  const codeURL = safeURL(config.codeUrl);
  if (codeURL) {
    const link = document.getElementById("code-link");
    link.href = codeURL;
    link.hidden = false;
    document.getElementById("code-pending").hidden = true;
  }

  // Lazy metadata loading keeps the page light; missing recordings retain
  // their editorial placeholders. Adding a video requires no HTML changes.
  function loadVideo(card) {
    const entry = (config.videos || {})[card.dataset.video];
    const src = safeURL(entry?.src);
    if (!src) return;
    const stage = card.querySelector(".video-stage");
    const placeholder = stage.querySelector(".video-placeholder");
    const video = document.createElement("video");
    video.controls = true;
    video.playsInline = true;
    video.preload = "metadata";
    video.hidden = true;
    video.setAttribute("aria-label", card.dataset.videoLabel || card.querySelector("h3")?.textContent || "Simulation demonstration");
    const poster = safeURL(entry.poster);
    if (poster) video.poster = poster;
    const captions = safeURL(entry.captions);
    if (captions) {
      const track = document.createElement("track");
      Object.assign(track, { kind: "captions", src: captions, srclang: "en", label: "English", default: true });
      video.append(track);
    }
    video.addEventListener("loadedmetadata", () => {
      placeholder.hidden = true;
      video.hidden = false;
      card.classList.add("video-ready");
    });
    video.addEventListener("error", () => {
      video.hidden = true;
      placeholder.hidden = false;
      card.classList.remove("video-ready");
    });
    stage.append(video);
    video.src = src;
  }
  const cards = document.querySelectorAll("[data-video]");
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver(entries => entries.forEach(entry => {
      if (entry.isIntersecting) { observer.unobserve(entry.target); loadVideo(entry.target); }
    }), { rootMargin: "250px" });
    cards.forEach(card => observer.observe(card));
  } else cards.forEach(loadVideo);

  // Accessible tabs; without JavaScript both result panels remain visible.
  const tabList = document.querySelector(".tab-list");
  const tabs = [...tabList.querySelectorAll('[role="tab"]')];
  function selectTab(selected, focus = false) {
    tabs.forEach(tab => {
      const active = tab === selected;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      const panel = document.getElementById(tab.getAttribute("aria-controls"));
      panel.setAttribute("role", "tabpanel");
      panel.tabIndex = 0;
      // The selected panel owns both its recording and its results.
      if (!active) panel.querySelectorAll("video").forEach(video => video.pause());
      panel.hidden = !active;
    });
    if (focus) selected.focus();
  }
  tabList.hidden = false;
  selectTab(tabs[0]);
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", event => {
      let next;
      if (event.key === "ArrowRight") next = (i + 1) % tabs.length;
      if (event.key === "ArrowLeft") next = (i - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") next = 0;
      if (event.key === "End") next = tabs.length - 1;
      if (next !== undefined) { event.preventDefault(); selectTab(tabs[next], true); }
    });
  });

  const copy = document.getElementById("copy-citation");
  const status = document.getElementById("copy-status");
  copy.hidden = false;
  copy.addEventListener("click", async () => {
    const citation = document.getElementById("bibtex");
    try {
      await navigator.clipboard.writeText(citation.textContent);
      status.textContent = "BibTeX copied to clipboard.";
      copy.textContent = "Copied ✓";
      setTimeout(() => { copy.textContent = "Copy BibTeX"; }, 2200);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(citation);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      status.textContent = "Citation selected. Press Ctrl+C (or ⌘C) to copy, or download the .bib file.";
    }
  });

  const dialog = document.getElementById("figure-dialog");
  if (typeof dialog.showModal === "function") {
    let opener;
    document.querySelectorAll("[data-zoom]").forEach(link => link.addEventListener("click", event => {
      event.preventDefault();
      opener = link;
      const source = link.querySelector("img");
      const expanded = document.getElementById("expanded-figure");
      expanded.src = link.href;
      expanded.alt = source.alt;
      document.getElementById("figure-description").textContent = source.alt;
      dialog.showModal();
      document.body.style.overflow = "hidden";
    }));
    document.getElementById("close-figure").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", event => {
      const box = dialog.getBoundingClientRect();
      if (event.target === dialog && (event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom)) dialog.close();
    });
    dialog.addEventListener("close", () => { document.body.style.overflow = ""; opener?.focus(); });
  }
})();
