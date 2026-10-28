#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py — FGVC-IR8 Web Interface (v9.5 with Auto Port-Takeover & Live Engine Hot-Reload)
========================================================================================
Local HTTP server at http://127.0.0.1:8080
- Automatically frees port 8080 on startup if a stale old Python server is still running in RAM.
- Automatically hot-reloads `engine.py` whenever `engine.py` is modified on disk (no restart needed!).
- Displays the active engine version badge (`v9.5`) in the top bar so you never hit a stale process.
"""

import base64
import importlib
import io
import json
import os
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from PIL import Image

import engine

CKPT_DIR = os.environ.get("FGVC_CKPT_DIR", "checkpoints")
PORT = int(os.environ.get("FGVC_PORT", "8080"))

HTML_PAGE = """<!DOCTYPE html>
<html lang="fa" dir="rtl" id="htmlRoot">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FGVC-IR8 v9.5</title>
<style>
  :root {
    --bg-main: #212121;
    --bg-panel: #2f2f2f;
    --bg-card: #171717;
    --bg-hover: #3a3a3a;
    --border: rgba(255, 255, 255, 0.09);
    --border-strong: rgba(255, 255, 255, 0.18);
    --text-primary: #ececec;
    --text-secondary: #b4b4b4;
    --text-muted: #7d7d7d;
    --accent: #ffffff;
    --green: #10a37f;
    --amber: #d97706;
    --red: #ef4444;
    --radius-lg: 18px;
    --radius-md: 12px;
  }

  * {
    box-sizing: border-box;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Tahoma, sans-serif;
    -webkit-font-smoothing: antialiased;
  }

  body {
    margin: 0;
    background-color: var(--bg-main);
    color: var(--text-primary);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  .topbar {
    height: 60px;
    padding: 0 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    background: rgba(33, 33, 33, 0.92);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
    z-index: 40;
  }

  .topbar-right {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .version-pill {
    background: rgba(16, 163, 127, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 163, 127, 0.4);
    border-radius: 999px;
    padding: 4px 11px;
    font-size: 11.5px;
    font-weight: 700;
    direction: ltr;
  }

  .model-switch {
    position: relative;
    display: inline-flex;
    align-items: center;
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    padding: 3px;
    width: 236px;
    height: 36px;
    cursor: pointer;
    user-select: none;
    direction: ltr;
  }

  .switch-thumb {
    position: absolute;
    top: 3px;
    bottom: 3px;
    width: calc(50% - 3px);
    background: var(--bg-panel);
    border: 1px solid rgba(255, 255, 255, 0.14);
    border-radius: 999px;
    transition: transform 0.22s cubic-bezier(0.22, 1, 0.36, 1);
  }

  .model-switch[data-mode="pro_392"] .switch-thumb { transform: translateX(100%); }
  .model-switch[data-mode="fast_224"] .switch-thumb { transform: translateX(0%); }

  .switch-opt {
    flex: 1;
    text-align: center;
    font-size: 12.5px;
    font-weight: 600;
    z-index: 2;
    color: var(--text-secondary);
    transition: color 0.18s ease;
  }

  .model-switch[data-mode="fast_224"] .opt-fast,
  .model-switch[data-mode="pro_392"] .opt-pro {
    color: var(--text-primary);
  }

  .lang-btn {
    background: var(--bg-panel);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .lang-btn:hover { background: var(--bg-hover); }

  .main-wrap {
    width: 100%;
    max-width: 880px;
    margin: 0 auto;
    padding: 32px 20px 80px;
    flex: 1;
  }

  .composer {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px;
    transition: border-color 0.2s ease;
  }

  .composer.dragover { border-color: var(--text-primary); }

  .drop-area {
    border: 1px dashed var(--border-strong);
    border-radius: var(--radius-md);
    padding: 28px 20px;
    text-align: center;
    cursor: pointer;
    transition: background 0.18s ease;
  }

  .drop-area:hover { background: var(--bg-hover); }

  .drop-title {
    font-size: 14.5px;
    font-weight: 500;
    color: var(--text-primary);
    margin-top: 8px;
  }

  .drop-sub {
    font-size: 12px;
    color: var(--text-secondary);
    margin-top: 4px;
  }

  .thumbs {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 16px;
  }

  .thumb-item {
    position: relative;
    width: 74px;
    height: 74px;
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border-strong);
    background: var(--bg-card);
  }

  .thumb-item img { width: 100%; height: 100%; object-fit: cover; }

  .thumb-remove {
    position: absolute;
    top: 4px;
    left: 4px;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: rgba(0, 0, 0, 0.78);
    color: #fff;
    border: none;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
  }

  .composer-footer {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
    gap: 12px;
  }

  .auto-tag { font-size: 12px; color: var(--text-muted); }
  .actions { display: flex; align-items: center; gap: 10px; }

  .btn-clear {
    background: transparent;
    color: var(--text-secondary);
    border: 1px solid var(--border-strong);
    padding: 7px 16px;
    border-radius: 999px;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s ease;
  }

  .btn-clear:hover { color: var(--text-primary); background: var(--bg-hover); }

  .btn-run {
    background: var(--accent);
    color: #111111;
    border: none;
    padding: 7px 22px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s ease;
  }

  .btn-run:hover:not(:disabled) { opacity: 0.9; }
  .btn-run:disabled { opacity: 0.32; cursor: not-allowed; }

  .progress-wrap { margin-top: 14px; display: none; }
  .progress-track {
    width: 100%;
    height: 3px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 999px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    width: 0%;
    background: var(--text-primary);
    border-radius: 999px;
    transition: width 0.22s ease;
  }

  .progress-label {
    font-size: 11.5px;
    color: var(--text-secondary);
    margin-top: 6px;
    display: flex;
    justify-content: space-between;
  }

  .results {
    margin-top: 24px;
    display: none;
    animation: fadeUp 0.25s ease forwards;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
  }

  .verdict-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin-bottom: 18px;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .verdict-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    flex-wrap: wrap;
  }

  .status-dot {
    width: 9px;
    height: 9px;
    border-radius: 50%;
    display: inline-block;
    background: var(--green);
    flex-shrink: 0;
  }

  .status-dot.ambiguous_206, .status-dot.conflict { background: var(--amber); }
  .status-dot.ood { background: var(--red); }

  .verdict-title {
    font-size: 19px;
    font-weight: 600;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .verdict-conf {
    font-size: 20px;
    font-weight: 600;
    color: var(--text-primary);
    direction: ltr;
  }

  .verdict-sub {
    margin: 0;
    font-size: 13.5px;
    color: var(--text-secondary);
    line-height: 1.6;
  }

  .telemetry-bar {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    padding-top: 12px;
    margin-top: 6px;
    border-top: 1px solid var(--border);
  }

  .tel-item { display: flex; flex-direction: column; gap: 2px; }
  .tel-label { font-size: 11px; color: var(--text-muted); }
  .tel-val { font-size: 13px; font-weight: 500; color: var(--text-primary); font-variant-numeric: tabular-nums; }

  .section-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 20px 24px;
    margin-bottom: 18px;
  }

  .section-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 16px;
  }

  .section-title { font-size: 14px; font-weight: 600; color: var(--text-primary); margin: 0; }
  .prob-list { display: flex; flex-direction: column; gap: 11px; }

  .prob-row {
    display: grid;
    grid-template-columns: 150px 1fr 110px;
    align-items: center;
    gap: 14px;
  }

  .prob-name {
    font-size: 13px;
    color: var(--text-primary);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .prob-track {
    height: 7px;
    background: var(--bg-panel);
    border-radius: 999px;
    overflow: hidden;
  }

  .prob-fill {
    height: 100%;
    width: 0%;
    background: var(--text-primary);
    border-radius: 999px;
    transition: width 0.45s cubic-bezier(0.22, 1, 0.36, 1);
  }

  .prob-row.top-rank .prob-fill { background: var(--green); }

  .prob-stats {
    font-size: 12px;
    color: var(--text-secondary);
    direction: ltr;
    text-align: end;
    font-variant-numeric: tabular-nums;
  }

  .frames-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 16px;
  }

  .frame-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .frame-img-wrap {
    position: relative;
    width: 100%;
    aspect-ratio: 4 / 3;
    background: #111;
    overflow: hidden;
  }

  .frame-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
    cursor: zoom-in;
  }

  .view-toggle {
    position: absolute;
    bottom: 8px;
    left: 8px;
    background: rgba(23, 23, 23, 0.85);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 11px;
    cursor: pointer;
  }

  .zoom-hint {
    position: absolute;
    top: 8px;
    right: 8px;
    background: rgba(23, 23, 23, 0.75);
    color: var(--text-secondary);
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 10.5px;
    pointer-events: none;
  }

  .frame-body { padding: 13px; display: flex; flex-direction: column; gap: 7px; }
  .frame-head { display: flex; justify-content: space-between; align-items: center; font-size: 12.5px; font-weight: 600; }

  .frame-weight-pill {
    font-size: 11px;
    font-weight: 500;
    color: var(--text-secondary);
    background: rgba(255, 255, 255, 0.06);
    padding: 2px 7px;
    border-radius: 5px;
  }

  .frame-table {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-top: 6px;
    border-top: 1px solid var(--border);
    font-size: 12px;
  }

  .frame-kv { display: flex; justify-content: space-between; align-items: center; gap: 8px; }
  .frame-k { color: var(--text-secondary); }
  .frame-v { color: var(--text-primary); font-variant-numeric: tabular-nums; }

  .lightbox {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.88);
    z-index: 100;
    display: none;
    align-items: center;
    justify-content: center;
    padding: 24px;
  }

  .lightbox.open { display: flex; }

  .lightbox-content {
    max-width: 900px;
    width: 100%;
    background: var(--bg-card);
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-lg);
    overflow: hidden;
  }

  .lightbox-img-box {
    width: 100%;
    max-height: 75vh;
    background: #0d0d0d;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .lightbox-img-box img { max-width: 100%; max-height: 75vh; object-fit: contain; }

  .lightbox-bar {
    padding: 12px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-top: 1px solid var(--border);
  }

  .lightbox-close {
    background: var(--bg-panel);
    color: var(--text-primary);
    border: 1px solid var(--border-strong);
    padding: 5px 14px;
    border-radius: 999px;
    font-size: 12.5px;
    cursor: pointer;
  }

  @media (max-width: 640px) {
    .telemetry-bar { grid-template-columns: repeat(2, 1fr); }
    .prob-row { grid-template-columns: 115px 1fr 85px; }
  }
</style>
</head>
<body>

<header class="topbar">
  <div class="topbar-right">
    <div id="modelSwitch" class="model-switch" data-mode="pro_392" onclick="toggleModelSwitch()">
      <div class="switch-thumb"></div>
      <span id="optFastTxt" class="switch-opt opt-fast">سریع ۲۲۴</span>
      <span id="optProTxt" class="switch-opt opt-pro">دقیق ۳۹۲</span>
    </div>
    <span id="verBadge" class="version-pill">FGVC-IR8 v9.5</span>
  </div>
  <button id="langBtn" class="lang-btn" onclick="toggleLanguage()">EN</button>
</header>

<main class="main-wrap">
  <section id="composerBox" class="composer">
    <div class="drop-area" onclick="document.getElementById('fileInput').click()">
      <input type="file" id="fileInput" multiple accept="image/*" style="display:none" onchange="addFiles(this.files)">
      <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="#b4b4b4" stroke-width="1.6" style="margin:0 auto; display:block;">
        <rect x="3" y="3" width="18" height="18" rx="3" ry="3"></rect>
        <circle cx="8.5" cy="8.5" r="1.5"></circle>
        <polyline points="21 15 16 10 5 21"></polyline>
      </svg>
      <div id="dropTitle" class="drop-title">تصاویر خودرو را انتخاب کنید یا اینجا رها نمایید</div>
      <div id="dropSub" class="drop-sub">۱ تا ۸ تصویر</div>
    </div>

    <div id="thumbsContainer" class="thumbs" style="display:none;"></div>

    <div id="progressWrap" class="progress-wrap">
      <div class="progress-track"><div id="progressBar" class="progress-bar"></div></div>
      <div class="progress-label">
        <span id="progressStepTxt">در حال پردازش...</span>
        <span id="progressPctTxt">0%</span>
      </div>
    </div>

    <div class="composer-footer">
      <span id="autoFeatureTag" class="auto-tag">پیش‌پردازش نوری و برش خودکار فعال (Hot-Reload v9.5)</span>
      <div class="actions">
        <button id="clearBtn" class="btn-clear" onclick="clearImages()">پاک کردن</button>
        <button id="analyzeBtn" class="btn-run" onclick="runAnalysis()" disabled>تحلیل</button>
      </div>
    </div>
  </section>

  <section id="resultsSection" class="results">
    <div class="verdict-card">
      <div class="verdict-top">
        <h2 class="verdict-title">
          <span id="statusDot" class="status-dot"></span>
          <span id="verdictTitleTxt"></span>
        </h2>
        <div id="verdictConfTxt" class="verdict-conf"></div>
      </div>
      <p id="verdictSubTxt" class="verdict-sub"></p>

      <div class="telemetry-bar">
        <div class="tel-item">
          <span id="lblTelCos" class="tel-label">شباهت کسینوسی</span>
          <span id="telCos" class="tel-val">-</span>
        </div>
        <div class="tel-item">
          <span id="lblTelConc" class="tel-label">تمرکز قطعات</span>
          <span id="telConc" class="tel-val">-</span>
        </div>
        <div class="tel-item">
          <span id="lblTel206" class="tel-label">خانواده ۲۰۶</span>
          <span id="tel206" class="tel-val">-</span>
        </div>
        <div class="tel-item">
          <span id="lblTelTime" class="tel-label">زمان پاسخ</span>
          <span id="telTime" class="tel-val">-</span>
        </div>
      </div>
    </div>

    <div class="section-card">
      <div class="section-header">
        <h3 id="secProbTitle" class="section-title">توزیع احتمال کلاس‌ها</h3>
      </div>
      <div id="probList" class="prob-list"></div>
    </div>

    <div class="section-card">
      <div class="section-header">
        <h3 id="secFramesTitle" class="section-title">تحلیل تصاویر و نقشه توجه</h3>
      </div>
      <div id="framesGrid" class="frames-grid"></div>
    </div>
  </section>
</main>

<div id="lightboxModal" class="lightbox" onclick="closeLightbox(event)">
  <div class="lightbox-content" onclick="event.stopPropagation()">
    <div class="lightbox-img-box">
      <img id="lightboxImg" src="">
    </div>
    <div class="lightbox-bar">
      <span id="lightboxCaption" style="font-size:13px; font-weight:600;"></span>
      <div style="display:flex; gap:8px;">
        <button id="lightboxToggleBtn" class="lightbox-close" onclick="toggleLightboxView()"></button>
        <button id="lightboxCloseBtn" class="lightbox-close" onclick="closeLightbox()"></button>
      </div>
    </div>
  </div>
</div>

<script>
let currentLang = 'fa';
let currentModel = 'pro_392';
let imagesB64 = [];
let lastResponseData = null;
let progTimer = null;
let lbState = { heat: '', orig: '', mode: 'heat', title: '' };

const I18N = {
  fa: {
    dir: 'rtl',
    langBtn: 'EN',
    optFast: 'سریع ۲۲۴',
    optPro: 'دقیق ۳۹۲',
    dropTitle: 'تصاویر خودرو را انتخاب کنید یا اینجا رها نمایید',
    dropSub: '۱ تا ۸ تصویر',
    autoTag: 'پیش‌پردازش نوری و برش خودکار فعال (Hot-Reload v9.5)',
    clearBtn: 'پاک کردن',
    analyzeBtn: 'تحلیل',
    prog1: 'پیش‌پردازش تصویر...',
    prog2: 'استخراج ویژگی‌ها...',
    prog3: 'ارزیابی نهایی...',
    lblCos: 'شباهت کسینوسی',
    lblConc: 'تمرکز قطعات',
    lbl206: 'خانواده ۲۰۶',
    lblTime: 'زمان پاسخ',
    secProb: 'توزیع احتمال کلاس‌ها',
    secFrames: 'تحلیل تصاویر و نقشه توجه',
    frameTitle: 'تصویر #',
    weightPill: 'وزن: ',
    indepPred: 'خروجی مستقل',
    cosSim: 'شباهت کسینوسی',
    partConc: 'تمرکز قطعات',
    photoQual: 'کیفیت نوری',
    restoredTag: ' (اصلاح نوری)',
    croppedTag: ' · برش خودکار',
    showOrig: 'تصویر اصلی',
    showHeat: 'نقشه توجه',
    zoomHint: 'بزرگ‌نمایی',
    closeModal: 'بستن'
  },
  en: {
    dir: 'ltr',
    langBtn: 'FA',
    optFast: 'Fast 224',
    optPro: 'Pro 392',
    dropTitle: 'Select vehicle images or drag and drop here',
    dropSub: '1 to 8 images',
    autoTag: 'Auto-crop and exposure recovery active (Hot-Reload v9.5)',
    clearBtn: 'Clear',
    analyzeBtn: 'Analyze',
    prog1: 'Preprocessing frames...',
    prog2: 'Extracting features...',
    prog3: 'Computing predictions...',
    lblCos: 'Cosine Similarity',
    lblConc: 'Part Concentration',
    lbl206: '206 Family Prob',
    lblTime: 'Latency',
    secProb: 'Class Probability Distribution',
    secFrames: 'Frame Inspection & Attention Maps',
    frameTitle: 'Frame #',
    weightPill: 'Weight: ',
    indepPred: 'Frame Verdict',
    cosSim: 'Cosine Similarity',
    partConc: 'Part Concentration',
    photoQual: 'Light Quality',
    restoredTag: ' (Restored)',
    croppedTag: ' · Cropped',
    showOrig: 'Original',
    showHeat: 'Attention Map',
    zoomHint: 'Zoom',
    closeModal: 'Close'
  }
};

function toggleModelSwitch() {
  currentModel = (currentModel === 'pro_392') ? 'fast_224' : 'pro_392';
  document.getElementById('modelSwitch').setAttribute('data-mode', currentModel);
  if (imagesB64.length > 0 && lastResponseData !== null) {
    runAnalysis();
  }
}

function toggleLanguage() {
  currentLang = (currentLang === 'fa') ? 'en' : 'fa';
  applyLanguage();
  if (lastResponseData !== null) {
    displayResults(lastResponseData);
  }
}

function applyLanguage() {
  const t = I18N[currentLang];
  const root = document.getElementById('htmlRoot');
  root.setAttribute('lang', currentLang);
  root.setAttribute('dir', t.dir);

  document.getElementById('langBtn').textContent = t.langBtn;
  document.getElementById('optFastTxt').textContent = t.optFast;
  document.getElementById('optProTxt').textContent = t.optPro;
  document.getElementById('dropTitle').textContent = t.dropTitle;
  document.getElementById('dropSub').textContent = t.dropSub;
  document.getElementById('autoFeatureTag').textContent = t.autoTag;
  document.getElementById('clearBtn').textContent = t.clearBtn;
  document.getElementById('analyzeBtn').textContent = t.analyzeBtn;
  document.getElementById('lblTelCos').textContent = t.lblCos;
  document.getElementById('lblTelConc').textContent = t.lblConc;
  document.getElementById('lblTel206').textContent = t.lbl206;
  document.getElementById('lblTelTime').textContent = t.lblTime;
  document.getElementById('secProbTitle').textContent = t.secProb;
  document.getElementById('secFramesTitle').textContent = t.secFrames;
  document.getElementById('lightboxCloseBtn').textContent = t.closeModal;
}

const comp = document.getElementById('composerBox');
comp.addEventListener('dragover', e => { e.preventDefault(); comp.classList.add('dragover'); });
comp.addEventListener('dragleave', () => comp.classList.remove('dragover'));
comp.addEventListener('drop', e => {
  e.preventDefault();
  comp.classList.remove('dragover');
  if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
});

function addFiles(fileList) {
  Array.from(fileList).forEach(file => {
    if (imagesB64.length >= 8) return;
    const reader = new FileReader();
    reader.onload = e => {
      imagesB64.push(e.target.result);
      renderThumbs();
    };
    reader.readAsDataURL(file);
  });
}

function removeImage(idx) {
  imagesB64.splice(idx, 1);
  renderThumbs();
}

function renderThumbs() {
  const box = document.getElementById('thumbsContainer');
  box.innerHTML = '';
  if (!imagesB64.length) {
    box.style.display = 'none';
    document.getElementById('analyzeBtn').disabled = true;
    return;
  }
  box.style.display = 'flex';
  document.getElementById('analyzeBtn').disabled = false;
  imagesB64.forEach((b64, i) => {
    const div = document.createElement('div');
    div.className = 'thumb-item';
    div.innerHTML = `<img src="${b64}"><button class="thumb-remove" onclick="removeImage(${i})">×</button>`;
    box.appendChild(div);
  });
}

function clearImages() {
  imagesB64 = [];
  lastResponseData = null;
  document.getElementById('fileInput').value = '';
  renderThumbs();
  document.getElementById('resultsSection').style.display = 'none';
}

function startProgress() {
  const t = I18N[currentLang];
  const wrap = document.getElementById('progressWrap');
  const bar = document.getElementById('progressBar');
  const stepTxt = document.getElementById('progressStepTxt');
  const pctTxt = document.getElementById('progressPctTxt');
  wrap.style.display = 'block';
  let pct = 12;
  bar.style.width = pct + '%';
  pctTxt.textContent = pct + '%';
  stepTxt.textContent = t.prog1;

  clearInterval(progTimer);
  progTimer = setInterval(() => {
    if (pct < 88) {
      pct += Math.max(2, Math.floor((90 - pct) * 0.18));
      bar.style.width = pct + '%';
      pctTxt.textContent = pct + '%';
      if (pct > 36 && pct <= 68) stepTxt.textContent = t.prog2;
      else if (pct > 68) stepTxt.textContent = t.prog3;
    }
  }, 150);
}

function finishProgress() {
  clearInterval(progTimer);
  const bar = document.getElementById('progressBar');
  const pctTxt = document.getElementById('progressPctTxt');
  bar.style.width = '100%';
  pctTxt.textContent = '100%';
  setTimeout(() => {
    document.getElementById('progressWrap').style.display = 'none';
    bar.style.width = '0%';
  }, 240);
}

async function runAnalysis() {
  if (!imagesB64.length) return;
  const btn = document.getElementById('analyzeBtn');
  btn.disabled = true;
  startProgress();

  try {
    const resp = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        images: imagesB64,
        model_mode: currentModel
      })
    });
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    lastResponseData = data;
    finishProgress();
    displayResults(data);
  } catch (err) {
    clearInterval(progTimer);
    document.getElementById('progressWrap').style.display = 'none';
    alert(err.message);
  } finally {
    btn.disabled = false;
  }
}

function toggleFrameImage(btn, idx) {
  const t = I18N[currentLang];
  const img = document.getElementById('fimg-' + idx);
  const mode = img.getAttribute('data-mode');
  if (mode === 'heat') {
    img.src = img.getAttribute('data-orig');
    img.setAttribute('data-mode', 'orig');
    btn.textContent = t.showHeat;
  } else {
    img.src = img.getAttribute('data-heat');
    img.setAttribute('data-mode', 'heat');
    btn.textContent = t.showOrig;
  }
}

function openLightbox(idx) {
  if (!lastResponseData) return;
  const t = I18N[currentLang];
  const f = lastResponseData.frames[idx - 1];
  const cardImg = document.getElementById('fimg-' + idx);
  const curMode = cardImg ? cardImg.getAttribute('data-mode') : 'heat';

  lbState.heat = f.heatmap_b64;
  lbState.orig = f.orig_b64;
  lbState.mode = curMode;
  lbState.title = `${t.frameTitle}${f.index} — ${currentLang === 'fa' ? f.top_class_fa : f.top_class_en}`;

  document.getElementById('lightboxImg').src = (curMode === 'heat') ? lbState.heat : lbState.orig;
  document.getElementById('lightboxCaption').textContent = lbState.title;
  document.getElementById('lightboxToggleBtn').textContent = (curMode === 'heat') ? t.showOrig : t.showHeat;
  document.getElementById('lightboxModal').classList.add('open');
}

function toggleLightboxView() {
  const t = I18N[currentLang];
  if (lbState.mode === 'heat') {
    lbState.mode = 'orig';
    document.getElementById('lightboxImg').src = lbState.orig;
    document.getElementById('lightboxToggleBtn').textContent = t.showHeat;
  } else {
    lbState.mode = 'heat';
    document.getElementById('lightboxImg').src = lbState.heat;
    document.getElementById('lightboxToggleBtn').textContent = t.showOrig;
  }
}

function closeLightbox(e) {
  document.getElementById('lightboxModal').classList.remove('open');
}

function displayResults(d) {
  const t = I18N[currentLang];
  const sec = document.getElementById('resultsSection');
  sec.style.display = 'block';

  if (d.engine_version) {
    document.getElementById('verBadge').textContent = 'FGVC-IR8 ' + d.engine_version;
  }

  document.getElementById('statusDot').className = 'status-dot ' + d.status;
  document.getElementById('verdictTitleTxt').textContent = (currentLang === 'fa') ? d.title_fa : d.title_en;
  document.getElementById('verdictConfTxt').textContent = d.final_conf_pct + '%';
  document.getElementById('verdictSubTxt').textContent = (currentLang === 'fa') ? d.subtitle_fa : d.subtitle_en;

  document.getElementById('telCos').textContent = d.max_cosine.toFixed(4);
  document.getElementById('telConc').textContent = d.mean_slot_conc.toFixed(3);
  document.getElementById('tel206').textContent = d.p_206_family_pct + '%';
  document.getElementById('telTime').textContent = d.elapsed_ms + ' ms';

  const plist = document.getElementById('probList');
  plist.innerHTML = '';
  d.class_probs.forEach((c, idx) => {
    const row = document.createElement('div');
    row.className = 'prob-row' + (idx === 0 && d.status !== 'ood' ? ' top-rank' : '');
    const cname = (currentLang === 'fa') ? c.name_fa : c.name_en;
    row.innerHTML = `
      <div class="prob-name">${cname}</div>
      <div class="prob-track"><div class="prob-fill" style="width:0%" data-w="${Math.min(100, c.prob_pct)}%"></div></div>
      <div class="prob-stats">${c.prob_pct.toFixed(1)}% · ${c.cosine.toFixed(3)}</div>
    `;
    plist.appendChild(row);
  });

  requestAnimationFrame(() => {
    document.querySelectorAll('.prob-fill').forEach(el => {
      el.style.width = el.getAttribute('data-w');
    });
  });

  const fgrid = document.getElementById('framesGrid');
  fgrid.innerHTML = '';
  d.frames.forEach(f => {
    const card = document.createElement('div');
    card.className = 'frame-card';
    const indepName = (currentLang === 'fa') ? f.top_class_fa : f.top_class_en;
    const cropBadge = f.is_auto_cropped ? t.croppedTag : '';
    const darkBadge = f.is_dark_restored ? t.restoredTag : '';

    card.innerHTML = `
      <div class="frame-img-wrap">
        <img id="fimg-${f.index}" src="${f.heatmap_b64}" data-heat="${f.heatmap_b64}" data-orig="${f.orig_b64}" data-mode="heat" onclick="openLightbox(${f.index})">
        <span class="zoom-hint">${t.zoomHint}</span>
        <button class="view-toggle" onclick="toggleFrameImage(this, ${f.index})">${t.showOrig}</button>
      </div>
      <div class="frame-body">
        <div class="frame-head">
          <span>${t.frameTitle}${f.index}${cropBadge}</span>
          <span class="frame-weight-pill">${t.weightPill}${f.weight_pct}%</span>
        </div>
        <div class="frame-table">
          <div class="frame-kv">
            <span class="frame-k">${t.indepPred}</span>
            <span class="frame-v">${indepName}</span>
          </div>
          <div class="frame-kv">
            <span class="frame-k">${t.cosSim}</span>
            <span class="frame-v">${f.max_cosine.toFixed(4)}</span>
          </div>
          <div class="frame-kv">
            <span class="frame-k">${t.partConc}</span>
            <span class="frame-v">${f.slot_concentration.toFixed(3)}</span>
          </div>
          <div class="frame-kv">
            <span class="frame-k">${t.photoQual}</span>
            <span class="frame-v">${f.photometric_quality}%${darkBadge}</span>
          </div>
        </div>
      </div>
    `;
    fgrid.appendChild(card);
  });
}
</script>
</body>
</html>
"""

ENGINE_INSTANCE = None
ENGINE_MTIME: float = 0.0


def get_or_reload_engine():
    global ENGINE_INSTANCE, ENGINE_MTIME
    eng_file = os.path.abspath(engine.__file__)
    cur_mtime = os.path.getmtime(eng_file) if os.path.exists(eng_file) else 0.0
    if ENGINE_INSTANCE is None or (cur_mtime > ENGINE_MTIME > 0.0):
        if ENGINE_INSTANCE is not None:
            print(f"[HOT-RELOAD] Detected updated engine.py on disk — reloading module...")
            importlib.reload(engine)
        ENGINE_INSTANCE = engine.FGVCInferenceEngine(ckpt_dir=CKPT_DIR, device_str="cpu")
        ENGINE_MTIME = cur_mtime
    return ENGINE_INSTANCE


def free_port_if_occupied(port: int) -> None:
    """Terminates any stale background python server holding port 8080 on Windows/Linux."""
    my_pid = os.getpid()
    if os.name == "nt":
        try:
            out = subprocess.check_output(["netstat", "-ano"], text=True, errors="ignore")
            for line in out.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    if pid != my_pid and pid > 0:
                        print(f"[PORT-GUARD] Terminating stale server process PID={pid} on port {port}...")
                        subprocess.run(["taskkill", "/F", "/PID", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_response(404)
            self.end_headers()
            return
        try:
            eng = get_or_reload_engine()
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            b64_list = body.get("images", [])
            model_mode = str(body.get("model_mode", "pro_392"))

            pil_imgs = []
            for b64_str in b64_list[:8]:
                raw_b64 = b64_str.split(",", 1)[-1]
                im = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
                pil_imgs.append(im)

            res = eng.predict(pil_imgs, model_mode=model_mode)
            cropped_cnt = sum(1 for f in res["frames"] if f["is_auto_cropped"])
            ver = res.get("engine_version", "v9.5")
            print(f"[PREDICT {ver}] mode={model_mode} frames={res['n_frames']} cropped={cropped_cnt} "
                  f"status={res['status']} max_cos={res['max_cosine']:.4f} conf={res['final_conf_pct']}% time={res['elapsed_ms']}ms")
            payload = json.dumps(res, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            err_payload = json.dumps({"error": str(e)}, ensure_ascii=False).encode("utf-8")
            self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(err_payload)


def main():
    free_port_if_occupied(PORT)
    eng = get_or_reload_engine()
    server = ThreadingHTTPServer(("127.0.0.1", PORT), DashboardHandler)
    url = f"http://127.0.0.1:{PORT}"
    print(f"[SERVER] FGVC-IR8 {eng.version} listening on {url}")
    print("[SERVER] Press Ctrl+C to stop.")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    server.serve_forever()


if __name__ == "__main__":
    main()