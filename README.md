# FGVC-IR8: Fine-Grained Iranian Vehicle Categorization and Open-Set Recognition

**FGVC-IR8** is a standalone computer vision inference engine and web application for fine-grained classification and open-set recognition of Iranian market vehicles. It is designed to run offline on standard consumer CPUs (8 GB RAM) with zero external framework dependencies.

- **Backbone:** Offline `DINOv2-ViT-B/14-reg4` with fused LoRA adapters (`688,128` trainable parameters merged into base weights)
- **Head Architecture:** Competitive Part-Slot Attention (8 foreground part slots + 2 background garbage slots), 5D Relational Graph Transformer, `SubCenterArcFace` (`K=4` sub-centers per class), and Multi-Instance Set Evidence Pooling (`Set-MIL`)
- **Open-Set Defense:** SVD Subspace Null-Residual verification (`OpenSetGuard`) based on ViM and NECO
- **Validation Performance:** `97.99%` Balanced Accuracy (`97.72%` Raw Accuracy) across `5,098` multi-image vehicle listings (`12,100` frames)

[English Documentation](#1-problem-statement-and-engineering-scope) | [مستندات فارسی](#مستندات-فارسی)

---

## 1. Problem Statement and Engineering Scope

Real-world vehicle listings present four technical challenges that cause standard closed-set classifiers to fail:

1. **Frontal Viewpoint Ambiguity in Sibling Models:** The **Peugeot 206 Hatchback (`206 HB`)** and **Peugeot 206 Sedan (`206 SD`)** share identical front bumpers, headlights, hoods, and grilles. When a photo captures only the front view, the rear trunk is occluded. Standard classifiers trained on imbalanced market data (`5.2 : 1` HB-to-SD ratio) default to the majority class (`206 HB`) with overconfident probabilities.
2. **Shortcut Learning on Localized Artifacts:** Vision Transformers often overfit to 1-3 patches containing brand emblems or license plates rather than overall body geometry.
3. **Out-of-Distribution (OOD) Overconfidence:** Standard Softmax layers force probabilities across known classes to sum to `100%`. Uploading an unsupported vehicle common in the local market (such as *Saipa Pride* or *IKCO Samand*) results in a high-confidence false positive on the nearest sedan or hatchback class.
4. **Harsh Lighting and Multi-Vehicle Clutter:** User-uploaded images frequently suffer from severe nighttime underexposure or contain multiple parked cars in the background.

---

## 2. System Pipeline and Architecture

```text
Input Images (1 to 8 frames per vehicle)
  |
  +--> [1. Photometric & Occlusion Preprocessor]
  |      - Rank-1 Separable Annular Inpainter (41x41 and 81x81 box filters)
  |      - Multi-Scale Retinex (MSR) shadow reflectance recovery for low-light frames
  |
  +--> [2. Single-Pass 224px Evaluator & Autonomous Instance Cropper]
  |      - 2D Connected-Component Flood-Fill + Multi-Peak Local Maxima NMS
  |      - Cosine-Gain Gate: applies crop only if cos_crop >= cos_full + 0.025
  |
  +--> [3. Fused DINOv2-ViT-B/14-reg4 Backbone]
  |      - 12 Transformer blocks with 4 register tokens
  |      - Top-block [CLS] Spatial Mass Capping (limits single-patch dominance)
  |      - Hollow-Ring (5x5 minus 3x3) token outlier filter for badge/emblem invariance
  |
  +--> [4. Competitive Slot Attention & 5D Relational Graph]
  |      - 8 Foreground Part Slots + 2 Background Garbage Slots (3 GRU iterations)
  |      - 5D Pairwise Relational Transformer -> 512-D L2-normalized embedding
  |      - Zero-Copy Confidence-Gated Dual-Head Fusion (Step-35686 + Step-28039 heads)
  |
  +--> [5. OpenSetGuard & Cross-Scale Decision Engine]
         - SVD Subspace Null-Residual check + Virtual OOD Logit probability damping
         - Multi-Vehicle Conflict Detector across uploaded frames
         - Symmetric Cross-Scale Consensus (392px + 224px) for Peugeot 206 HB vs. 206 SD
```

### Data Engineering Pipeline (ETL)

1. **Resilient API Harvesting:** Built a stateful crawler targeting structured listing endpoints using temporal cursor pagination (`last_post_date`), compressed `viewed_tokens` state tracking, Gaussian jitter (`1.8s` to `3.4s`), bounded exponential backoff (`180s` cap), and dual-storage persistence (**SQLite WAL** + real-time flushed **JSONL**). Ground-truth class labels were extracted directly from structured `BREADCRUMB` hierarchies.
2. **Zero-Shot Scene Filtering:** Used **CLIP (`ViT-B/32`)** to compute cosine similarity against positive exterior prompts and negative noise prompts, automatically rejecting cabin interiors, dashboards, engine bays, and paper documents.
3. **Target Vehicle Localization:** Applied **YOLOv8n** with a weighted target-selection heuristic combining bounding-box area ratio, center proximity, and detection confidence, followed by a `12%` dynamic safety padding margin to preserve mirrors, wheels, and bumpers.
4. **Dataset Sharding:** Packaged into **92 training shards** and **15 validation shards** (**5,098 validation listings / 12,100 frames**) across 8 target vehicle classes.

---

## 3. Validation Benchmarks (5,098 Listings)

The model was trained for **35,686 optimizer steps** across a two-phase resolution curriculum (`224x224` in Phase 1 for 8 epochs; `392x392` in Phase 2 for 6 epochs with gradient checkpointing and cosine annealing).

### 3.1. Training Progression Across Phases

| Phase & Epoch | Step | Listing Acc (`acc`) | Balanced Acc (`bal`) | Frame Acc (`frame`) | `206HB -> SD` | `206SD -> HB` | Checkpoint Tag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `s224` - Epoch 1 | `2,549` | `97.00%` | `97.31%` | `96.22%` | `7.6%` | `7.8%` | `s224-epoch1` |
| `s224` - Epoch 3 | `7,647` | `97.37%` | `97.58%` | `96.69%` | `7.5%` | `7.8%` | `s224-epoch3` |
| `s224` - Epoch 5 | `12,745` | `97.37%` | `97.91%` | `96.41%` | `8.5%` | `3.9%` | `s224-epoch5` |
| `s224` - Epoch 7 | `17,843` | `97.63%` | `97.91%` | `96.96%` | `7.3%` | `6.1%` | `s224-epoch7` |
| `s224` - Epoch 8 | `20,392` | `97.45%` | `97.90%` | `96.71%` | `8.2%` | `5.2%` | `s224-epoch8` |
| `s392` - Epoch 1 | `22,941` | `97.65%` | `97.94%` | `97.28%` | `7.3%` | `6.1%` | `s392-epoch1` |
| **`s392` - Epoch 3** | **`28,039`** | **`97.72%`** | **`97.99%`** | **`97.25%`** | **`7.1%`** | **`6.1%`** | **`best.pt`** |
| **`s392` - Epoch 6** | **`35,686`** | **`97.72%`** | **`97.97%`** | **`97.02%`** | **`7.1%`** | **`6.1%`** | **`final-evaluated`** |

### 3.2. Per-Class Recall and Family Accuracy

| Class ID | Vehicle Model | Validation Listings | Strict Recall | Family-Level Accuracy | Notes |
| :---: | :--- | :---: | :---: | :---: | :--- |
| `c0` | **Peugeot Pars** | `1,225` | `99.84%` (`1,223/1,225`) | `99.84%` | 2 errors |
| `c1` | **Peugeot 206 HB** | `1,148` | `92.33%` (`1,060/1,148`) | `99.39%` | 81 frontal-only listings (`HB -> SD`) |
| `c2` | **Peugeot 206 SD** | `231` | `93.94%` (`217/231`) | `100.00%` | 14 frontal-only listings (`SD -> HB`), 0 non-206 errors |
| `c3` | **Peugeot 207 HB** | `628` | `99.04%` (`622/628`) | `99.04%` | Aftermarket 206-to-207 headlight swaps |
| `c4` | **IKCO Dena** | `667` | `99.85%` (`666/667`) | `99.85%` | 1 error |
| `c5` | **IKCO Tara** | `259` | `99.23%` (`257/259`) | `99.23%` | 2 errors |
| `c6` | **Saipa Quick** | `661` | `99.85%` (`660/661`) | `99.85%` | 1 error |
| `c7` | **Saipa Shahin** | `279` | `100.00%` (`279/279`) | `100.00%` | 0 errors |
| **Total** | **All 8 Classes** | **`5,098`** | **`97.72%` Raw / `97.99%` Bal** | **`99.59%` (`5,077/5,098`)** | Only 21 cross-family errors in total |

---

## 4. Local CPU Runtime Benchmarks

Tested on a **Microsoft Surface Laptop 3 (Intel Core CPU, 8 GB RAM, Windows 11, CPU-Only)**:

| Operation / Mode | Resolution | CPU Latency | Verification Result |
| :--- | :---: | :---: | :--- |
| **Engine Initialization (`load_all_artifacts`)** | - | `0.73 s - 0.98 s` | Offline load of fused FP16 backbone + dual heads |
| **Separable Annular Inpainter (`81x81`)** | Native | `2.2 ms` | Exact equivalence to 2D box pool (`max_diff = 8.94e-8`) |
| **Fast Mode (`fast_224`)** | `224 x 224` | `~450 - 550 ms` | Single-pass cached backbone + 8-slot graph + heatmap |
| **Pro Mode (`pro_392`)** | `392 x 392` + `224 x 224` | `~1,100 - 1,350 ms` | Multi-scale consensus pass + part-slot attention map |
| **Emblem Occlusion Stability Test** | `224 x 224` | - | `pixel MAE = 2.66 / 255`, `cosine drift = 0.0030 < 0.025` |
| **Open-Set OOD Rejection (Saipa Pride)** | `392 x 392` | `~1,150 ms` | Rejected (`status = ood`, max class bar damped to `< 28%`) |

---

## 5. Quick Start and Installation

The application uses Python's standard `ThreadingHTTPServer` alongside `torch`, `torchvision`, `numpy`, and `Pillow`.

### Option 1: 1-Click Windows Launch (`run.bat`)

1. Clone the repository:
   ```bash
   git clone [https://github.com/ahmadZMVD/FGVC-IR8.git](https://github.com/ahmadZMVD/FGVC-IR8.git)
   cd FGVC-IR8
   ```
2. Double-click **`run.bat`** (or execute `.\run.bat` in PowerShell / Command Prompt).
   - **First run:** Checks Python packages, creates a local `.venv` if required, automatically downloads `fgvc_ir8_laptop_standalone_fp16.pt` (`172.4 MB`) and `best.pt` (`15.4 MB`) from [GitHub Releases (`v1.0.0`)](https://github.com/ahmadZMVD/FGVC-IR8/releases/tag/v1.0.0) into `checkpoints/`, and saves `.env_verified.json`.
   - **Subsequent runs:** Skips setup checks and starts the web server immediately at `http://127.0.0.1:8080`.

### Option 2: Manual CLI and Self-Test Execution

Install dependencies and run the 9-stage unit test suite:
```bash
pip install -r requirements.txt
python engine.py --selftest
```

Launch the web interface (supports instant `FA` / `EN` language switching and `Fast 224` / `Pro 392` mode toggling):
```bash
python app.py
```

---

## 6. Repository Structure

```text
FGVC-IR8/
├── run.bat                  # 1-Click Windows launcher
├── launcher.py              # Environment bootstrapper, weight downloader & fast-boot manager
├── engine.py                # Standalone DINOv2-LoRA + 8-Slot Graph + OpenSetGuard engine (v8.2)
├── app.py                   # Bilingual (EN/FA) web dashboard & REST API server
├── requirements.txt         # CPU dependencies (torch, torchvision, numpy, Pillow)
└── checkpoints/
    ├── fgvc_ir8_laptop_standalone_fp16.pt  # Fused standalone bundle (in Releases v1.0.0)
    ├── best.pt                             # Step 28,039 best balanced accuracy head (in Releases v1.0.0)
    ├── BEST.json                           # Validation metadata for best checkpoint (0.9799)
    ├── MANIFEST.json                       # Checkpoint SHA-256 manifest
    └── train_log.txt                       # Complete 35,686-step training and evaluation log
```

---

## مستندات فارسی

**FGVC-IR8** یک موتور استنتاج بینایی ماشین و وب‌اپلیکیشن مستقل برای طبقه‌بندی ریزدانه‌ای ۸ خودروی پرتیراژ بازار ایران (پژو پارس، پژو ۲۰۶ هاچ‌بک، پژو ۲۰۶ صندوقدار، پژو ۲۰۷، دنا، تارا، کوییک و شاهین) و تشخیص خودروهای خارج از لیست (مانند پراید و سمند) است.

### ویژگی‌های کلیدی معماری

1. **مهندسی داده و پایپ‌لاین ETL:** استخراج خودکار آگهی‌ها با مدیریت وضعیت در SQLite WAL و استریم JSONL، حذف تصاویر نامربوط (کابین، موتور، اسناد) با مدل CLIP به صورت Zero-Shot، و برش ایمن خودروی اصلی با YOLOv8n همراه با حاشیه امن ۱۲ درصدی در قالب ۱۰۷ شارد آموزشی و ارزیابی.
2. **معماری DINOv2-LoRA و گراف قطعات:** ادغام ریاضی آداپتورهای LoRA در وزن‌های پایه DINOv2 به همراه ۸ اسلات توجه رقابتی برای قطعات بدنه و ۲ لایه گراف رابطه‌ای ۵ بعدی.
3. **تفکیک بدون سوگیری پژو ۲۰۶ هاچ‌بک و صندوقدار:** استفاده از قانون اجماع دومقیاسه (۳۹۲ و ۲۲۴ پیکسل) که در تصاویر صرفاً روبه‌رو (بدون دید صندوق عقب) وضعیت «نمای جلو - نیازمند دید عقب» را اعلام می‌کند و با دیدن کوچک‌ترین نشانه از ستون C یا صندوق عقب در هر یک از عکس‌ها، تیپ دقیق را مشخص می‌سازد.
4. **تشخیص خودروهای ناشناس (`OpenSetGuard`):** محاسبه باقی‌مانده فضای تهی با تجزیه مقادیر منفرد (SVD) زیرمراکز کلاس‌ها جهت رد کردن خودروهای خارج از ۸ کلاس و جلوگیری از باد کردن کاذب درصدها در نمودار.
5. **نصب و اجرای تک‌کلیکی:** با دابل‌کلیک روی فایل `run.bat` در ویندوز، پیش‌نیازها و فایل‌های وزن مدل به صورت خودکار از بخش Releases دریافت شده و رابط کاربری دو زبانه (فارسی و انگلیسی) روی آدرس `http://127.0.0.1:8080` اجرا می‌شود.
