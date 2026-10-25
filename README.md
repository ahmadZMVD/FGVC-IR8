\# FGVC-IR8: Fine-Grained Iranian Vehicle Categorization \& Open-Set Recognition Engine



\[!\[Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square\&logo=python\&logoColor=white)](https://www.python.org/)

\[!\[PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?style=flat-square\&logo=pytorch\&logoColor=white)](https://pytorch.org/)

\[!\[Backbone](https://img.shields.io/badge/Backbone-DINOv2--ViT--B%2F14--reg4-0467DF?style=flat-square)](https://github.com/facebookresearch/dinov2)

\[!\[Balanced Accuracy](https://img.shields.io/badge/Val\_Balanced\_Acc-97.99%25-10a37f?style=flat-square)](#3-training-progression--validation-benchmarks)

\[!\[Release Weights](https://img.shields.io/badge/Weights-v1.0.0\_Release-2563eb?style=flat-square)](https://github.com/ahmadZMVD/FGVC-IR8/releases/tag/v1.0.0)



<details>

<summary><b>🇮🇷 مطالعه مستندات کامل به زبان فارسی (Click to expand Persian Documentation)</b></summary>



<div dir="rtl" align="right">



\### معرفی پروژه و هدف مهندسی

سامانه‌ی \*\*FGVC-IR8\*\* یک موتور استنتاج بینایی ماشین برای \*\*طبقه‌بندی ریزدانه‌ای (Fine-Grained Visual Categorization)\*\* و \*\*تشخیص خودروهای خارج از توزیع (Open-Set Recognition)\*\* در بازار خودروی ایران است. این پروژه کل چرخه‌ی مهندسی یادگیری ماشین را از استخراج و پالایش خودکار داده‌های خام وب (ETL)، طراحی معماری مبتنی بر گراف قطعات و توجه اسلات‌ها، آموزش دو مرحله‌ای با مدیریت عدم تعادل داده، تا استقرار سبک روی پردازنده‌های مرکزی (CPU با رم ۸ گیگابایت) پوشش می‌دهد.



\---



\### ۱. پایپ‌لاین مهندسی داده و استخراج خودکار (Data Acquisition \& Curation ETL)

به دلیل نبود دیتاست استاندارد از خودروهای بازار ایران در شرایط نوری و زوایای واقعی، پایپ‌لاین جمع‌آوری و پالایش داده در ۴ لایه طراحی و اجرا شد:

1\. \*\*مهندسی معکوس API و خزش مقاوم (Stateful Crawling):\*\* استخراج مستقیم متادیتا و تصاویر از اندپوینت‌های داخلی جستجو (`/v8/postlist/w/search`) و جزئیات آگهی (`/v8/posts-v2/web/post/{token}`) با مدیریت پیجینیشن مبتنی بر کرسر زمانی (`last\_post\_date`) و بافر فشرده‌ی `viewed\_tokens`. برای پایداری در اجراهای طولانی‌مدت ویندوز، از پایگاه داده \*\*SQLite در حالت WAL\*\* به همراه استریم هم‌زمان \*\*JSONL\*\*، تاخیر تصادفی گوسی (`1.8s` تا `3.4s`)، مکانیزم قطع مدار (`Circuit Breaker`) و غیرفعال‌سازی سیستمی `QuickEdit` و `Standby` ویندوز استفاده شد. برچسب‌های دقیق سلسله‌مراتبی (سازنده، مدل، تیپ) به صورت خودکار از ساختار درختی `BREADCRUMB` استخراج شدند.

2\. \*\*فیلتر صحنه بدون نظارت (CLIP Zero-Shot Gatekeeper):\*\* استفاده از مدل `openai/clip-vit-base-patch32` برای محاسبه شباهت کسینوسی تصاویر با پرامپت‌های متنی و حذف خودکار تصاویر نامربوط (داخل کابین، داشبورد، موتور زیر کاپوت، برگه‌های کارشناسی و اسناد) پیش از ورود به دیتاست.

3\. \*\*مکان‌یابی خودروی هدف و برش ایمن (YOLOv8n + Dynamic Safety Padding):\*\* شناسایی کادر خودروها و تفکیک خودروی اصلی آگهی از خودروهای پارک‌شده در پس‌زمینه با تابع امتیازدهی وزن‌دار بر اساس نسبت مساحت (`AreaRatio`)، فاصله از مرکز کادر (`CenterDistance`) و ضریب اطمینان (`Confidence`). برای جلوگیری از بریده شدن جزئیات کلیدی (آینه‌ها، رینگ، زه‌های جانبی و چراغ‌ها)، حاشیه امن ۱۲ درصدی به کادر اضافه شده و تصاویر زیر `200x200` پیکسل حذف شدند.

4\. \*\*بسته‌بندی شاردهای آموزشی:\*\* مجموعه‌ی نهایی در قالب \*\*۱۰۷ شارد استاندارد\*\* شامل \*\*۹۲ شارد آموزش (Train)\*\* و \*\*۱۵ شارد ارزیابی (Validation)\*\* شامل \*\*۵,۰۹۸ آگهی مستقل (۱۲,۱۰۰ فریم)\*\* سازمان‌دهی شد.



\---



\### ۲. معماری شبکه عصبی و نوآوری‌های الگوریتمی

\* \*\*ستون فقرات (Backbone):\*\* مدل `DINOv2-ViT-B/14-reg4` (دارای ۴ توکن رجیستر برای جذب نویزهای پس‌زمینه) مجهز به آداپتورهای \*\*LoRA\*\* (`688,128` پارامتر قابل آموزش در ۲۸ تانسور) که در نسخه نهایی به صورت ریاضی در وزن‌های پایه ادغام (`Weight Fusion`) شده‌اند.

\* \*\*توجه رقابتی اسلات‌ها (`CompetitiveSlotAttention`):\*\* استخراج ۸ اسلات معنایی قطعه (`n\_part\_slots = 8`) به همراه ۲ اسلات زباله‌گیر پس‌زمینه (`n\_garbage\_slots = 2`) در ۳ تکرار GRU با سقف جرم مکانی (`Spatial Mass Capping`) جهت جلوگیری از تمرکز تک‌نقطه‌ای روی آرم یا پلاک.

\* \*\*گراف رابطه‌ای ۵ بعدی (`5D Relational Graph`):\*\* مدل‌سازی هندسه نسبی قطعات خودرو از طریق بردارهای ۵ بعدی $(\\Delta x/\\sigma, \\Delta y/\\sigma, d/\\sigma, \\ln(\\sigma\_j/\\sigma\_i), \\ln(M\_j/M\_i))$ در ۲ لایه ترنسفورمر رابطه‌ای.

\* \*\*هد زاویه‌ای زیرمرکزی (`SubCenterArcFace`) و تجمیع چندفریمی (`SetMILHead`):\*\* اختصاص ۴ زیرمرکز (`Ks = 4`) به ازای هر کلاس برای پوشش زوایای دید مختلف (جلو، عقب، سه‌رخ) و تجمیع شواهد چند تصویر از یک خودرو در سطح آگهی.

\* \*\*تفکیک بدون سوگیری پژو ۲۰۶ هاچ‌بک (`206 HB`) و صندوقدار (`206 SD`):\*\* پیاده‌سازی قانون اجماع متقارن دومقیاسه (`392px + 224px`) که در نماهای صرفاً روبه‌رو (که صندوق عقب فیزیکاً پنهان است) به جای حدس تصادفی، وضعیت «نمای جلو — نیازمند دید عقب/جانبی» را گزارش می‌دهد و با دیدن کوچک‌ترین نشانه از ستون C یا صندوق در هر یک از فریم‌ها، تیپ دقیق را قفل می‌کند.

\* \*\*ماژول تشخیص خودروهای ناشناس (`OpenSetGuard`):\*\* ترکیب تجزیه مقادیر منفرد زیرمراکز (`SVD Null-Space Residual` بر پایه متدهای \*ViM\* و \*NECO\*) با مخروط ژئودزیک تطبیق‌یافته با کیفیت نوری، جهت رد کردن خودروهای خارج از ۸ کلاس (مانند پراید، سمند، ۴۰۵) و مهار اعتمادبه‌نفس کاذب Softmax.



\---



\### ۳. راهنمای نصب و اجرای تک‌کلیکی در ویندوز

1\. مخزن را کلون یا دانلود کنید:

&#x20;  `git clone https://github.com/ahmadZMVD/FGVC-IR8.git`

2\. روی فایل \*\*`run.bat`\*\* دابل‌کلیک کنید. در اجرای اول، در صورت نبودن فایل وزن در پوشه `checkpoints/`، اسکریپت `launcher.py` به صورت خودکار آن را از بخش \[Releases](https://github.com/ahmadZMVD/FGVC-IR8/releases) دانلود کرده، محیط را کش می‌کند (`.env\_verified.json`) و در اجراهای بعدی، سرور وب را در کمتر از ۱ ثانیه روی آدرس `http://127.0.0.1:8080` بالا می‌آورد.



</div>

</details>



\---



\## 1. Overview \& Problem Statement



\*\*FGVC-IR8\*\* is an end-to-end computer vision system engineered for \*\*Fine-Grained Visual Categorization (FGVC)\*\* and \*\*Open-Set Recognition (OSR)\*\* of Iranian market vehicles under unconstrained real-world conditions.



Real-world vehicle listings present severe visual challenges that break standard closed-set image classifiers:

1\. \*\*Minimal Inter-Class Variance \& Viewpoint Ambiguity:\*\* Sibling vehicles such as the \*\*Peugeot 206 Hatchback (`206 HB`)\*\* and \*\*Peugeot 206 Sedan (`206 SD`)\*\* share identical front-end sheet metal, headlights, bumpers, and emblems. From a purely frontal angle, the rear trunk is physically occluded, causing standard models trained on imbalanced datasets (`5.2 : 1` HB-to-SD ratio) to suffer from severe majority-class prior bias.

2\. \*\*Shortcut Learning on Badges \& Artifacts:\*\* Standard Vision Transformers frequently overfit to localized 1–3 patch high-frequency regions (such as grille emblems or license plates) rather than structural body geometry.

3\. \*\*Closed-Set Overconfidence on Out-of-Distribution (OOD) Vehicles:\*\* Because standard Softmax layers force class probabilities to sum to `1.0`, uploading an unsupported vehicle common in the local market (e.g., \*Saipa Pride\*, \*IKCO Samand\*) produces high-confidence false positives due to Neural Collapse projection onto the nearest known sedan/hatchback subspace.

4\. \*\*Low-Light \& Multi-Car Clutter:\*\* User-captured photos include severe nighttime underexposure, multi-vehicle street backgrounds, and partial framing.



\*\*FGVC-IR8\*\* addresses these failure modes through a custom \*\*DINOv2-ViT-B/14-reg4\*\* architecture fused with \*\*LoRA adapters\*\*, \*\*Competitive Part-Slot Attention\*\*, a \*\*5D Scale-Invariant Relational Graph\*\*, \*\*Multi-Instance Set Evidence Pooling (Set-MIL)\*\*, and a training-free \*\*SVD Subspace Open-Set Guard (`OpenSetGuard`)\*\*.



\---



\## 2. System Architecture \& Mathematical Formulation



```text

Raw Input Frames (1..8 images per vehicle)

&#x20; │

&#x20; ├─► \[Stage 1: Photometric \& Occlusion Preprocessor]

&#x20; │     ├─ Rank-1 Separable Annular Inpainter (41x41 \& 81x81 rings, 13x13 morphological margin)

&#x20; │     └─ Multi-Scale Retinex (MSR) Homomorphic Shadow Reflectance Recovery

&#x20; │

&#x20; ├─► \[Stage 2: Single-Pass 224px Evaluator \& OOD-Safe Autonomous Cropper]

&#x20; │     ├─ 2D Connected-Component Flood-Fill + Multi-Peak Local Maxima NMS

&#x20; │     └─ Strict Cosine-Gain Gate (adopts crop iff cos\_crop >= cos\_full + 0.025 \& cos\_crop >= 0.935)

&#x20; │

&#x20; ├─► \[Stage 3: Fused DINOv2-ViT-B/14-reg4 Backbone (12 Blocks, 4 Register Tokens)]

&#x20; │     ├─ LoRA Weight Fusion (W\_fused = W\_0 + (alpha/r) B A, numerical error <= 9.54e-7)

&#x20; │     ├─ Top-Block \[CLS] Spatial Mass Capping (max patch share <= 5.5 / N\_p)

&#x20; │     └─ Hollow-Ring (5x5 \\ 3x3 Donut) Counterfactual Token Outlier Filter

&#x20; │

&#x20; ├─► \[Stage 4: Competitive Slot Attention \& 5D Relational Part Graph]

&#x20; │     ├─ 10 Competitive Slots (8 Foreground Part Slots + 2 Background Garbage Slots)

&#x20; │     ├─ 5D Pairwise Geometry: \[Δx/σ, Δy/σ, d/σ, ln(σ\_j/σ\_i), ln(M\_j/M\_i)]

&#x20; │     ├─ Dual-Layer Relational Transformer + Mass-Weighted Graph Pooling -> 512-D Embedding

&#x20; │     └─ Zero-Copy Confidence-Gated Dual-Head Fusion (Step-35686 Final + Step-28039 Best Head)

&#x20; │

&#x20; └─► \[Stage 5: OpenSetGuard \& Symmetric Cross-Scale Decision Engine]

&#x20;       ├─ SVD Subspace Null-Residual r\_perp(e) + View/Quality-Conditioned Geodesic Cone

&#x20;       ├─ Virtual OOD Logit Matching (ViM probability damping on unrecognized vehicles)

&#x20;       ├─ Multi-Vehicle Conflict Detector (flags mixed uploads across frames)

&#x20;       └─ Symmetric Cross-Scale Consensus (392px + 224px) for Peugeot 206 HB vs. 206 SD

```



\### Key Algorithmic Modules



\#### 2.1. Data Acquisition \& Multi-Modal Curation Pipeline (ETL)

\* \*\*Stateful API Harvesting:\*\* Built a fault-tolerant scraper targeting internal JSON endpoints (`/v8/postlist/w/search` and `/v8/posts-v2/web/post/{token}`) using cursor-based temporal pagination (`last\_post\_date`), gzipped Base64 `viewed\_tokens` state tracking, Gaussian jitter (`1.8s–3.4s`), bounded exponential backoff (`180s` cap), and a Dual-Storage engine (\*\*SQLite WAL\*\* + real-time flushed \*\*JSONL\*\*). Exact hierarchical ground-truth labels were parsed directly from structured `BREADCRUMB` trees.

\* \*\*Zero-Shot Scene Gatekeeping \& Safe Cropping:\*\* Filtered non-exterior noise (dashboards, steering wheels, engine bays, inspection documents) using \*\*CLIP (`ViT-B/32`)\*\* cosine similarity against positive/negative scene prompts, followed by \*\*YOLOv8n\*\* primary vehicle selection ($\\text{Score} = w\_1 \\text{AreaRatio} + w\_2(1 - \\text{CenterDistance}) + w\_3 \\text{Confidence}$) with a `12%` dynamic safety margin to preserve side mirrors, wheels, and bumpers.

\* \*\*Sharded WebDataset Storage:\*\* Compiled into \*\*92 training shards\*\* and \*\*15 validation shards\*\* (\*\*5,098 validation listings / 12,100 frames\*\*) across 8 major Iranian vehicle classes.



\#### 2.2. Shortcut-Invariant Token \& Slot Regularization

To prevent localized artifacts (e.g., painted masks over grille badges or license plates) from shifting predictions:

1\. \*\*Hollow-Ring ($5\\times 5 \\setminus 3\\times 3$) Token Outlier Filter:\*\* Computes spatial context strictly from the 16 patches in the outer $5\\times 5$ ring, excluding the inner $3\\times 3$ core so a $2\\times 2$ emblem anomaly cannot contaminate its own reference mean:

&#x20;  $$M\_{\\text{ring}}(x, y) = \\frac{25 \\cdot \\text{AvgPool}\_{5\\times 5}(x, y) - 9 \\cdot \\text{AvgPool}\_{3\\times 3}(x, y)}{16}$$

2\. \*\*Rank-1 Separable Annular Inpainter:\*\* Detects zero-texture digital brush masks ($\\sigma^2\_{\\text{local}} < 2.5$) and reconstructs them in `\~2.2 ms` on CPU using separable 1D box convolutions ($\\text{AvgPool}\_{K\\times 1} \\circ \\text{AvgPool}\_{1\\times K}$ for $K \\in \\{41, 81\\}$) with a $13\\times 13$ morphological margin.



\#### 2.3. `OpenSetGuard`: SVD Subspace Residual \& Virtual Logit Matching

Inspired by \*\*ViM\*\* (\*Wang et al., CVPR 2022\*) and \*\*NECO\*\* (\*Ammar et al., ICLR 2024\*), `OpenSetGuard` performs Singular Value Decomposition ($W = U \\Sigma V^\\top$) on the $32 \\times 512$ normalized `SubCenterArcFace` weights at initialization to extract the orthonormal basis $V\_{\\mathcal{P}} \\in \\mathbb{R}^{512 \\times 18}$ spanning the 8 in-distribution classes. For any test embedding $e \\in \\mathbb{S}^{511}$:

$$r\_{\\perp}(e) = \\sqrt{1 - \\left\\Vert{} V\_{\\mathcal{P}}^\\top e \\right\\Vert{}\_2^2}, \\quad \\mathcal{C}\_k = \\cos\_{\\text{eff}, k} + \\delta\_{\\text{sib}, k} + 0.035 \\cdot \\text{clip}(f\_{\\text{conc}, k} - 0.42, -0.10, 0.30)$$

Frames failing the photometric-quality-conditioned threshold $\\mathcal{C}\_k < \\tau(q\_k)$ are rejected as `ood`, and display probabilities are damped via a Virtual OOD Logit so unsupported vehicles never exhibit inflated confidence on known classes.



\---



\## 3. Training Progression \& Validation Benchmarks



The model was trained for \*\*35,686 optimizer steps\*\* (14 total epochs across a two-phase resolution curriculum) on \*\*92 training shards\*\* and evaluated on \*\*15 validation shards (5,098 listings, 12,100 frames)\*\*:

\* \*\*Phase 1 (`s224`, Epochs 1–8, Steps `0 .. 20,392`):\*\* Resolution $224 \\times 224$ ($16 \\times 16$ patch grid), batch configuration $P=16 \\times K=4$, learning rate $2.0\\times 10^{-4}$ (LoRA) / $1.0\\times 10^{-3}$ (Head).

\* \*\*Phase 2 (`s392`, Epochs 1–6, Steps `20,392 .. 35,686`):\*\* Resolution $392 \\times 392$ ($28 \\times 28$ patch grid, 784 patches/image) with gradient checkpointing and cosine annealing down to $4.0\\times 10^{-6}$ (LoRA) / $2.0\\times 10^{-5}$ (Head).



\### 3.1. Validation Progression Across Training Phases (5,098 Validation Listings)



| Phase \& Epoch | Global Step | Listing Acc (`acc`) | Balanced Acc (`bal`) | Frame Acc (`frame`) | `206HB -> SD` Error | `206SD -> HB` Error | Checkpoint Tag |

| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |

| `s224` — Epoch 1 | `2,549` | `97.00%` | `97.31%` | `96.22%` | `7.6%` | `7.8%` | `s224-epoch1` |

| `s224` — Epoch 3 | `7,647` | `97.37%` | `97.58%` | `96.69%` | `7.5%` | `7.8%` | `s224-epoch3` |

| `s224` — Epoch 4 | `10,196` | `97.51%` | `97.66%` | `96.59%` | `7.2%` | `7.8%` | `s224-epoch4` |

| `s224` — Epoch 5 | `12,745` | `97.37%` | `97.91%` | `96.41%` | `8.5%` | `3.9%` | `s224-epoch5` |

| `s224` — Epoch 7 | `17,843` | `97.63%` | `97.91%` | `96.96%` | `7.3%` | `6.1%` | `s224-epoch7` |

| `s224` — Epoch 8 | `20,392` | `97.45%` | `97.90%` | `96.71%` | `8.2%` | `5.2%` | `s224-epoch8` |

| `s392` — Epoch 1 | `22,941` | `97.65%` | `97.94%` | `97.28%` | `7.3%` | `6.1%` | `s392-epoch1` |

| \*\*`s392` — Epoch 3\*\* | \*\*`28,039`\*\* | \*\*`97.72%`\*\* | \*\*`97.99%`\*\* | \*\*`97.25%`\*\* | \*\*`7.1%`\*\* | \*\*`6.1%`\*\* | \*\*`best.pt` (`BEST.json`)\*\* |

| \*\*`s392` — Epoch 6\*\* | \*\*`35,686`\*\* | \*\*`97.72%`\*\* | \*\*`97.97%`\*\* | \*\*`97.02%`\*\* | \*\*`7.1%`\*\* | \*\*`6.1%`\*\* | \*\*`final-evaluated`\*\* |



\### 3.2. Per-Class Validation Performance \& Confusion Analysis (5,098 Listings)



| Class ID | Vehicle Class (Make \& Model) | Validation Listings | Exact Class Recall (`Step 28,039` / `Step 35,686`) | Family-Level Identification Accuracy | Primary Error Mode |

| :---: | :--- | :---: | :---: | :---: | :--- |

| \*\*`c0`\*\* | \*\*Peugeot Pars\*\* (`پژو پارس`) | `1,225` | \*\*`99.84%`\*\* (`1,223 / 1,225`) | \*\*`99.84%`\*\* | 2 listings (`1` 206, `1` 207) |

| \*\*`c1`\*\* | \*\*Peugeot 206 HB\*\* (`پژو ۲۰۶ هاچ‌بک`) | `1,148` | \*\*`92.33%`\*\* (`1,060 / 1,148`) | \*\*`99.39%`\*\* (`1,141 / 1,148`) | `81` frontal-only listings (`206HB -> SD`) |

| \*\*`c2`\*\* | \*\*Peugeot 206 SD\*\* (`پژو ۲۰۶ صندوقدار`) | `231` | \*\*`93.94%`\*\* (`217 / 231`) | \*\*`100.00%`\*\* (`231 / 231`) | `14` frontal-only listings (`SD -> HB`), `0` non-206 errors |

| \*\*`c3`\*\* | \*\*Peugeot 207 HB\*\* (`پژو ۲۰۷ هاچ‌بک`) | `628` | \*\*`99.04%`\*\* (`622 / 628`) | \*\*`99.04%`\*\* | Aftermarket 206-to-207 headlight conversions |

| \*\*`c4`\*\* | \*\*IKCO Dena\*\* (`دنا / دنا پلاس`) | `667` | \*\*`99.85%`\*\* (`666 / 667`) | \*\*`99.85%`\*\* | 1 listing |

| \*\*`c5`\*\* | \*\*IKCO Tara\*\* (`تارا`) | `259` | \*\*`99.23%`\*\* (`257 / 259`) | \*\*`99.23%`\*\* | 2 listings (`Dena`) |

| \*\*`c6`\*\* | \*\*Saipa Quick\*\* (`کوییک`) | `661` | \*\*`99.85%`\*\* (`660 / 661`) | \*\*`99.85%`\*\* | 1 listing |

| \*\*`c7`\*\* | \*\*Saipa Shahin\*\* (`شاهین`) | `279` | \*\*`100.00%`\*\* (`279 / 279`) | \*\*`100.00%`\*\* | \*\*0 errors\*\* |

| \*\*Total\*\* | \*\*All 8 Classes\*\* | \*\*`5,098`\*\* | \*\*`97.72%` Raw / `97.99%` Balanced\*\* | \*\*`99.59%` (`5,077 / 5,098`)\*\* | Only 21 cross-family errors across 5,098 listings |



> \*\*Note on Peugeot 206 Frontal Ambiguity Handling:\*\* Across the entire 5,098-listing validation set, \*\*82% of all strict classification errors (95 out of 116 listings)\*\* occur exclusively between `Peugeot 206 HB` and `Peugeot 206 SD` on listings containing only frontal photos where the trunk is physically invisible. When the engine's \*\*Frontal-View Ambiguity Gate\*\* is enabled, decisive rear/side accuracy on the 206 family reaches \*\*>97.0%\*\*, and overall system accuracy across all 5,098 listings exceeds \*\*99.1%\*\*.



\---



\## 4. CPU Inference Latency \& Robustness Benchmarks



Benchmarked locally on a \*\*Microsoft Surface Laptop 3 (Intel Core CPU, 8 GB RAM, Windows 11, No GPU)\*\*:



| Benchmark Test / Operating Mode | Resolution | Latency (CPU) | Numerical / Behavioral Verification |

| :--- | :---: | :---: | :--- |

| \*\*Model Bundle Initialization (`load\_all\_artifacts`)\*\* | — | \*\*`0.73 s – 0.98 s`\*\* | Loads merged FP16 DINOv2 + Dual-Head (`best.pt` + `step\_0035686.pt`) offline |

| \*\*Rank-1 Separable Annular Inpainter (`81x81`)\*\* | Native | \*\*`2.2 ms`\*\* | Exact mathematical equivalence to 2D pool (`max\_diff = 8.94e-8 < 1e-5`) |

| \*\*Fast Mode (`fast\_224`, Single-Pass Cached)\*\* | `224 x 224` | \*\*`\~450 – 550 ms`\*\* | Full 12-block DINOv2 + 8-Slot Graph + Dual-Head + Heatmap generation |

| \*\*Pro Mode (`pro\_392`, Multi-Scale Consensus)\*\* | `392 x 392` + `224 x 224` | \*\*`\~1,100 – 1,350 ms`\*\* | High-resolution 784-patch pass + 224px cross-scale consensus |

| \*\*Adversarial Emblem Occlusion Invariance\*\* | `224 x 224` | — | `pixel MAE = 2.66 / 255`, `cosine drift = 0.0030 < 0.025` |

| \*\*Open-Set OOD Rejection (e.g., \*Saipa Pride\*)\*\* | `392 x 392` | \*\*`\~1,150 ms`\*\* | Rejected (`status = ood`, max bar damped to `< 28%` via ViM virtual logit) |



\---



\## 5. Quick Start (1-Click Local Deployment)



The inference engine and bilingual web interface (`FA` / `EN`) run entirely on Python's standard library HTTP server (`ThreadingHTTPServer`) plus `torch`, `torchvision`, `numpy`, and `Pillow`—requiring \*\*zero heavy web frameworks\*\* and \*\*zero internet access\*\* once weights are present.



\### Option A: 1-Click Windows Launch (`run.bat`)

1\. Clone this repository or download the source zip:

&#x20;  ```bash

&#x20;  git clone \[https://github.com/ahmadZMVD/FGVC-IR8.git](https://github.com/ahmadZMVD/FGVC-IR8.git)

&#x20;  cd FGVC-IR8

&#x20;  ```

2\. Double-click \*\*`run.bat`\*\* (or run `.\\run.bat` in Command Prompt / PowerShell).

&#x20;  \* \*\*First launch:\*\* Automatically verifies Python dependencies, creates an isolated `.venv` if needed, downloads `fgvc\_ir8\_laptop\_standalone\_fp16.pt` (`172.4 MB`) and `best.pt` (`15.4 MB`) directly from \[\*\*GitHub Releases (`v1.0.0`)\*\*](https://github.com/ahmadZMVD/FGVC-IR8/releases/tag/v1.0.0) if not already in `checkpoints/`, and writes `.env\_verified.json`.

&#x20;  \* \*\*Subsequent launches:\*\* Bypasses setup checks and launches the web interface at `http://127.0.0.1:8080` in under \*\*1 second\*\*.



\### Option B: Manual CLI \& Unit-Test Suite Execution

Run the 9-stage modular, adversarial, and open-set diagnostic test suite:

```bash

python engine.py --selftest

```

Expected console output:

```text

==================================================================

\[SELFTEST] FGVC-IR8 v8.2 Lossless Fast-Engine Unit-Test Suite

==================================================================

\[INFO] FGVC-IR8 Engine v8.2 ready in 0.78s | Loaded: fgvc\_ir8\_laptop\_standalone\_fp16.pt, best.pt, ...

\[PASS 1/9] Rank-1 Separable Box Filter Exact Equivalence Verified (max\_diff=8.94e-08 < 1e-5)

\[PASS 2/9] Distributed Slot Attention Mass-Capping Verified (Zero Single-Patch Collapse)

\[PASS 3/9] Hollow-Ring (5x5 \\ 3x3) Token Outlier Filter Verified (2.85x 2x2-block attenuation)

\[PASS 4/9] Screenshot (90) \& (91) Logo-Tampering Invariance + Screenshot (88) 206-SD Lock Verified

\[PASS 5/9] Separable Annular Inpainter Verified in 2.2ms (MAE=2.66, drift=0.0030)

\[PASS 6/9] OpenSetGuard Verified: Saipa Pride (SS 93) -> REJECTED (OOD) | Genuine Cars -> ACCEPTED (ID)

\[PASS 7/9] Multi-Scale Retinex Shadow Recovery Verified (q\_factor=0.73)

\[PASS 8/9] Autonomous Multi-Peak \& Connected-Component Extractor Verified (4 proposals)

\[PASS 9/9] Latency Benchmark: Fast-224 = 528.0 ms | Pro-392 = 1360.8 ms

\[SUCCESS] All 9 modular, open-set, adversarial \& speed unit tests passed with zero errors!

```



Start the local web server directly:

```bash

python app.py

```



\---



\## 6. Repository Structure



```text

FGVC-IR8/

├── run.bat                  # 1-Click Windows launcher (bypasses PowerShell ExecutionPolicy)

├── launcher.py              # Environment bootstrapper, weight downloader \& fast-boot manager

├── engine.py                # Standalone DINOv2-LoRA + 8-Slot Graph + OpenSetGuard inference engine (v8.2)

├── app.py                   # Zero-dependency bilingual (EN/FA) web dashboard \& REST API

├── requirements.txt         # Minimal CPU dependencies (torch, torchvision, numpy, Pillow)

└── checkpoints/

&#x20;   ├── fgvc\_ir8\_laptop\_standalone\_fp16.pt  # Fused standalone bundle (auto-downloaded from Releases)

&#x20;   ├── best.pt                             # Step 28,039 best balanced accuracy head checkpoint

&#x20;   ├── BEST.json                           # Validation metadata for best checkpoint (0.9799)

&#x20;   ├── MANIFEST.json                       # Checkpoint SHA-256 integrity manifest

&#x20;   └── train\_log.txt                       # Complete 35,686-step training \& evaluation telemetry log

```

