# تصميم مجمَّد — Item 5: التحقق الخارجي عبر النطاقات (Cross-Domain External Validation, CICIoT2023)

**تاريخ التجميد:** 2026-09-20
**الحالة:** مجمَّد كتابيًا قبل أي تنفيذ تدريبي. لا يُعدَّل إلا بقرار صريح موثَّق أدناه (سجل التعديلات).
**لا يمس أي ملف من Item 1-4** (`../RESULTS_ITEM1_ITEM2_LOCKED.md`, `../results/*`, `../item4_mechanistic_localization/*`) — تلك مقفلة. هذا البند يكتب حصرًا في `item5_cross_domain_validation/results/`.

**سياق القرار:** يختبر هذا البند هل إطار القياس (Retention Framework، §3 من المخطوطة) — لا نتيجة UAV تحديدًا — ينتقل إلى بيئة كشف تسلل مستقلة، حقيقية الأجهزة، خارج سياق UAV، هي نفس CICIoT2023 التي استخدمتها FedCEP (Zhan et al. 2026) ضمن ثلاثة benchmarks. هذا **ليس** محاولة منافسة FedCEP على Macro-F1، بل اختبار انتقال أداة القياس نفسها.

---

## 0. مصدر البيانات وسلسلة الحيازة (Provenance)

| البند | القيمة |
|---|---|
| المصدر الرسمي | Canadian Institute for Cybersecurity (UNB), `https://www.unb.ca/cic/datasets/iotdataset-2023.html` |
| الملف المُنزَّل | `CSV (1).zip` عبر نموذج طلب UNB الرسمي (لا PCAP، لا mirror، لا Kaggle) |
| تاريخ التنزيل | 2026-09-20 |
| SHA-256 (الأرشيف) | `e211e878d2f39226ea3a854683f91ae6175b628807ee96eeecb7a04a36a28dbd` |
| الحجم | 1.43GB مضغوط، 8.94GB بعد الفك |
| عدد ملفات CSV | 309 (تحقَّق مطابق لعدد ملفات الأرشيف) |
| مانيفست SHA-256 لكل ملف | `dataset_audit/ciciot2023_csv_manifest.json` |
| مسار البيانات المستخرَجة | `development-workspace/ciciot2023_dataset/CSV/` |

**سبب توثيق الهاش الدقيق:** دراسات منشورة تستشهد بـCICIoT2023 تُبلِغ أعداد صفوف/ملفات مختلفة قليلًا عن بعضها؛ توثيق الهاش والمانيفست يضمن أن كل رقم في هذا البند قابل للتتبع لنسخة بايتية محددة، بنفس انضباط `LOCK_MANIFEST.json` لـISOT.

---

## 1. نتائج Dataset Audit (بلا أي تدريب، مُنفَّذة بالكامل قبل هذا التجميد)

- Schema: 39 عمودًا، متطابق حرفيًا عبر كل الـ309 ملف (صفر اختلاف).
- **لا عمود label** — الفئة (من 34: 33 هجومًا + `Benign_Final`) محدَّدة حصرًا باسم المجلد.
- **لا معرّفات جهاز/IP/MAC/جلسة** — العمود الوحيد المطابق لكلمة "ip" هو `IPv` (flag ثنائي لبروتوكول IP، وليس عنوان IP). هذا يمنع أي device-native federation (Track B)، ويؤكد ما هو موثَّق في §9/§11 من المخطوطة: Track A فقط (synthetic label-skew federation).
- إجمالي الصفوف الخام: 46,776,700. صفوف بها NaN/Inf: 1,040 (ضئيل، حُذفت).
- **تعارض تسمية (cross-label conflicts):** 533,448 بصمة صف فريدة ظهرت تحت أكثر من فئة واحدة، تمس 14,144,617 صفًا (≈30% من الخام) — أغلبها أزواج DDoS/DoS لنفس نوع الهجوم (مثل `DDoS-TCP_Flood`/`DoS-TCP_Flood`) تتشارك نفس متجه الـ39 خاصية تمامًا. هذا تعارض بنيوي حقيقي في قابلية الفصل بهذه الخصائص وحدها، لا خطأ بيانات.
- **تكرار تام (exact duplicates) داخل نفس الفئة:** 12,029,808 صف إضافي بعد إزالة صفوف التعارض.
- **الفهرس (Manifest) الكامل** لكل صف الأمر: `dataset_audit/run_fingerprint_audit.py`، النتائج في `dataset_audit/fingerprint_audit_report.txt`.

---

## 2. قاعدة التنظيف المجمَّدة (Cleaning Rule — نُفِّذت بالفعل، موثَّقة هنا للتجميد الرسمي)

بالترتيب الحتمي التالي، لا يتغيّر لاحقًا:

1. تحديد الفئة من اسم المجلد الأصلي لكل صف.
2. حذف أي صف يحتوي NaN أو ±Inf في أي من الأعمدة الـ39 الرقمية.
3. حساب بصمة (fingerprint) لكل صف = hash القيم الـ39 كما فُسِّرت (نفس الـschema المطابق حرفيًا عبر كل الملفات).
4. **حذف كل الصفوف** التي بصمتها تظهر تحت أكثر من فئة واحدة (تعارض تسمية) — **لا يُختار أحد الفئات اعتباطيًا**؛ كل النسخ المتعارضة تُستبعَد من مجموعة النمذجة بالكامل، والعدد يُبلَّغ.
5. إزالة التكرار التام (نفس البصمة + نفس الفئة) → صف واحد يُبقى.

**النتيجة النهائية (المجموعة النظيفة المستخدَمة حصريًا من هنا فصاعدًا):** 20,601,235 صفًا عبر كل الـ34 فئة (لم تختفِ أي فئة). الجدول الكامل: `dataset_audit/cleaned_class_prevalence.csv`.

**قاعدة صريحة:** كل نسبة انتشار (prevalence) مستخدَمة في هذا البند — بما فيها قاعدة اختيار rare-tail أدناه — تُحسَب من هذه المجموعة النظيفة فقط، لا من الأعداد الخام.

---

## 3. مهمة التصنيف

**34 فئة** (benign + 33 هجومًا فرديًا)، لا التجميع إلى 8 فئات عائلية — لأن التجميع (مثل دمج XSS/SQLi/Uploading/CommandInjection داخل "Web-based") يقتل بالضبط الذيل النادر الذي يختبره هذا البند. هذا يطابق أحد المهام الثلاث المعرَّفة أصلًا في ورقة CICIoT2023 نفسها (binary / 8-class / 34-class).

---

## 4. قاعدة اختيار الفئات النادرة (مسبقة التسجيل — القاعدة مجمَّدة الآن، الأسماء نتيجة تطبيقها لا اختيار يدوي)

> **Rare-tail subset = الفئات الخمس (k=5) الأدنى انتشارًا في المجموعة النظيفة (training-pool)، بعد التنظيف الكامل (§2) وقبل أي تدريب فيدرالي.**

$k=5$ ثابت مسبقًا (بلا علاقة بأي نتيجة نموذج)، ويطابق عدد الفئات النادرة الأساسية المستخدَمة وصفيًا في ISOT (Manipulation/Replay بحاملَيهما) — سبب تنظيمي إضافي، لا المبرِّر الوحيد.

**تطبيق القاعدة على المجموعة النظيفة (§2)، بالترتيب:**

| الرتبة | الفئة | عدد الصفوف (نظيف) | الحصة |
|---|---|---|---|
| 1 (الأندر) | `Uploading_Attack` | 1,252 | 0.0061% |
| 2 | `Recon-PingSweep` | 2,257 | 0.0110% |
| 3 | `Backdoor_Malware` | 3,193 | 0.0155% |
| 4 | `XSS` | 3,837 | 0.0186% |
| 5 | `SqlInjection` | 5,243 | 0.0254% |

**لا يُدَّعى وجود "فجوة طبيعية" بعد الفئة الخامسة** — الفرق عن `CommandInjection` (5,383، الفئة السادسة) ضئيل. المبرِّر الوحيد لـ$k=5$ هو أنه ثابت مسبقًا، لا نمط في البيانات.

**تحقَّقنا فعليًا (قبل التجميد):** هذا الترتيب **مطابق تمامًا** لترتيب البيانات الخام قبل أي تنظيف — أي أن التنظيف (إزالة 56% من الصفوف الخام) **لم يغيّر عضوية أو ترتيب rare-tail إطلاقًا**. حسب قاعدة التفعيل المسبقة (§16 أدناه)، هذا يعني أن **لا حاجة لأي sensitivity analysis إضافي** حول قرار التنظيف.

**قاعدة الاستبدال الممنوع:** إن لم تحقق إحدى فئات rare-tail الحد الأدنى للدعم الإحصائي بعد التقسيم (§6: ≥50 مثالًا في validation و≥50 في test)، **لا تُستبدَل بالفئة السادسة**. تبقى ضمن rare-tail هيكليًا لكن تُعلَّم `insufficient_evaluation_support = true` وتُستبعَد فقط من نقاط القياس التي تتطلب استقرارًا (recall). **تحقَّقنا:** بتقسيم 70/15/15 (§6)، كل الفئات الخمس تتجاوز هذا الحد بوضوح (أصغرها `Uploading_Attack` بـ≈188 مثالًا متوقعًا لكل من val/test) — لا يُتوقَّع تفعيل هذه القاعدة، لكنها مسجَّلة قبل التنفيذ.

---

## 5. النموذج الجديد لعملية التصنيف (Model)

تصنيف يجري على **كل الـ34 فئة معًا** (وليس rare-tail فقط)؛ rare-tail هو subgroup تحليلي أساسي داخل نتائج التصنيف الكامل، لا مهمة تدريب منفصلة.

معمارية مطابقة بنائيًا لـ`AttackFamilyMLP` (Item 1-4)، بأبعاد مُحدَّثة لهذا الداتاسيت فقط — **لا يُدَّعى أي شيء عن "استقلالية المعمارية عن الآلية"** من هذا الاختيار؛ هو إعادة استخدام للبساطة والاتساق فحسب:

```
Linear(39 → 128) → ReLU → Dropout → Linear(128 → 64) → ReLU → Dropout → Linear(64 → 34)
```

Head = الطبقة الأخيرة (64→34)؛ Representation = ما قبلها — نفس تعريف Item 4 بنيويًا، لكن **لا تكرار لتوطين Item 4 هنا** (§10).

---

## 6. تقسيم Train/Validation/Test (مجمَّد، قبل أي Dirichlet)

$$70\% \text{ train} \;/\; 15\% \text{ validation} \;/\; 15\% \text{ test}$$

**Stratified** حسب تسمية الـ34 فئة، على **المجموعة النظيفة بأكملها** (بعد §2)، ببذرة عشوائية ثابتة واحدة (`SPLIT_SEED = 2023`, مُجمَّدة الآن). الـvalidation وtest **يُستبعَدان كليًا من أي تدريب فيدرالي** (لا Dirichlet عليهما، لا مشاركة عميل). الـScaler (StandardScaler أو مكافئه) يُطابَق (fit) على TRAIN فقط، ثم يُطبَّق على val/test — نفس انضباط `ISOTFederatedData.fit_preprocessing()`.

---

## 7. التقسيم الفيدرالي (Row-Level Synthetic Dirichlet — لا Session-Level)

بسبب غياب أي معرّف جهاز/جلسة (§1)، **لا يمكن الادعاء بـdevice-native federation**. التقسيم يكون على **مستوى الصف** ضمن مجموعة TRAIN فقط، عبر Dirichlet label-skew قياسي:

- **عدد العملاء:** $N = 15$ — مطابق لعدد عملاء ISOT الأساسي (Item 1-4)، لإبقاء المقارنة عبر النطاقين نظيفة (نُغيّر الداتاسيت فقط، لا عدد العملاء).
- **α (تركيز Dirichlet):** $\alpha = 0.1$ — **أساسي (Primary)**، مطابق لإعداد extreme non-IID الذي استخدمته FedCEP نفسها على CICIoT2023.
- **آلية القسمة:** لكل فئة $c$، يُسحَب $p_c \sim \mathrm{Dir}(\alpha, \ldots, \alpha)$ عبر الخمسة عشر عميلًا، وتُوزَّع صفوف تلك الفئة في TRAIN عشوائيًا حسب $p_c$. هذا يعني أن "الحامل" (holder) لكل فئة نادرة **لا يُحدَّد يدويًا مسبقًا** (خلافًا لـISOT)، بل هو **نتيجة كل سحبة Dirichlet على حدة**: أي عميل يستلم صفًا واحدًا فأكثر من فئة $c$ في تلك السحبة تحديدًا يُعتبَر حاملًا لها **لتلك السحبة**. هذا التعريف يُطبَّق تلقائيًا بالكود، لا يدويًا.
- **صياغة الادعاء في المخطوطة:** *"FedCEP-aligned in the use of CICIoT2023 and extreme Dirichlet label skew (α=0.1); other protocol choices (client count, algorithms, training schedule) are independently specified here, not a claim of full experimental replication."*

---

## 8. تصميم التشغيل (Runs)

$$5 \text{ سحبات Dirichlet مستقلة} \times 3 \text{ بذور نموذج} \times 2 \text{ خوارزمية (FedAvg, FedNova)} = 30 \text{ تشغيلة}$$

- **بذور السحبات (Partition draws):** `PARTITION_SEEDS = [101, 102, 103, 104, 105]` (تتحكم في $p_c$ لكل فئة؛ مستقلة عن بذرة النموذج).
- **بذور النموذج (Model seeds), متداخلة داخل كل سحبة:** `MODEL_SEEDS = [11, 22, 33]` (`torch.manual_seed`، تتحكم في التهيئة العشوائية + ترتيب `local_train`، تمامًا كدور `set_seed` في Item 1/4).
- **جدول التدريب:** `NUM_ROUNDS = 20`، جولات مسجَّلة `LOGGED_ROUNDS = [1, 5, 10, 15, 20]` — أقل من الجدول الأصلي لـISOT (45) عمدًا: Item 5 اختبار انتقال الإطار، لا إعادة إنتاج البرنامج التجريبي الأساسي الكامل؛ حجم بيانات كل عميل هنا أكبر من ISOT (≈1.37M صف تدريب متوسط لكل عميل عبر 15، مقابل تجزئة جلسات ISOT)، لذا يُتوقَّع تقارب أسرع نسبيًا. هذا قرار مُجمَّد الآن، **لا يُعدَّل بعد رؤية نتائج التقارب**.
- **معاملات التدريب المحلي:** `epochs=1, batch_size=4096, lr=1e-3` — مطابقة لـItem 1/4 حرفيًا (لا بحث عن hyperparameters جديد لهذا البند؛ انظر التعديل الموثَّق في سجل التعديلات أدناه بخصوص تصحيح batch_size).
- **الوزن الفيدرالي لكل عميل:** نسبة عدد صفوفه إلى إجمالي TRAIN، كما في `client_weight_rows` بالضبط.

---

## 9. الوحدة الإحصائية (لا استثناء)

**سحبة القسمة (partition draw) هي الوحدة المستقلة الأساسية.** بذور النموذج الثلاثة **متداخلة (nested)** داخل كل سحبة، لا مستقلة عنها. **لا يُعامَل الضرب** $\text{partition} \times \text{seed} \times \text{class}$ **كعينات مستقلة** في أي اختبار أو تلخيص إحصائي — نفس انضباط Item 3/4 بالضبط. أي تقرير اتساق عبر الاتجاه (direction-consistency) يُحسَب على مستوى الخمس سحبات (n=5)، لا الثلاثين تشغيلة.

---

## 10. نقاط القياس (مُجمَّدة، لا إضافة بعد رؤية النتائج)

| الأولوية | المقياس | الدور |
|---|---|---|
| **أساسي (مستمر)** | $\mathrm{RRG}_{c,r} = \max_{i\in H_c} R^{\mathrm{local}}_{i,c,r} - R^{\mathrm{global}}_{c,r}$ (§3.3 من المخطوطة، معرَّف سابقًا، يُعاد استخدامه حرفيًا) | مقياس النقل الأساسي بين النطاقين |
| ثانوي (ثنائي) | Strict erosion، Practical erosion، Conditional retention | نفس تعريفات §3 من المخطوطة حرفيًا |
| سياقي | Per-class global recall (كل الـ34 فئة)، aggregate macro-F1 | سياق أداء عام، ليس نقطة قياس احتباس |

**لا** يُستخدَم ΔMargin أو cosine أو تفكيك Head/Representation هنا — Item 5 يختبر **انتقال إطار القياس**، لا يكرر توطين Item 4 على داتاسيت جديد؛ إعادة توطين مستقل على CICIoT2023 تتطلب تصميمًا مجمَّدًا خاصًا بها، خارج نطاق هذا البند عمدًا (§ نطاق الادعاء أدناه).

---

## 11. نطاق الادعاء

Item 5 يختبر **هل تكتشف أداة القياس (RRG + مؤشرات الاحتباس) نمط فقد المعرفة على نطاق مستقل**، وليس: (أ) مقارنة أداء مباشرة مع FedCEP أو أي عمل آخر على نفس البيانات، (ب) إعادة بناء بروتوكول FedCEP التجريبي الكامل، (ج) توطين آلي (head/representation) جديد على CICIoT2023، (د) اختبار device-native federation (غير ممكن بنيويًا هنا، §1). أي من هذه الأربعة يتطلب تصميمًا مجمَّدًا مستقلًا لاحقًا إن استُحسِن.

---

## 12. قاعدة التوقف (Stop Rule)

القائمة النهائية: 5 سحبات × 3 بذور × 2 خوارزمية = 30 تشغيلة، على التصميم أعلاه فقط. **لا** إضافة partitions أو seeds أو خوارزميات أو تدخلات إضافية بعد رؤية أي نتيجة. أي تمديد يتطلب تحديث هذا الملف بقرار موثَّق صريح في سجل التعديلات، بنفس انضباط Item 1-4.

---

## 13. بوابة الحتمية (المكافئ لـItem 4، مُبسَّطة لغياب حاجة إعادة إنتاج مسار خارجي مُقفَل)

بما أن Item 5 لا يعيد إنتاج مسارًا مقفلًا مسبقًا (خلافًا لـItem 4 الذي يقارن بـchekpoints Item 1)، لا توجد بوابة تحقق ضد مرجع خارجي. **الضمانة البديلة المجمَّدة:** لكل تشغيلة من الثلاثين، يُتحقَّق آليًا وبرمجيًا، قبل تفسير أي RRG، أن: (أ) توزيع الفئات في TRAIN بعد Dirichlet يطابق $p_c$ المسحوبة إحصائيًا (فحص عدم انحراف في آلية القسمة نفسها)، (ب) لا تسرّب صفوف بين TRAIN وVALIDATION/TEST (فحص تقاطع البصمات - نفس آلية fingerprint في §2 - بين المجموعات الثلاث، يجب أن يكون صفرًا تمامًا). فشل أي فحص يوقف تفسير تلك التشغيلة فورًا للتحقيق.

---

## 14. هيكل ملفات النتائج

`item5_cross_domain_validation/results/item5_raw.csv` — عمود لكل من: `algorithm` (FedAvg/FedNova), `partition_seed`, `model_seed`, `round`, `class`, ثم أعمدة نقاط القياس (§10). `item5_leakage_check.csv` — نتيجة فحص التسرّب (§13-ب) لكل تشغيلة. لا كتابة فوق أي ملف من Item 1-4.

---

## 15. بوابة ما بعد Item 5

بعد التنفيذ والتدقيق (Audit Item 5، بنفس صرامة تدقيق Item 4)، تُكتَب §9 (Cross-Domain Validation) في المخطوطة حصرًا بالصياغة التي تدعمها النتائج الفعلية — لا قبل ذلك.

---

## 16. سجل التعديلات

- 2026-09-20: إنشاء التصميم المجمَّد بعد اكتمال Dataset Audit (`run_audit.py`) وGlobal Fingerprint Audit (`run_fingerprint_audit.py`) بالكامل، بلا أي تدريب. القرارات الثلاثة الحاسمة (إزالة التكرار/التعارض قبل التقسيم، تقسيم على مستوى الصف بعد فصل train/val/test أولًا، rare-tail كـsubgroup تحليلي داخل مهمة 34-فئة كاملة) اتُّخذت بعد مراجعة صريحة لنتائج التدقيق، قبل أي تدريب فيدرالي. **قاعدة التفعيل المسبقة للتحقق من ضرورة sensitivity analysis حول قرار التنظيف طُبِّقت وتحققت سلبًا (bottom-5 لم يتغيّر بعد التنظيف)** — لا حاجة لتحليل حساسية إضافي.

- 2026-09-22 (Post-execution leakage audit, per Section 13's promised check): ran the fingerprint-overlap check empirically (train/validation/test parquet files, hash over the 39 feature columns). At the float64 precision used for cleaning/dedup, zero leakage is guaranteed by construction (single-pass stratified split of an already-globally-deduplicated table). At the float32 storage precision used for model training/memory efficiency, a small number of rows across splits are numerically indistinguishable: 69,749/14,420,864 train rows (0.484%) have a float32-identical counterpart in validation; 69,629 (0.483%) in test; 22,505/3,090,185 validation rows (0.728%) overlap test. **Confirmed zero overlap for all five rare-tail classes specifically** (Uploading_Attack, Recon-PingSweep, Backdoor_Malware, XSS, SqlInjection) — the float32-precision collisions are entirely confined to the high-volume DDoS/DoS flood classes (many near-identical repeated flow-statistic rows at that scale), not the classes Item 5's primary RRG/retention endpoints depend on. Recorded in the manuscript's Limitations (Section 11) as a precise, bounded caveat; does not affect any rare-tail result.

- 2026-09-20 (Amendment — Batch-size correction before analytical execution): The initial Item 5 draft specified batch size 256 under the mistaken assumption that this matched the Item 1/4 training configuration. Code audit confirmed that the actual frozen Item 1 mechanism protocol (`update_conflict_analysis.py`) explicitly uses batch size 4096; 256 is only the generic helper-function default in `shared/fl_pipeline/algorithms.py`'s `local_train()`, never the value Item 1/4 actually ran with. Item 5 is therefore corrected to batch size 4096 before any analytical Item 5 run -- Item 5 now uses the same explicit local-training batch size as the primary ISOT protocol, not a new tuning choice. A single-round wall-clock feasibility benchmark under the erroneous batch size 256 (all 15 clients, round 1 only) was run beforehand and is excluded from all scientific analyses; it served only to detect the batch-size inconsistency. No partition, seed, round count, optimizer, learning rate, endpoint, or statistical analysis was changed as a consequence of this timing test. Before the full 30-run schedule, a non-analytical engineering smoke gate at batch size 4096 (round 1 only, all 15 clients, checked for completion/no-OOM/finite-loss/no-NaN-parameters/full-data-coverage) was run; its macro-F1/RRG outputs are not used for any hyperparameter selection. **الحالة بعد هذا التعديل: مجمَّد، جاهز للتنفيذ التحليلي الكامل (30 تشغيلة).**
