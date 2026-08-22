# خطة تحليل مجمَّدة — Item 3: مرحلة التأكيد (Confirmation)

**تاريخ التجميد:** 2026-08-22، **قبل** فتح `results/item3_confirmation_mechanistic_round45.csv` (2,412 صف) بالتفصيل الميكانيكي. البيانات المصدر مُشغَّلة ومقفلة سلفًا (29 تشغيلة مكتملة، صفر لمسة تدريب بعد هذا التجميد). لا تُعدَّل هذه التعريفات بعد فتح النتائج.

## تصحيح مصطلحات (يُلزم كل مخرَج لاحق)

مرحلة التأكيد = **30 تشغيلة مخطَّطة** على 3 تقسيمات مرساة ثابتة (α=0.1/عملاء=30/partition_seed=102، α=0.3/عملاء=15/partition_seed=103، α=1.0/عملاء=10/partition_seed=101):
- **24 تشغيلة جديدة** = 3 تقسيمات × 2 خوارزمية (FedAvg-SGD, FedNova-SGD) × **4 model_seeds جديدة** (22, 33, 44, 55). هذه **أربع بذور جديدة لكل تقسيم×خوارزمية**، **وليست 24 بذرة جديدة منفصلة** — التصحيح اللغوي مُثبَّت هنا نهائيًا.
- **6 تشغيلات إعادة تشخيصية** = 3 تقسيمات × 2 خوارزمية بـ model_seed=11 (نفس بذرة الاستكشاف)، غرضها الوحيد إرفاق تشخيص Local-to-Global المفقود من الاستكشاف.
- **الناتج الفعلي: 29/30 مكتملة.** التشغيلة المفقودة: `fedavg_sgd, α=1.0, n_clients=10, partition_seed=101, model_seed=33` — تباعدت (diverged) عند الجولة 16. لا تُعوَّض بصفر أداء ولا بأي قيمة أخرى؛ تُستبعَد من كل مقام (denominator) وتُذكَر صراحة أينما ظهر عدد التشغيلات.

## وحدة التحليل الأساسية

**ليست** الصفوف الميكانيكية الـ2,412 (كل صف = فئة×عميل-حامل×eval_data، مترابطة داخليًا لأنها تشترك نفس التقسيم والتشغيل والنموذج العالمي). وحدة التحليل هي:

\[ (\text{partition\_seed},\ \text{algorithm},\ \text{model\_seed},\ \text{class}) \]

أي **29 تشغيلة × 10 فئات = 290 وحدة كحد أقصى** (تحقَّق: كل الـ29 تشغيلة سجَّلت جميع الفئات العشر بـ n_holders_this_class ≥ 1 — لا فئة صفرية الحاملين في هذه البيانات تحديدًا؛ لو ظهرت مستقبلًا فئة بلا حاملين تُستبعَد من مقام معدل المحو صراحة). **تُجمَّع صفوف الحاملين (holders) داخل كل وحدة قبل أي إحصاء أو ارتباط** — لا يُعامل الحامل الفردي أو eval_data كملاحظة مستقلة في أي اختبار إحصائي.

## مصدر الأعمدة (توثيق مطابقة الأعمدة الفعلية في CSV)

`item3_confirmation_mechanistic_round45.csv` يحتوي صفَّين لكل (فئة، عميل-حامل): `eval_data ∈ {own_client_rows, validation}`.
- **`own_client_rows`**: أداء/margin **محلي** على بيانات العميل نفسه — يتفاوت `recall_pre_local`/`recall_post_local`/`logit_margin_*` حسب الحامل. **هذا مصدر كل مقاييس "local" أدناه.**
- **`validation`**: أداء/margin **عالمي** على مجموعة Validation الثابتة — `recall_pre_local`/`recall_post_agg`/`logit_margin_pre_local`/`logit_margin_post_agg` **ثابتة عبر كل حاملي نفس الفئة×التشغيلة** (نفس النموذج العالمي يُقيَّم لكل حامل بنفس الطريقة) — يُؤخَذ **القيمة الفريدة الواحدة** لكل (فئة×تشغيلة)، لا المتوسط أو التكرار. **هذا مصدر `global_recall` وmargin العالمي.**

## تعريفات Local-to-Global (مُجمَّدة، بالصيغة والمصدر العمودي الدقيق)

لكل وحدة (partition, algorithm, model_seed, class)، وبأوزان `client_fedavg_weight` للحاملين (مُعاد توزينها بين حاملي هذه الفئة فقط):

- **`max_local_recall`** = max( `recall_post_local` ) عبر حاملي الفئة، من `own_client_rows`.
- **`weighted_local_recall`** = Σ(`client_fedavg_weight_i` × `recall_post_local_i`) / Σ(`client_fedavg_weight_i`) عبر حاملي الفئة، من `own_client_rows`.
- **`global_recall`** = `recall_post_agg` من صف `validation` الفريد لهذه الفئة×التشغيلة.
- **`local_learning_gain`** = weighted_local_recall(post_local) − weighted_local_recall(pre_local)، كلاهما من `own_client_rows` وبنفس ترجيح الحاملين (اختيار تحليلي مُسبَّق التسجيل: نستخدم النسخة الموزونة لا القصوى، لتفادي حساسية Gain لحامل شاذ واحد).
- **`aggregation_recall_loss`** = max_local_recall − global_recall.
- **`aggregation_margin_change`** = (`logit_margin_post_agg` من `validation`) − max(`logit_margin_post_local` عبر الحاملين، من `own_client_rows`).
- **حدث المحو الصارم (strict erosion)** = 𝟙[ max_local_recall > 0 ∧ global_recall = 0 ].
- **حدث المحو العملي (practical erosion، عتبة حساسية)** = 𝟙[ max_local_recall ≥ 0.05 ∧ global_recall = 0 ] — لتفادي معاملة استدعاء محلي ضئيل جدًا (مثل 0.001) كمعرفة محلية "قوية" ثم وصف فقدانها بأنه محو.

كلا مؤشري المحو يُبلَّغان **معًا دائمًا**، جنبًا إلى جنب، لا أحدهما بديلًا عن الآخر.

## الإجراء الإحصائي المُجمَّد

لكل **نقطة مرساة (anchor) × خوارزمية**:
1. **قائمة نتائج كل model_seed منفردة** (لا تُخفى خلف متوسط فقط).
2. **متوسط ± SD و95% t-CI** عبر الـmodel_seeds المتاحة لتلك الخلية (n=5 عادة، n=4 في حالة FedAvg بالمرساة الخفيفة — انظر أدناه).
3. **عدد حالات المحو (صارم وعملي) من أصل الفئات القابلة للتقييم** لتلك الخلية — بصيغة "k من N"، ليس نسبة مئوية مجردة بلا مقام.
4. **median و IQR** لـ `aggregation_recall_loss` و `aggregation_margin_change`.
5. **مقارنة FedAvg مقابل FedNova مقترنة (paired)** على مفتاح (partition_seed, model_seed, class) — **فقط على الـmodel_seeds المشتركة بين الخوارزميتين لتلك المرساة**.
6. **معدل المحو يُحسَب لكل model_seed أولًا** (نسبة الفئات المُمحُوَّة من أصل 10 لذلك الـseed×algorithm×anchor)، **ثم** تُقارَن/تُلخَّص هذه المعدلات عبر الـseeds. **لا تُعامل الفئات أو الحاملين كملاحظات مستقلة** في أي اختبار أو فاصل ثقة — الوحدة الإحصائية للمقارنة بين الخوارزميات أو المراسي هي الـseed (التشغيلة)، لا الفئة المفردة.

### الحالة الخفيفة (α=1.0, n_clients=10) — قيد صريح

- **FedAvg لديها 4 تشغيلات ناجحة فقط** من أصل 5 المخطَّطة (model_seed=33 تباعد عند الجولة 16). **يُعرَض `1/5 = 20% divergence` داخل هذه النقطة صراحة** في كل جدول يخصها.
- **كل رقم عددي لهذه الخلية (FedAvg، α=1.0، عملاء=10) conditional on four successful runs** — تُذكَر هذه العبارة حرفيًا في تعليق كل جدول/شكل يعرضها.
- **المقارنة المقترنة مع FedNova في هذه المرساة تستخدم البذور المشتركة الأربع فقط** (11, 22, 44, 55) — بذرة FedNova رقم 33 (ناجحة عند FedNova) **تُستبعَد من هذه المقارنة المقترنة تحديدًا** لعدم وجود نظير FedAvg لها، وإن ظهرت في الجداول غير المقترنة لـFedNova وحدها.
- **لا تعويض للتشغيلة المنهارة بصفر أداء، ولا إدخال أي تشخيص من الجولة 45 لها** — لأنها لم تصل للجولة 45 أصلًا (تباعدت عند 16)، فلا وجود لصف ميكانيكي لها في CSV أصلًا؛ هذا يُوثَّق كحقيقة بيانات لا كقرار تحليلي بديل.

### معدلات الانهيار على مستوى مرحلة التأكيد كاملة

- **FedAvg-SGD: 1/15 = 6.7%** (تشغيلة واحدة من 15 مخطَّطة عبر المراسي الثلاث).
- **FedNova-SGD: 0/15.**
- **يُبرَز صراحة** أن كامل فشل FedAvg في مرحلة التأكيد وقع **داخل المرساة الخفيفة تحديدًا (1/5 هناك)** — لا فشل في المرساتين الأخريين (α=0.1 وα=0.3)، فلا يُعمَّم "معدل انهيار FedAvg = 6.7%" كخاصية ثابتة عبر كل نظام Non-IID دون ربطه بأنه محصور بالنظام الأقرب لـIID تحديدًا.

## احتراز مُلزم بشأن نتيجة HHI/تركيز الفئة والإنقاذ (Rescue)

فجوة HHI بين حالات "أنقذتها FedNova" وحالات "بقيت صفرية عند الخوارزميتين" ملحوظة وصفيًا، لكن **وحدات الملاحظة (فئة×تقسيم) مترابطة داخل نفس partition_id** (10 فئات تتشارك نفس بنية التقسيم). **قبل وصف هذه الفجوة بأنها نتيجة "صامدة" (robust)، يُطبَّق cluster-bootstrap بإعادة أخذ العينات على مستوى `partition_id` (وليس على مستوى الفئة المفردة)**، وتُبلَّغ فاصل الثقة الناتج من الـbootstrap جنبًا إلى جنب مع الفرق الوصفي الخام.

**الصياغة اللفظية المعتمدة الوحيدة لهذه النتيجة، في أي مكان بالتقرير أو الشكل التوضيحي:**

> "FedNova rescue was descriptively concentrated in moderately concentrated class allocations, while highly concentrated cases were rarely rescued."

**لا يُسمَّى Gini أو holder_fraction أو HHI أو holder_weight_share "سببًا" (cause) لأي نتيجة في أي مكان.** توصف هذه المتغيرات فقط بأنها **"أقوى الارتباطات المستقرة في التحليل الاستكشافي"** (strongest stable exploratory associations) — لا أكثر.

## تحقق سلامة مُلزَم (يُذكَر كنتيجة سلامة، لا كنتيجة علمية جديدة)

**تطابق model_seed=11 بين مرحلة التأكيد والاستكشاف: 6/6 حالات مطابقة تمامًا** (فرق val_macro_f1 عند الجولة 45 = 0.0 في كل الحالات الست: كل تركيبة algorithm×anchor). هذا **يثبت حتمية التشغيل (run determinism)** تحت البيئة والإعدادات المثبَّتة لهذه الحالات — نتيجة سلامة ممتازة تُذكَر في قسم منهجية/تحقق البيانات بالمخطوطة، لا كادعاء علمي عن الظاهرة قيد الدراسة.

## المخرجات المطلوبة (8 مخرجات، أسماء الملفات مُجمَّدة)

1. `confirmation_table1_performance_by_anchor_algo_seed.csv` — جدول الأداء حسب المرساة والخوارزمية والبذرة.
2. `confirmation_table2_erosion_rates_strict_practical.csv` — معدلات المحو الصارم والعملي.
3. `confirmation_table3_recall_margin_pipeline.csv` — recall/margin عبر pre-local → post-local → post-aggregation.
4. `confirmation_table4_paired_fedavg_vs_fednova.csv` — المقارنة المقترنة.
5. `confirmation_table5_erosion_vs_concentration.csv` — علاقة المحو بـ Gini، holder_fraction، holder_weight_share، HHI (+ ملحق cluster-bootstrap لفجوة الإنقاذ).
6. `confirmation_fig_heatmap_class_x_anchor.png` — Heatmap للفئات × النقاط الثلاث.
7. `confirmation_fig_local_to_global_path.png` — مسار المحلي إلى العالمي (pre-local → post-local → post-agg) لكل فئة.
8. `confirmation_table6_divergence_report.csv` — تقرير الانهيار، منفصل تمامًا عن جداول الأداء (لا يُدمَج).

**كل المخرجات تُكتَب في `item3_sensitivity/results/`. لا تدريب جديد، لا لمس لـ`held_out_test`، لا تعديل على هذا الملف بعد بدء التحليل الميكانيكي.**

## سجل التغييرات
- 2026-08-22: تجميد أولي، قبل فتح `item3_confirmation_mechanistic_round45.csv` تفصيليًا. تحقق سلامة seed-11 (6/6) وتفاصيل التشغيلة المتباعدة (fedavg_sgd/α=1.0/nc10/ms33 عند الجولة 16) مُثبَّتان أعلاه بعد فحص أعمدة CSV والانهيار فقط — لا فتح لنتائج ميكانيكية تفصيلية بعد.

## تصحيح ما بعد المراجعة الثانية (2026-08-22، بعد فتح النتائج الأولى من `analyze_confirmation.py`)

**لا تعديل على التعريفات الأصلية أعلاه — هذا قسم إضافي فقط، يوثّق خللين منهجيين اكتُشفا بعد أول تشغيل للتحليل وقبل اعتماد أي استنتاج نهائي لـItem 3.**

### الخلل 1 — فئة Password Cracking غير قابلة للتقييم على Validation

`Password Cracking` لديها **صفر صف** في مجموعة Validation المُثبَّتة (تحقُّق مباشر من البيانات: `np.bincount` على `yval` = 0 لهذه الفئة). دالة `all_class_predictions_and_margins()` في `run_confirmation.py` تستدعي `precision_recall_fscore_support(..., labels=ALL_LABELS, zero_division=0)` بكل الفئات العشر على Validation — وليس `VAL_LABEL_INDICES` المستخدمة فقط في تقييم macro-F1 الجولة-بجولة. النتيجة: `recall_post_agg = 0` و`logit_margin_post_agg = NaN` لهذه الفئة **في كل تشغيلة على الإطلاق** بحكم بنية الحساب، بغض النظر عمّا تعلّمه النموذج فعليًا — و`strict_erosion` احتُسبت `1` تلقائيًا لها في كل الـ29 تشغيلة (لأن `max_local_recall≈1.0` محليًا مقابل `global_recall=0` عالميًا دائمًا).

**التصحيح:** تعريف "الفئة القابلة للتقييم" = دعم حقيقي (>0 صف) في Validation، مُحتسَب من البيانات مباشرة (لا بالاسم). فئة واحدة فقط غير قابلة للتقييم في هذه البيانات: `Password Cracking`. تُستبعَد من:
- كل مقامات معدل المحو (Table 2) وبسطها.
- المقارنة المقترنة FedAvg×FedNova (Table 4).
- الارتباطات مع Gini/HHI/holder_fraction (Table 5).
- تحليل الإنقاذ (rescue، أسفل Table 5).
- الـ Heatmap (الشكل 1) والمسار المحلي-للعالمي (الشكل 2).

تبقى ظاهرة في Table 3 بعمود `class_evaluable=False`، لأن الاستبقاء المحلي (local fitting/retention) لا يزال ملاحظة صالحة — فقط لا يُقاس تعميمها العالمي.

**الأرقام المصحَّحة (بعد الاستبعاد، تحقَّقت حسابيًا من `results/confirmation_table2_erosion_rates_strict_practical.csv`):**

| المرساة | FedAvg (صارم) | FedNova (صارم) |
|---|---:|---:|
| α=0.1، 30 عميلًا | 5/45 = 11.1% | 13/45 = 28.9% |
| α=0.3، 15 عميلًا | 6/45 = 13.3% | 9/45 = 20.0% |
| α=1.0، 10 عملاء (4 تشغيلات ناجحة فقط لـFedAvg) | 0/36 = 0% | 3/45 = 6.7% |

**تحليل الإنقاذ (rescue) بعد التصحيح:** `12` حالة FedAvg-صفرية (بعد الاستبعاد، كانت `26` قبله)، منها **صفر أُنقذت بواسطة FedNova (0/12)**. الصياغة اللفظية المعتمدة سابقًا ("FedNova rescue was descriptively concentrated...") تصف نمط الشبكة الاستكشافية فقط ولا تنطبق على مراسي التأكيد الثلاثة؛ الحقل `approved_wording_confirmation_scope` في مخرج Table 5 يوثّق هذا صراحة.

### الخلل 2 — تجميع ALL_ANCHORS_POOLED يعامل صفوفًا مترابطة كمستقلة

اختبار `ALL_ANCHORS_POOLED` في Table 4 (الأصلي) جمع 14 صفًا بمفتاح (مرساة×بذرة) كأنها 14 ملاحظة مستقلة، بينما نفس الخمس بذور (11, 22, 33, 44, 55) تتكرر عبر حتى 3 مراسي لكل منها. هذا تضخيم زائف لحجم العينة (pseudo-replication) — الوحدة الإحصائية الصحيحة هي **البذرة (seed)**، لا (مرساة×بذرة). لا يُعتمَد `p=0.00037` كدليل دلالة إحصائية.

**التصحيح:** أُضيف صف `ALL_ANCHORS_SEED_AGGREGATED` — متوسط الفرق (FedAvg−FedNova في معدل المحو الصارم) لكل بذرة عبر المراسي التي ظهرت فيها (n=5 بذور مستقلة)، ثم اختبار Wilcoxon مُقترن. النتيجة: **جميع البذور الخمس بنفس الاتجاه** (FedNova أعلى محوًا من FedAvg في كل بذرة)، لكن عند n=5 فإن الحد الأدنى الممكن لـ Wilcoxon ثنائي الطرف هو `p=0.0625` (= 2/2⁵) — لا يصل لعتبة الدلالة التقليدية رغم اتساق الاتجاه الكامل. الصياغة المعتمدة: **"اتجاه وصفي متسق عبر كل البذور والمراسي، وليس تفوقًا إحصائيًا حاسمًا."**

### القرار النهائي المعتمد لصياغة النتيجة الجوهرية — ⚠️ نُسخة متجاوَزة (superseded)

**هذه الفقرة تجاوزها التصحيح الثالث أدناه (n=5 غير متجانس → n=4 متجانس، p=0.0625 → p=0.125). أُبقيت هنا فقط كسجل تاريخي لسلسلة التصحيحات؛ الصياغة المعتمدة فعليًا هي الموجودة تحت "الصياغة النهائية المعتمدة" في قسم "تصحيح ما بعد المراجعة الثالثة".**

بعد تصحيح الخللين، الاتجاه الجوهري **لم يختفِ ولم يتغيّر اتجاهه** — فقط حجمه وصياغة يقينه الإحصائي:

> ~~"The modest exploratory reduction in global zero-recall cases did not transfer to the three prespecified confirmation anchors. Within these anchors, FedNova exhibited consistently higher local-to-global erosion than FedAvg — descriptively across all three anchors and all five paired model seeds — despite FedNova's previously observed advantages in stability and overall test performance. This directional consistency does not reach conventional statistical significance at the correct seed-level unit of analysis (n=5, exact Wilcoxon floor p=0.0625), and is reported as a consistent descriptive pattern, not a statistically confirmed effect."~~ (superseded — see below)

### مبدأ عام مُستخلَص (يُطبَّق على أي تحليل لاحق في هذا المشروع)

"فئة لها حامل تدريبي" لا تعني "فئة قابلة للتقييم عالميًا" — يجب التحقق من الدعم الفعلي (>0 صف) في أي مجموعة تقييم قبل استخدام أي مقياس recall/margin/erosion مبني عليها، بصرف النظر عن وجود بيانات تدريب لتلك الفئة عند أي عميل.

### ملفات مُعاد توليدها (بلا أي تدريب جديد، بلا لمس لـheld_out_test)

`analyze_confirmation.py` عُدِّل ليحسب `EVALUABLE_CLASSES` من بيانات Validation الفعلية (لا بالاسم)، ويستبعدها من كل مقام/بسط/ارتباط/رسم كما ورد أعلاه، ويضيف صف `ALL_ANCHORS_SEED_AGGREGATED`. أُعيد توليد كل مخرجات `results/confirmation_table{1..6}_*.csv` و`figures/confirmation_fig_*.png` من نفس `item3_confirmation_mechanistic_round45.csv` المُقفَل دون أي تعديل عليه.

## تصحيح ما بعد المراجعة الثالثة (2026-08-22، تصحيحان إحصائيان صغيران قبل إقفال Item 3)

**لا تعديل على الأقسام أعلاه — إضافة فقط.**

### الخلل 3 — `ALL_ANCHORS_SEED_AGGREGATED` (من التصحيح الثاني) يجمع كميات غير متجانسة

بذرة `model_seed=33` ممثَّلة في مرساتين فقط (heavy وmoderate) لأن FedAvg تباعدت في المرساة الخفيفة، بينما بقية البذور الأربع (11, 22, 44, 55) ممثَّلة في المراسي الثلاث كاملةً. المتوسط عبر-المراسي لبذرة 33 إذن ليس نفس المقدار التجريبي المتوسَّط لبقية البذور — جمعها في اختبار واحد بـn=5 كان خطأ.

**التصحيح:** صار `ALL_ANCHORS_SEED_AGGREGATED` مُقيَّدًا الآن على البذور الأربع الموجودة في **المراسي الثلاث كاملةً** (`n_anchors_averaged == 3`): `11, 22, 44, 55` → **n=4**. بذرة 33 تبقى معروضة في الجدول الفرعي "per-seed diff" (موسومة `n_anchors_averaged=2`) لكنها **مُستبعَدة من الاختبار الموحَّد**. النتيجة: **4/4 بذور بنفس الاتجاه**، Wilcoxon ثنائي الطرف الحد الأدنى الممكن عند n=4 هو **p=0.125** (بدلًا من p=0.0625 عند n=5 غير المتجانس سابقًا). تُحذَف عبارة "all five paired model seeds" أينما وردت وتُستبدَل بـ"all available seed-level contrasts had the same direction".

### الخلل 4 — Table 5: p-values غير قابلة للاستشهاد + وسم "robustness check" غير دقيق

261 وحدة (فئة×تشغيلة) في Table 5 مترابطة داخل **3 تقسيمات فقط** — عدد أصغر بكثير مما يلزم لأي تصحيح تجميعي (clustering correction) موثوق. عمود `p_value` أُعيد تسميته **`naive_unadjusted_p_do_not_report`** صراحة — Spearman ρ يبقى تحليلًا وصفيًا فقط. تعليق "robustness check" على الجدول المُجمَّع حسب التقسيم (3 تقسيمات) حُذف؛ 3 تقسيمات لا تكفي لاختبار متانة بأي معيار. أي استنتاج عن التركيز (Gini/HHI/holder_fraction) في المخطوطة **يُبنى على تحليل الاستكشاف الصحيح ذي الـ27 تقسيمًا** (`results/analysis_q3b_partition_level_spearman.csv`)، لا على Table 5 التأكيدي — الأخير وصفي/توضيحي فقط لمراسي التأكيد الثلاثة تحديدًا.

### الصياغة النهائية المعتمدة (تَنسخ صياغة "القرار النهائي" في القسم السابق)

> "Across the three prespecified confirmation anchors, FedNova showed descriptively higher local-to-global erosion than FedAvg. All available seed-level contrasts pointed in the same direction; however, the small number of independent seeds and one incomplete FedAvg condition preclude a confirmatory significance claim."

### ملفات مُعاد توليدها في هذا التصحيح

`Table 4` و`Table 5` فقط (`confirmation_table4_paired_fedavg_vs_fednova.csv`, `confirmation_table5_erosion_vs_concentration.csv`) — بلا تدريب جديد، بلا لمس لـ`item3_confirmation_mechanistic_round45.csv` أو أي ملف مصدر آخر. الجداول 1، 2، 3، 6 والشكلان 1 و2 لم يتأثرا ولم يُعاد توليدهما (لا تغيير في منطقهما).

## Item 3 — حالة الإقفال

**مُقفَل رسميًا 2026-08-22** بعد التصحيحين الرابع والثالث أعلاه. لا عمل إحصائي أو تحليلي إضافي معلَّق على شبكة الاستكشاف (27 تقسيمًا) أو مرحلة التأكيد (29/30 تشغيلة). الخطوة التالية: تجميع `RESULTS_ITEM3_LOCKED.md` (على نمط `RESULTS_ITEM1_ITEM2_LOCKED.md`) ثم كتابة الورقة الأولى.
