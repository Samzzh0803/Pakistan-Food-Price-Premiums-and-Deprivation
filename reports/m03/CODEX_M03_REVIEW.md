# Codex Milestone 03 independent review

Reviewed `milestone-03` at `ef223aa` on 24 October 2026 (PKT). Scope: Prompt 2, read-only implementation review. Only this review file is to be committed; code, data, notebook and manuscript were not edited.

**Verdict: changes required before submission.** The published numbers reproduce, but the last CV validation block uses the first test week's prices to construct its outcomes. The final-fit purge does not protect hyperparameter or cost-threshold selection from this boundary contamination. The formal H3 wording also exceeds what the current uncertainty calculations establish.

## Verification scope and independent results

I extracted a clean tracked snapshot with `git archive HEAD` into an isolated scratch directory and ran the documented model script there, regenerated the notebook, and executed it using nbconvert. Local cloning was prevented by Git ownership checks; this was a clean tracked snapshot, not a fresh network clone. No original implementation artifacts were overwritten. The model uses only the existing M02 master parquet; no holdout workbook was parsed or used.

- Model script: exit 0; all five headline rows reproduced.
- Regenerated notebook: 36 executed code cells, zero error outputs, zero unexecuted code cells.
- Repeated computation through the notebook: all 30 model artifacts (JSON, tables, figures) were byte-identical to the first scratch run.
- Against committed artifacts: all 22 tables/text tables matched bytes, all JSON values matched, but JSON serialization and seven PNGs differed in bytes. Same-environment determinism passes; identical rendering across environments is not established. Dependencies have lower bounds rather than a locked environment.
- Installation was not independently completed: an offline `pip install --dry-run --no-index -r requirements.txt` could not satisfy `pdfplumber>=0.11` from local packages. Existing installed dependencies were sufficient for model/notebook execution. This is an environment verification limit, not proof of a broken requirement.
- PDF: 15 pages, all containing extractable text. Visual layout was not independently rendered. The optional 12–14-page target is exceeded by one page; it is not a page limit in the supplied course rubric.

Independent reconstruction used the parquet directly, recomputed shifted next-week changes, fitted item quantiles, applied the final purge, constructed features, and fitted a separate StandardScaler/logistic pipeline rather than taking the reported scores on trust:

| Quantity | Independently obtained | Reported agreement |
|---|---:|---|
| Whole-period data-card labelled rows / positives | 7,095 / 205 | Yes |
| Whole-period positive rate | 0.0288935870 | Yes, rounded 0.0289 |
| Final train rows / positives | 4,560 / 135 | Yes |
| Final test rows / positives | 1,824 / 54 | Yes |
| Train and test positive rate | 0.0296052632 each | Yes, rounded 0.0296 |
| Logistic test ROC-AUC, C=0.01 | 0.6924461184 | Yes, 0.6924 |
| Category-rule test ROC-AUC | 0.7076585060 | Yes, 0.7077 |
| Own-seed shuffled fit, seed 20261024 | 0.5553044570 | Chance check is noisy |
| Own-seed 30 shuffled fits, mean / sample SD | 0.4970412220 / 0.0709049465 | Consistent with chance on average |

## A. Reproducibility

**1. CONCERN — Computation passes; fresh-install verification remains incomplete.** See execution evidence above. README.md:40–44 gives the model/notebook/report commands; scripts/m03_models.py calls `run_all(write=True)` and scripts/m03_make_notebook.py:197 does the same. Same-environment reruns match all 30 artifacts. Do not claim that a fresh dependency installation or cross-environment byte stability was verified. Model execution emitted a joblib physical-core warning; notebook startup emitted Windows event-loop/profile-permission warnings, but neither prevented successful execution.

**2. PASS — Requirements parse as individual package specifications.** requirements.txt has 17 nonempty requirement lines, including separate `pyarrow>=15.0` and `scipy>=1.11`, plus `nbformat>=5.9`. Pip parsed these successfully before failing to resolve an unavailable local package. A lockfile or tested version set would improve portability.

## B. Leakage

**3. PASS — Week-14 holdout remains excluded from the modelling path.** pfp/model.py:135–139 reads only the M02 master and checks `week_end.max() <= TRAIN_CUTOFF`; :142–155 checks the maximum label endpoint as well. pfp/config.py defines cutoff 2026-09-24. The M03 runners do not read `data/raw/pbs_holdout/`. The assertion is after reading the master: it checks the stored dataset's scope, rather than preventing the physical reading of an accidentally contaminated parquet. On the supplied master it passes. This is separate from the internal April–August test boundary issue below.

**4. PASS — Primary label quantiles are fitted on the appropriate training rows.** pfp/model.py:195–232 computes non-zero-change 90th percentiles from `fit_rows`, drops items with fewer than five non-zero changes, and applies those thresholds unchanged to evaluation rows. `make_fold_data` rebuilds them per fold. The final split drops five unlabelled items and retains 27. Whole-period thresholds at :731–739 are explicitly data-card-only. Exception: the deliberately random-fold demonstration reuses final training labels rather than refitting label thresholds for each random fold; its score is a diagnostic with a different labelling protocol, not a clean alternative CV estimate.

**5. PASS for each local split; FAIL for the outer development boundary.** pfp/model.py:215–232 removes training rows whose `next_week` is in that split's evaluation block. Final training therefore has 10 effective weeks, after removing 541 candidate rows at 2026-03-26. However, :741–743 builds CV folds from the original 11 pre-test feature weeks before applying that final boundary restriction. Fold 5 validates on 2026-03-26. Independently checked: all **456 labelled validation rows**, including **23 positives**, have `next_week=2026-04-02`, the first test week. Those labels flow into `tune` and pooled CV cost thresholds. The final training purge does not remove them from model selection. Fix by excluding every development outcome whose endpoint reaches the outer test period before constructing folds; revise the fold schedule transparently if five folds must be retained. Add an assertion covering both CV training and validation label endpoints against the outer test boundary. Rerun all downstream results.

**6. PASS — Scalers and category rates fit training rows only.** pfp/model.py:236–293 uses StandardScaler and MinMaxScaler in cloned pipelines; `CategoryRateRule.fit` computes rates from its supplied training labels only. No full-panel fitted scaler appears in the primary models. City deprivation/population terciles are descriptive groups based on fixed city covariates, not outcome-fitted model features.

**7. FAIL — Outer test-period prices enter model selection through CV labels.** The direct grid search (`tune`, :346–374) and threshold search (`best_threshold`, :403–427; `cost_analysis`, :438–458) consume CV predictions, not test AUCs. Nevertheless those CV predictions include the contaminated fifth validation block described in item 5. This affects both hyperparameters and operating thresholds. R_REF=20 is defensible as a descriptive reference: independent pooling confirms every nontrivial model raises CV alerts at r=20, while category and KNN raise none at r=10. The code's `h3.best_test_model` and test-cost winners should be treated only as descriptive comparisons, not as validated model/policy selection. Multiple predeclared diagnostics use the test data; describe 'scored once' as one final evaluation per fixed model, not one literal access to test outcomes.

**8. PASS as a sanity check, not a proof of no leakage.** Own seed 20261024 gives AUC 0.5553 for one permuted fit; 30 independent permutations average 0.4970 (SD 0.0709). The committed 200-shuffle mean 0.503 is plausible. Passing this check cannot detect the CV boundary problem and does not justify 'so nothing outside the features carries the label' (scripts/m03_report.py:295–300).

**9. PASS — Predictor columns use information available at feature week t.** FEATURES excludes `dlog_next`, `next_week`, label cutoffs and future prices. pfp/model.py:142–181 constructs current/lagged changes and fixed category/province indicators. pfp/build.py:136 and :153 compute same-week relative prices and backward-looking lag changes. Same-week cross-city quotes are assumed available when the PBS release is available. External population/deprivation sources predate the prediction period.

## C. Numbers and definitions

**10. PASS — Independent counts and primary AUCs match.** See the table above. The pre-test estimates were explicitly approximate; the final counts follow training-only thresholds and item exclusions. The whole-period 7,095-row data card and the 6,384-row final train-plus-test sample use different threshold fit scopes plus a purge, so their difference must not be described as just the purged week.

**11. PASS for the required numerical spot-check; CONCERN for universal provenance claims.** I checked the following 12 rendered manuscript values against the named JSON paths, all present and correctly rounded:

| JSON path | Rendered manuscript value |
|---|---:|
| data.rows | 7,095 |
| data.positives | 205 |
| split.train_rows | 4,560 |
| split.train_pos | 135 |
| split.test_rows | 1,824 |
| split.test_pos | 54 |
| test.logistic.roc_auc | 0.692 |
| test.category.roc_auc | 0.708 |
| test.logistic.auc_ci_lo | 0.636 |
| test.logistic.auc_ci_hi | 0.746 |
| leak.random_kfold_auc_mean | 0.710 |
| leak.forward_auc_mean | 0.637 |

The builder formats JSON values and also renders CSV-derived tables. It does not establish that every numerical statement has its own M03 JSON entry. Fixed settings, calendar dates and inherited M02 numbers have other provenance. The Eid date statement is specifically unverified/incorrect in the submitted prose (item 17). Do not turn a successful spot-check into a universal verification claim.

**12. PASS — Metric formulas and matrix orientation are correct.** pfp/model.py:297–308 computes specificity TN/(TN+FP), precision TP/(TP+FP), recall TP/(TP+FN), accuracy and F1 correctly, returning zero for undefined precision/recall denominators. :659–678 displays [[TP,FN],[FP,TN]], actual rows and predicted columns, abnormal first. For logistic at r=20: TP=45, FN=9, FP=892, TN=878; recall=45/54=0.8333, precision=45/937=0.0480, specificity=878/1770=0.4960. `pr_auc` is sklearn average precision, not trapezoidal PR area; label it 'average precision (AP; reported as PR-AUC)' for precision in the viva. The category and KNN scores below 0.5 explain their all-negative predictions at that threshold.

**13. CONCERN — Bootstrap arithmetic is correct, inferential claims need qualification.** pfp/model.py:311–324 samples 1,824 row indices with replacement 2,000 times, rejects one-class resamples, and shares indices across models for paired differences. The reported logistic CI [0.636,0.746], tree [0.576,0.650] and KNN [0.529,0.665] support rejection under this row-independent calculation, as requested by Prompt 1. But rows share item/week shocks and repeat city-item series; the report itself admits this at scripts/m03_report.py:468. Thus the intervals do not establish generalisation over independent future weeks, and 'tree significantly worse' has the same limitation. Present the row-bootstrap result as conditional/exploratory, with dependence-aware uncertainty as a sensitivity analysis or clearly deferred limitation given only four test weeks.

Separately, pfp/model.py:388–400 shuffles only training labels, holds test labels fixed, and reports the raw exceedance share as a 'permutation p-value for H3'. This is a useful training-label sanity experiment, but it does not establish an exchangeable null distribution for the stated panel H3 or reproduce the model-selection procedure under the null. It should not be cited as an independent formal confirmation. Even for a valid Monte Carlo permutation test, 0/200 is an observed exceedance count, not p=0: use (b+1)/(B+1), giving 1/201=0.00498 when b=0. See the [sklearn permutation-test documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html). Prefer keeping this explicitly as a sanity check in this milestone.

## D. Rubric coverage

**14. PASS on broad coverage; CONCERN/FAIL on the qualifications below.**

| Brief requirement | Evidence | Assessment |
|---|---|---|
| A: naive benchmark and at least two distinct ML models | pfp/model.py:236–279; report section 6.1; notebook Part A | PASS: majority, category, logistic, CART, KNN |
| A: theoretical and domain justification | report section 6.1; notebook Part A | PASS |
| B: split ratios, rare-class representation | report sections 5–6; tables/data_card.csv and folds.csv; notebook B.1–B.2 | PASS: ratios and 16–26 fold positives stated; temporal alternative justified |
| B: leakage-free CV with fold variance | report section 6; tables/cv_per_fold.csv; notebook Part B | FAIL: outer boundary issue in items 5/7; mean and SD otherwise supplied |
| B: systematic tuning evidence | tables/tuning_logistic.csv, tuning_tree.csv, tuning_knn.csv; fig16_tuning.png | PASS: 4, 30 and 48 candidate settings; simplicity rule documented |
| C: six required metrics and extra PR-AUC | tables/metrics_main.csv, metrics_at_cost_threshold.csv; report tables 12–13/A2 | PASS: metrics at 0.5 and cost threshold; CV means/SD at 0.5 |
| C: quantitative/social asymmetric costs | report section 8; tables/costs.csv, costs_equity.csv; notebook C.2–C.4 | PASS: FP=1, FN=r; five r values; users and harms explained; thresholds need rerun |
| C: formatted matrices | report section 8.2, fig18_confusion.png; tables/confusion_matrices.csv | CONCERN: majority/logistic/tree plotted; KNN only in numeric table |
| C: edge cases and systematic errors | report section 8.3; edge_false_negatives.csv, edge_false_positives.csv; error_breakdown.csv | PASS: five groupings; nine available FNs, not ten, correctly reflects only nine misses |
| D: driver ranking and interpretation | report section 9; tables/importance.csv, tree_rules.csv; notebook Part D | PASS: standardized coefficients, odds ratios, empirical leaf rates, cross-model ranks |
| D: preliminary policy takeaways | report section 9 | PASS, with inference cautions in item 16 |
| D: advanced models, bias/fairness, ethics roadmap | report section 10; notebook D.5 | PASS: ensembles, fairness audit, threshold ethics, limitations |
| Submission: executed documented notebook | notebooks/M03_baseline_models.ipynb | PASS: regenerated/executed successfully |
| Submission: manuscript enhancement embedded in M02 | scripts/m03_report.py:60–91 and reports/m03/M03_report.html/.pdf | PASS: inherited M02 manuscript plus sections 5–10 |
| Submission: modular training/evaluation scripts | pfp/model.py; scripts/m03_models.py | PASS |

Prompt-specific gaps beyond the course minimum: `alert_budget` at pfp/model.py:461–486 aggregates catches/positives across weeks rather than supplying recall@N and precision@N for each test week; add the four-week detail table alongside the pooled totals. The visual confusion matrices omit KNN due to the explicit filter at :998; add its two panels or equivalent formatted matrices. Category matrices are also absent if interpreting 'each model' to include the heuristic. Cross-model feature ranking covers logistic/tree, which is reasonable for the taught coefficient/impurity methods; KNN has no intrinsic ranking.

**15. PASS — The temporal choice is clearly defended.** Report section 6 and VIVA_PREP.md questions 4–5 explain future-to-past leakage, panel dependence and rare-class checks. Random stratified AUC 0.710 versus temporal 0.637 is shown. However, do not say the entire 0.073 gap causally measures leakage: different training sizes, label-fit protocols and validation regimes also contribute. Class weights address the learning objective; they are not a substitute for stratification. Revise the clean-CV claim after fixing item 5.

## E. Honesty and wording

**16. CONCERN — Mostly candid, but several assertions are too strong.** The category rule's 0.708 versus logistic's 0.692 and category-only ablation are prominent; low precision and modest signal are explicit. Good. Revise the definitive H3 rejection/paired significance language and formal permutation agreement (item 13). 'No shuffle predicts, so nothing outside the features carries the label' is not a valid leakage proof. Report section 8's 'model pays off only if ... roughly 20 or more' should say 'in these observed test comparisons'; r=10/20/35 does not identify a universal break-even point. Section 8 compares r=20 costs approximately, not exactly: logistic **587.7193** versus flag-nothing **592.1053** per 1,000. Claude's handoff saying they cost 'the same' is rounding shorthand. Coefficients are associations, and the report generally says so.

**17. FAIL — Eid dates were not checked before writing.** scripts/m03_report.py:413–417 and scripts/m03_make_notebook.py:401–404 call Eid-ul-Fitr 'expected around 20 March 2026'. Pakistan actually observed **21 March 2026**, confirmed by [Associated Press of Pakistan, 19 March announcement](https://www.app.com.pk/national/eid-ul-fitr-on-saturday-as-shawwal-moon-not-sighted/). Eid-ul-Adha was **27 May 2026**, confirmed by [Radio Pakistan, 17 May announcement](https://radio.gov.pk/17-05-2026/moon-sighted-eid-ul-adha-to-be-celebrated-on-27-may). Use these Pakistan-specific observed dates and cite them. Keep the suggested price connection explicitly untested: dates alone provide no evidence of a holiday effect. The false negatives also include price falls, so do not treat every abnormal move as a harmful upward shock.

**18. PASS — Costs are explicitly normative and not double-counted.** Report section 8 and `expected_cost` use TP=TN=0, FP=1, FN=r, with r in {1,5,10,20,35}; deprivation weights apply only to FN. `best_threshold` correctly searches unique score cutoffs plus infinity and resolves equal-cost ties toward fewer alerts. The constant-probability 'flag all versus none' crossover is (1-p)/p = **32.7778**, rather than exactly 1/p = 33.7778. The inverse-rate language is acceptable as a rough intuition, but the exact derivation is easy to add. The alert-budget presentation would benefit from item 14's weekly table.

**19. CONCERN — Viva answers largely match the code, but reproduce the weak claims.** VIVA_PREP.md:124–128 gives the row-bootstrap H3 decision and a dependence caveat, but still treats the shuffle experiment as formal corroboration; align with item 13. Questions 4–5 should distinguish temporal validation from actual stratification and frame the random/temporal gap as a comparison, not a controlled attribution. Question 13's 'min-max puts every feature on 0–1' should specify training values: held-out values can exceed that range with the default scaler. Question 1's 7,095/205 describes the whole-period data card, not the final fitted sample; explain the distinction. Gini arithmetic, class weights, odds ratios, cost sensitivity and the negative result against the heuristic are otherwise defensible.

## F. Code quality

**20. PASS with small concerns.** pfp/model.py separates features, split construction, model definitions, evaluation, tuning, costs, diagnostics and output generation into commented functions. No silent exception handler or global warning suppression was found. Assertions give useful local checks, but the opening claim that all leakage rules are enforced is too broad without the outer CV endpoint assertion. scripts/m03_report.py:181–205 hardcodes expected result assertions: useful against accidental prose drift, but legitimate reruns can fail because the narrative assumes particular winners/results; refresh assertions and text after the boundary fix. `tree_rules` reports actual unweighted training rates correctly despite class-weighted fitting. The optional Edge-specific PDF command is documented, so portable PDF generation remains a reproducibility limitation rather than a hidden dependency.

## Severity-ranked fix list

### Blocker — before submission

1. **Protect the outer test boundary throughout selection:** pfp/model.py:741–743 must construct development folds only from outcomes ending before the first test week; remove/reassign the contaminated 456-row CV block, assert endpoint isolation, rerun tuning, thresholds, notebook and manuscript, and explain any revised fold count/schedule.
2. **Correct the formal evidence claims:** pfp/model.py:388–400, scripts/m03_report.py:295–324 and VIVA_PREP.md:124–128 must label shuffled-training fits as a sanity check and qualify row-bootstrap H3/paired significance as conditional/exploratory; either justify dependence-aware inference or defer the definitive population-level conclusion. Never report zero as a Monte Carlo p-value.

### Should fix

3. **Correct and source Eid dates:** scripts/m03_report.py:413–417 and scripts/m03_make_notebook.py:401–404 should use Pakistan's 21 March / 27 May 2026 observed dates, preserving the untested-hypothesis framing.
4. **Add per-test-week alert-budget results:** extend pfp/model.py:461–486 with week-specific caught/positive/precision/recall rows for each N, retaining the current pooled summary.
5. **Include all requested formatted confusion matrices:** remove the KNN exclusion at pfp/model.py:998 or provide separate readable panels; clarify whether the category heuristic also gets its own matrix.
6. **Tighten explanatory claims:** qualify the random-CV gap and empirical cost break-even language in report sections 6/8 and VIVA_PREP.md; distinguish data-card counts from the fitted sample and AP from trapezoidal PR area.

### Nice to have

7. **Record a tested dependency environment:** requirements.txt/README.md should state tested Python/package versions and make the PDF generation dependency explicit; a fresh-install check remains outstanding.
8. **Trim one PDF page if practical:** reduce the 15-page manuscript toward Prompt 1's 12–14-page target without cutting evaluation evidence.
9. **Use the exact constant-policy cost crossover:** report section 8 can state r=(1-p)/p (32.78 here) while keeping 1/p as an approximation.
