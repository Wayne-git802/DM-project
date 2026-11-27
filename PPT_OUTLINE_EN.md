# Presentation Outline: Using Recall Notices to Guide Product Design Review

**Research question.** Given a product category, name, and visual description, which two hazard labels should a designer check first, and which component does the historical evidence suggest reviewing?

| Presenter | Slides | Question | Time |
|---|---|---|---:|
| 1 | 1–4 | Where did the records come from, and how did we clean them? | 4:00 |
| 2 | 5–7 | What patterns did we find, and did they recur on held-out notices? | 4:00 |
| 3 | 8–11 | How well do two suggestions work, and what should a designer inspect? | 4:00 |

Allow 12 minutes to present and 2 minutes for questions. Use the main headline and visual on each slide. Keep the explanation and hand-off in the speaker notes. [Course requirements](Project%202_requirementpdf.pdf) · [Chinese outline](PPT制作方案.md) · [Project workflow](OUTLINE_NEO.md)

## Terminology for every slide

| Use this term | Field and meaning |
|---|---|
| **Hazard label** | `hazard_classification`, the RECALL-MM annotation, such as `FALLING` or `FIRE`. Use **reference hazard label** for the target and **predicted hazard label** for the model output. |
| **Official hazard statement** | `official.hazard_description`, the CPSC text preserved as evidence. |
| **Recall description** or **reason for recall** | `recall_description`, the RECALL-MM text describing why the product was recalled. |
| **Failure mechanism** | A description of what fails and how, such as cracking at a hinge. In the preprocessing notebook, `failure_modes` holds categories extracted from `recall_description` using keywords. |
| **Visual product description** and **visible structure cue** | `visual_product_description` is the image-derived appearance text. A cue such as a hinge or grille is extracted from that text. The preprocessing export calls these `visible_structures`. The official product description, where available, is a separate field under `official.product_description`. |
| **Source record** and **recall notice** | A source record is an original RECALL-MM row. A recall notice is an official CPSC announcement identified by its public recall number. |

## Presenter 1 — From source records to verified notices

### 1. Where the data came from (0:45)

- Show the data path from public CPSC notices through RECALL-MM to this project's reconciled dataset. RECALL-MM covers 2000–2024 and adds image-derived appearance text and derived category, hazard, and remedy labels.
- Report 6,874 source rows, 6,464 distinct source IDs, and 11 product categories. Ask why we would need to inspect a record before counting it. [RECALL-MM, Section 3.1](https://arxiv.org/html/2503.23213v1#S3.SS1)

### 2. What one recall record tells us (0:55)

- Show a Tern folding bicycle card. The product description mentions the folding structure, the official hazard statement says the top-tube hinge can crack, and the RECALL-MM hazard label is `FALLING`.
- In the notes, identify `recall_number` and `recall_date`, `product_classification` and `product_name`, `visual_product_description`, `hazard_classification` and `recall_description`, plus `product_quantity`, `remedies` and `remedies_classification`. Product information supplies model inputs. Recall and remedy descriptions help explain the case afterward.
- For a quick data overview, the raw category labels include home appliances on 1,203 rows and toys and children's products on 1,201 rows. Label these as source-row counts. One row can carry multiple category labels. [Tern official notice](https://www.cpsc.gov/Recalls/2013/Stile-Products-Recalls-Tern-Folding-Bicycles) · [Preprocessing notebook](DataPreprocessing.ipynb)

### 3. Two anomalies that changed our preprocessing (1:00)

- **Unit mismatch.** The largest raw quantity is 118,000,000. CPSC notice 16-101 reports approximately 118 million **gallons of propane**. Preserve the original unit and leave the comparable item count missing. The analysis counts notices. [CPSC propane notice](https://www.cpsc.gov/Recalls/2016/Crescent-Point-Energy-Recalls-To-Inspect-Propane-Gas)
- **Reused source ID.** Three chandelier rows share source ID 1177. Official reconciliation identifies three notices, 12-719, 16-748 and 17-184. This explains why deduplication needs verified official numbers. [Official reconciliation](recall_audit/01_公告核对与字段关联.ipynb)

### 4. Build one row per official recall notice (1:20)

- Show **6,874 source rows → 6,835 matched rows → 6,637 official recall notices**. The remainder consists of 38 unresolved rows and one settlement news release.
- One preprocessing entry point reads the original JSON, checks dates, text, label spelling and missing quantities, and reconciles those source rows against CPSC snapshots. The main notebooks share the resulting notice dataset. Tern 13-231 illustrates how multiple source rows become one official notice while retaining their appearance variants and source-row references.
- Transition: “Now that each row represents a traceable notice, what hazard labels recur in each product category?” [Reconciliation summary](recall_audit/data/reconciliation_summary.json)

## Presenter 2 — Discover and check patterns

### 5. Which hazard labels are common within a category? (1:05)

Use Apriori to count co-occurring category and hazard labels. Require at least 30 joint notices for candidate pairs. Present each category's count and share beside the overall share.

| Category and hazard label | Category share | Overall share |
|---|---:|---:|
| Electrical products and FIRE | 374/594 (63.0%) | 24.8% |
| Home appliances and FIRE | 621/1,174 (52.9%) | 24.8% |
| Transportation and CRASH | 352/769 (45.8%) | 6.4% |

The fire pattern agrees with the original paper's product–hazard heatmap. Both analyses use the same underlying data source. The category frequencies give us a baseline, and the broad categories motivate looking for more specific groups. [Full-data rules](recall_audit/01_公告核对与字段关联.ipynb) · [RECALL-MM, Section 4.2](https://arxiv.org/html/2503.23213v1#S4.SS2)

### 6. Group products before looking at the labels (1:30)

- Show **name and visual description → Sentence-BERT → PCA → K-means → reveal labels**. Of 6,413 notices with usable text, 5,126 fitted the clusters and 1,287 were held out. A bicycle-related cluster contains the FALLING label on 35/84 held-out notices (41.7%), compared with 16.8% across the holdout.
- The paper used Sentence-BERT and t-SNE to visualize similar product names or recall descriptions. Our K-means task uses name and appearance to define groups, then checks the label distribution of held-out products. This moves from a similarity plot to a measurable comparison. [Clustering notebook](recall_audit/01_公告核对与字段关联.ipynb) · [RECALL-MM, Section 3.2](https://arxiv.org/html/2503.23213v1#S3.SS2)

### 7. Do the patterns recur on held-out notices? (1:25)

- Diagram five folds. Group notices by normalized official product name and confirmed links before splitting. Mine the rules on four folds, then check the **reference hazard labels** in the remaining fold.
- Eleven representative rules were selected in all five training rounds. All eleven retain higher category shares than the overall shares in each held-out fold.
- Transition: “We know what recurs within a category. For a specific product, can its name and appearance help us pick two labels?” [Fold-by-fold rule checks](recall_audit/02_危险分类评估.ipynb)

## Presenter 3 — Evaluate suggestions and trace them to components

### 8. Which extra fields are useful? (0:50)

Hold the logistic regression setting, grouped folds and two-label budget fixed. Show the change as each field is added.

| Inputs | Notices with a hit |
|---|---:|
| Product category | 4,526/6,637 |
| Add product name | 5,300/6,637 |
| Add visual product description | 5,390/6,637 |
| Add visible structure cues | 5,393/6,637 |

The product name adds the most hits. Structure cues add three hits and also give us a way to find relevant components later. [Feature comparison](recall_audit/02_危险分类评估.ipynb)

### 9. What did the course baselines and advanced methods teach us? (1:20)

- **kNN** directly tests whether similar products suggest useful hazards. Find the closest training notices and let their hazard labels vote, weighted by similarity. All five inner validations selected 21 neighbors.
- **Gini decision tree and XGBoost** share inputs and folds. A single Gini tree repeatedly splits cases into purer groups. XGBoost adds trees sequentially to improve the current predictions. In this matched comparison, XGBoost has 606 more hits.
- **Soft voting** combines logistic regression and XGBoost scores. It gains 35 hits over logistic regression alone. kNN has 20 more hits than XGBoost in the main comparison, with different feature representations.

| Model | Hits out of 6,637 | Hit@2 | micro-F1 for backup |
|---|---:|---:|---:|
| Category frequency | 4,526 | 68.19% | 0.4546 |
| Gini decision tree | 4,711 | 70.98% | 0.4731 |
| kNN | 5,337 | 80.41% | 0.5361 |
| XGBoost | 5,317 | 80.11% | 0.5340 |
| TF-IDF logistic regression | 5,415 | 81.59% | 0.5438 |
| Soft voting | 5,450 | 82.12% | 0.5474 |

Hit@2 means at least one of the two predicted hazard labels matches a reference label. Transition to a concrete component, “The model suggests FALLING. Which part of the bicycle should we inspect?” [Classifier results](recall_audit/02_危险分类评估.ipynb) · [XGBoost paper](https://arxiv.org/abs/1603.02754)

### 10. From FALLING to the top-tube hinge (1:05)

1. Show the held-out Tern notice 13-231. Soft voting predicts `CRASH` and `FALLING`. The reference label is `FALLING`.
2. In that fold's training reference set, keep transportation notices whose names contain bicycle or bicycles, with usable appearance text and one source row. FALLING occurs in 7/7 notices that mention folding or a hinge and 102/151 of the others. Seven cases are a useful retrieval clue with a small denominator.
3. Reveal the official hazard statement. It identifies possible cracking at the hinge on the top tube. For a similar folding design, ask whether the hinge region can crack under repeated loading. [Held-out case and evidence](recall_audit/02_危险分类评估.ipynb) · [Official Tern notice](https://www.cpsc.gov/Recalls/2013/Stile-Products-Recalls-Tern-Folding-Bicycles)

### 11. What can a design team take away? (0:45)

- Historical notices help prioritize which hazard labels to consider. Product names add the largest measured improvement among the tested fields.
- Official notices help identify which component to review. For similar air conditioner or heat pump designs, the Carrier notice points to possible overheating at the power-cord plug.
- These results describe the catalog of historical recalls. Estimating field incident rates would require sales, exposure and incident data. Show each member's name and actual contribution. [Carrier official notice](https://www.cpsc.gov/Recalls/2017/carrier-expands-recall-of-packaged-terminal-air-conditioners-and-heat-pumps)

## Backup slides

- Data-quality checks, the choice of 15 clusters, and the silhouette score of 0.125. The clusters overlap, so use the holdout shares and case evidence to explain what they capture.
- micro-F1 pools correct, extra and missed hazard labels. Macro-F1 averages the per-label scores equally. Cramér’s V summarizes categorical association strength.
- The original A/B/C and appliance-subtype experiments use their original sampling units. The paper’s prediction task uses a different output budget and scoring protocol, so compare findings and methods in the main talk. [Original project outline](PROJECT_OUTLINE.md) · [RECALL-MM, Section 3.3](https://arxiv.org/html/2503.23213v1#S3.SS3)
