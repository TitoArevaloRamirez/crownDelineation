# Implementation Pipeline

## Objective
Determine whether coarse crown labels generated from tree-top seeds and clustering are good enough to improve supervised tree crown detection models.

---

## Stage 0. Organize the data (DONE)

**Goal:** prepare all datasets in a common format so every later experiment uses the same inputs.

### Tasks
1. Collect the datasets:
   - NEON dataset
   - Custom dataset with 3 illumination conditions **Only RGB**

2. For each dataset, prepare:
   - images
   - manual crown annotations
   - train/validation/test split
   - metadata file with:
     - dataset name
     - illumination condition
     - image ID

3. Convert all annotations into one common format:
   - preferably COCO or YOLO
   - use the same format for all models whenever possible

4. Define one fixed evaluation split and do not change it later.

### Output
- Clean dataset folders
- Common annotation format
- Fixed train/validation/test split

---

## Stage 1. Run pretrained baselines (In Progress)

**Goal:** measure how well public pretrained models perform without adaptation.

### Tasks
1. Run inference with:
   - Detectree2 pretrained
   - DeepForest pretrained
   - TCD pretrained

2. Evaluate on:
   - NEON test set
   - Custom dataset test set under all 3 illumination conditions

3. Save:
   - predictions
   - metrics
   - example visualizations

### Output
- Pretrained baseline predictions
- Baseline metrics table
- Visualization examples

---

## Stage 2. Fine-tune baselines with manual labels (To Do)

**Goal:** establish the best achievable performance using high-quality annotations.

### Tasks
1. Fine-tune each baseline model using manual annotations:
   - Detectree2
   - DeepForest
   - TCD

2. For each model, test 3 to 5 freezing strategies:
   - no freezing
   - shallow freezing
   - medium freezing
   - deep freezing

3. Evaluate all fine-tuned models on the same fixed test sets.

4. Select the best fine-tuning configuration for each model.

### Output
- Best manual fine-tuned model for each baseline
- Metrics for each freezing strategy
- Best-performance reference results

---

## Stage 3. Simulate degraded labels (To Do)

**Goal:** determine how much label degradation can be tolerated before fine-tuning stops being useful.

### Tasks
1. Start from the manual annotations.

2. Generate degraded labels using two types of degradation:

#### A. Random displacement
- shift crown positions by controlled random offsets

#### B. Shape distortion
- random scaling
- random affine bounding box deformation
- partial crown removal

3. For each degradation type, generate quality levels (compare with ground truth, manual annotations):
   - 75%
   - 50%
   - 25%

4. Fine-tune the best baseline models from Stage 2 using these degraded labels.

5. Evaluate all results on the same fixed test sets.

6. Compare against:
   - pretrained baseline
   - best manual fine-tuned model

### Output
- Degraded annotation sets
- Performance-versus-label-quality results
- Practical quality range for useful pseudo-labels

---

## Stage 4. Generate tree-top seeds (In Progress, to edit)

**Goal:** produce initial seed points for the clustering pipeline.

I have done it as part of the clustering method

### Tasks
1. Implement the tree-top seed extraction method on the custom dataset.

2. Use low-solar-angle images to:
   - extract bright canopy-top candidate pixels
   - convert candidate regions into point seeds

3. Save:
   - seed coordinates
   - optional seed confidence
   - overlay visualizations

4. Manually inspect a small sample and record:
   - correct seeds
   - missed trees
   - false seeds

### Output
- Seed set for each image
- Seed visualization overlays
- Basic seed quality summary

---

## Stage 5. Run the 4 clustering alternatives (In Progress, to edit) 

**Goal:** convert tree-top seeds into coarse crown segments.

### Tasks
1. Implement clustering alternative 1.
2. Implement clustering alternative 2.
3. Implement clustering alternative 3.
4. Implement clustering alternative 4.

For each clustering alternative:
- input = image + seed points
- output = coarse crown segments or patches

5. Convert each clustering result into pseudo-label format usable for training.

6. Evaluate each clustering result directly against manual annotations using simple geometric metrics:
   - IoU or overlap
   - crown center distance
   - precision/recall if applicable

7. Select the best 1 or 2 clustering alternatives for the next stage.

### Output
- Pseudo-labels from 4 clustering methods
- Direct quality comparison of clustering methods (ground truth)
- Selected best clustering alternatives

---

## Stage 6. Optional YOLO refinement (To Do)

**Goal:** refine coarse pseudo-labels before using them for baseline fine-tuning.

### Tasks
1. Train YOLO using:
   - clustered pseudo-labels as training input
   - or clustered patches converted to detection labels

2. Run YOLO on the training images to obtain refined detections.

3. Compare YOLO outputs against manual annotations.

4. Measure whether YOLO improves label quality relative to raw clustering outputs.

5. Keep YOLO only if it clearly improves pseudo-label quality.

### Output
- YOLO-refined pseudo-labels
- Comparison between raw clustering labels and YOLO-refined labels
- Decision on whether YOLO is worth keeping

---

## Stage 7. Fine-tune baselines using pseudo-labels (To Do)

**Goal:** test whether generated pseudo-labels are useful for downstream learning.

### Tasks
1. Fine-tune Detectree2, DeepForest, and TCD using:
   - best clustering pseudo-labels
   - optionally YOLO-refined pseudo-labels

2. Use the same fine-tuning settings selected in Stage 2.

3. Evaluate on the same fixed test sets.

4. Compare against:
   - pretrained baseline
   - manual fine-tuning upper bound
   - degraded-label fine-tuning results

### Output
- Pseudo-label fine-tuned models
- Final comparison between pseudo-label training and other training strategies

---

## Stage 8. Compare all results (To Do)

**Goal:** determine whether the generated labels are good enough to support fine-tuning.

### Tasks
1. Build a final comparison including:
   - pretrained models
   - manual fine-tuned models
   - degraded-label fine-tuned models
   - clustering pseudo-label fine-tuned models
   - clustering + YOLO pseudo-label fine-tuned models

2. Answer the following questions:
   - Do pseudo-labels improve over pretrained models?
   - How close are pseudo-label results to manual fine-tuning?
   - Which clustering alternative works best?
   - Is YOLO worth keeping?

### Output
- Final results table
- Final plots
- Final decision on pseudo-label usefulness

---

## Student 2: Baseline training (Mario)
- run pretrained Detectree2, DeepForest, and TCD
- fine-tune with manual labels
- test freezing strategies

## Student 3: Degraded-label experiments (Mario) 
- implement label degradation
- generate 75%, 50%, and 25% variants
- fine-tune with degraded labels

## Student 4: Seed extraction (Michel)
- implement tree-top seed detection
- generate seed files
- visualize successes and failures
- implement the 4 clustering alternatives
- convert outputs into pseudo-labels
- compare clustering quality

## Student 6: YOLO refinement (Michel)
- train YOLO using clustered outputs
- evaluate whether YOLO improves pseudo-label quality

## Student 7: Final pseudo-label fine-tuning (Mario)
- fine-tune baselines using pseudo-labels
- compare with pretrained, manual, and degraded-label results

---

1. Use the same dataset split.
2. Use the same annotation format.
3. Use the same evaluation metrics.
4. Use the same file naming convention.
5. Use the same experiment logging format.

---

# Minimal Experiment Log Format

Each experiment record should contain:
- dataset
- illumination condition
- model
- training label type
- clustering method
- freezing strategy
- metrics
- notes

---

# Final Decision Logic

At the end of the project, answer:

1. Do pseudo-labels improve over pretrained models?
2. Are pseudo-label results inside the useful range defined by degraded manual labels?
3. Which clustering method is best?
4. Does YOLO help enough to justify its inclusion?

If the answer to questions 1 and 2 is yes, then the proposed pseudo-label pipeline is useful.
