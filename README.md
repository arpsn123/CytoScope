
<h1 align="center">🔬 CytoScope </h1>
<h2 align="center">Computer Vision Meets LLMs for End-to-End Tissue Analysis and Explainable AI </h2>

## Overview : 
CytoScope is a **hybrid AI system** that integrates **computer vision**, structured reasoning, and **local LLM-based** interpretation into a unified pipeline for histopathology analysis.

At its core, the system performs nuclei segmentation using YOLOv8, but it does not stop at detection. Instead, segmentation outputs are transformed into quantitative features such as nuclei count, spatial density, and normalized nucleus size, enabling a shift from pixel-level predictions to structured data representation.

These features are then analyzed using statistical methods to uncover relationships in tissue structure, including distribution patterns, inter-feature correlations, and variability across samples. This allows the system to move beyond visual output and into **data-driven understanding of cellular organization**.

To formalize interpretation, a **rule-based profiling engine** converts numerical features into semantic categories (e.g., density levels, nucleus size classes, and composite tissue profiles). This creates a deterministic reasoning layer grounded in measurable properties.

On top of this, CytoScope integrates a local LLM via **Ollama** (using the `gemma4:e4b` model) to generate **natural language explanations** from structured profiles. 

The LLM does not make decisions—it explains them. This separation ensures reliability (rule-based logic) while enabling interpretability (language generation).

---

## System Philosophy

Most pipelines stop at:

```text
detection → visualization
```

CytoScope extends this into a full analytical stack:

```text
segmentation → feature extraction → statistical analysis → structured reasoning → LLM-based interpretation
```

---

## 01. Data Engineering

Transform raw histopathology data into a **YOLOv8-compatible segmentation dataset**.



### Dataset

This work uses the **PanNuke Dataset (Experimental Data)**, which provides:

* RGB tissue images
* nucleus-level segmentation masks
* diverse tissue distributions



### Core Constraint

YOLOv8 segmentation requires **polygon annotations**, not binary masks.

```text
binary mask → polygon (normalized) 
```

### Transformation Pipeline

```text
Mask → Contours → Polygons → YOLO Format
```

### Key Operations

#### 1. Contour Extraction (Instance Separation)

```python
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

* isolates individual nuclei
* converts pixel regions → boundaries


#### 2. Polygon Representation

```python
polygon = contour.reshape(-1, 2)
```

* represents each nucleus as a geometric object
* enables shape-preserving annotations


#### 3. Normalization (Resolution Invariance)

```python
(x / width, y / height)
```

* maps coordinates to [0,1]
* ensures scale consistency across images


#### 4. YOLO Segmentation Format

```text
class_id x1 y1 x2 y2 ... xn yn
```

* single-class setup → `class_id = 0`
* each line = one nucleus

---

## 02. Model Training & Initial Inference

This stage focuses on enabling the model to **learn nucleus-level segmentation** and validating whether that learning is meaningful on unseen data.



### Model Choice

The system uses YOLOv8m segmentation:

* optimized for **real-time instance segmentation**
* supports **polygon-based annotations directly**
* balances **speed and accuracy**


### Training Setup

The model is initialized using pretrained weights and adapted for nuclei segmentation:

```python
from ultralytics import YOLO

model = YOLO("yolov8m-seg.pt")  # pretrained segmentation model
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
```


### Key Hyperparameters

* **imgsz (640)** → balances detail vs memory
* **batch size** → constrained by GPU (RTX 4060)
* **epochs** → enough for convergence without overfitting


### Immediate Inference

After training, the model is tested on unseen images:

```python
results = model.predict("test_image.png", conf=0.25)
```



#### Key Issue Encountered

A critical observation during inference:

* model internally resizes images (e.g., 256 → 640)
* raw mask outputs reflect resized dimensions
* saved outputs revert to original resolution

**This mismatch required using:**

```python
r.orig_shape
```

instead of mask dimensions for correct feature computation.

---
#### Original Image : 

<img width="256" height="256" alt="img3_original" src="https://github.com/user-attachments/assets/b51ced4c-a8e3-4b5a-9197-27d907dd646f" />

#### Segmented Image : 

<img width="256" height="256" alt="img3_segmented" src="https://github.com/user-attachments/assets/eec81b7d-65d9-4f05-a95f-eddfb82fcb3e" />






---

## 03. Feature Extraction & System-Level Output

This is the point where the project transitions from:

```text
segmentation → analysis
```

Instead of stopping at predicted masks, the system converts geometric outputs into **quantitative descriptors of tissue structure**.

Each segmented nucleus is treated as a **spatial object**, not just pixels.

```text
Mask → Polygon → Geometry → Metrics
```



### Step 1 — Extract Instance Geometry

YOLOv8 segmentation outputs polygon coordinates directly:

```python
polygons = r.masks.xy  # list of (N, 2) arrays
```

Each polygon represents one nucleus in original image scale



### Step 2 — Compute Per-Instance Area

Instead of counting pixels (resolution-dependent), geometric area is computed:

```python
from shapely.geometry import Polygon

area = Polygon(poly).area
```

### Step 3 — Aggregate Image-Level Features

For each image:

#### Nucleus Count

```python
nuclei_count = len(polygons)
```

#### Total Area

```python
total_area = sum(areas)
```


#### Average Area

```python
avg_area = total_area / nuclei_count
```


#### Normalized Area

```python
avg_norm_area = avg_area / (img_h * img_w)
```


#### Density

```python
density = nuclei_count / (img_h * img_w)
```


### Interpretation

| Feature       | Meaning              |
| ------------- | -------------------- |
| nuclei_count  | population size      |
| total_area    | tissue coverage      |
| avg_area      | typical nucleus size |
| avg_norm_area | scale-invariant size |
| density       | spatial packing      |




### Output dataset:

```text
image_name | nuclei_count | total_area | avg_norm_area | density .csv
```

---


## 04. Statistical Analysis

At this point, the system has already transformed images into structured data:

```text
image → segmentation → features → dataset
```

Statistical Analysis answers the only question that matters:

> **What patterns exist in this data, and what do they imply?**



### 1. Distribution Analysis of Average Nucleus Size(Histograms)

Understand how features are **distributed across the dataset**

```python
plt.hist(df["avg_norm_area"], bins=30)
plt.title("Distribution of Normalized Nucleus Area")
plt.xlabel("Normalized Area")
plt.ylabel("Frequency")

```
<img width="581" height="455" alt="image" src="https://github.com/user-attachments/assets/e36d341b-3d0c-4fb5-86be-59b3e0095fca" />
<br>

* Whether nucleus sizes are **consistent or variable**
* Whether distribution is **narrow (uniform)** or **wide (heterogeneous)**


#### Interpretation

> A wide spread in normalized nucleus area indicates significant variability in cellular morphology, suggesting heterogeneous tissue structure across samples.

---


### 2. Density Distribution(Histograms)

```python
plt.hist(df["density"], bins=30)
plt.title("Distribution of Nuclei Density")
plt.xlabel("Density")
plt.ylabel("Frequency")
```

<img width="599" height="455" alt="image" src="https://github.com/user-attachments/assets/0c8f31aa-7545-4075-8a2a-6ec812fa1810" />
<br>

* variation in **cellular packing**
* presence of **sparse vs dense regions**

#### Interpretation

> The variation in density across images reflects differences in tissue compactness, ranging from sparse cellular arrangements to highly crowded regions.

---



### 3. Correlation Analysis (RELATIONSHIPS)

Understand how features **interact with each other**
#### Code  :
```python
corr = df.select_dtypes(include='number').corr()
print(corr)
```
#### Output : 
```text
                     nuclei_count  total_area   avg_area   avg_norm_area   density   
--------------------------------------------------------------------------------------
nuclei_count            1.000000     0.660697    -0.120822     -0.120822     1.000000          
total_area              0.660697     1.000000     0.494489      0.494489     0.660697          
avg_area               -0.120822     0.494489     1.000000      1.000000    -0.120822          
avg_norm_area          -0.120822     0.494489     1.000000      1.000000    -0.120822          
density                 1.000000     0.660697    -0.120822     -0.120822     1.000000          
         
```

### Key Relationships Observed

#### 1. Density ↔ Avg Nucleus Size

```text
≈ -0.12 (weak negative)
```
Interpretation:
> As density increases, nucleus size tends to slightly decrease.



#### 2. Nuclei Count ↔ Total Area

```text
≈ 0.66 (moderate positive)
```
Interpretation:

> More nuclei → more total coverage, but not perfectly proportional (size variability exists).

#### 3. Total Area ↔ Avg Size

```text
≈ 0.49
```

> Both count and size contribute to tissue coverage.



### Insight

> Tissue structure is governed by a combination of population (count) and morphology (size), not a single dominant factor.

---


### 4. Scatter Analysis (RELATION VISUALIZATION)

```python
plt.scatter(df["density"], df["avg_norm_area"])
plt.xlabel("Density")
plt.ylabel("Avg Normalized Area")
plt.title("Density vs Nucleus Size")
plt.show()
```
<img width="612" height="455" alt="image" src="https://github.com/user-attachments/assets/8d7f80cb-80da-4dd2-a57b-7e76c42a38e4" />
<br>

* shape of relationship (linear / nonlinear / none)
* presence of clusters or trends

#### Observation

* no strong linear pattern
* slight downward tendency

#### Interpretation

> While density influences nucleus size to a limited extent, the relationship is weak, indicating the presence of additional structural factors.

---

### 5. Extreme Case Analysis 

Look at **edge cases**, not averages

#### Code : 

```python
top_dense = df.sort_values("density", ascending=False).head(5)
```
#### Output :
```text
image_name   nuclei_count   avg_norm_area   density
---------------------------------------------------
2-844.png    125            0.002888        0.001907
1-2610.png   102            0.003950        0.001556
1-1486.png   99             0.004519        0.001511
```


#### Code : 
```python
top_dense = df.sort_values("avg_norm_area", ascending=False).head(5)
```
#### Output : 
```text
image_name   nuclei_count   avg_norm_area   density
---------------------------------------------------
2-1715.png   2              0.038942        0.000031
3-24.png     10             0.034994        0.000153
2-1642.png   8              0.034659        0.000122
```
* most crowded tissues  
* largest nuclei cases

#### Observations

* high-density images → tightly packed nuclei
* large nuclei images → generally lower density


#### Interpretation

> A structural trade-off is observed between nucleus size and spatial packing, where larger nuclei tend to occupy less densely populated regions.

---

### All observations combined:

> Nuclei density and size exhibit a weak inverse relationship, while total tissue coverage is jointly influenced by both nucleus count and individual size. The dataset shows significant variability across samples, indicating heterogeneous structural organization and potential spatial constraints in densely packed regions.

Perfect—this is the **selling stage**.
We’ll keep it sharp, technical, and clearly position it as a **CV + LLM hybrid system** (not an LLM toy).

---

## 05. Tissue Profiling & LLM Interpretation

At this stage, the system transitions from:

```text
features → interpretation → explanation
```

This is where CytoScope stops being just a vision pipeline and becomes an **intelligent reasoning system**.

---

### Module 1 — Rule-Based Profiling Engine

Convert quantitative features into **structured, interpretable categories**
```text
numerical features → semantic labels
```

#### 1. Feature → Label Mapping

```python
def classify_density(x):
    if x <= low_density:
        return "Low Density"
    elif x >= high_density:
        return "High Density"
    else:
        return "Medium Density"
```

```python
def classify_size(x):
    if x <= small_size:
        return "Small Nuclei"
    elif x >= large_size:
        return "Large Nuclei"
    else:
        return "Medium Nuclei"
```



#### 2. Profile Composition

```python
def profile(row):
    if row["density_label"] == "High Density" and row["size_label"] == "Small Nuclei":
        return "Crowded & Small Cells"
    elif row["density_label"] == "Low Density" and row["size_label"] == "Large Nuclei":
        return "Sparse Tissue with Enlarged Cells"
    else:
        return "Moderate Tissue Structure"
```



* deterministic and explainable
* grounded in statistical thresholds
* converts raw numbers into **machine-readable reasoning units**

> This layer acts as the **logical backbone** of the system.

---

### Module 2 — LLM Interpretation Layer (Core Differentiator)

Transform structured labels into **natural language explanations**

#### Architecture

```text
features → rule-based labels → LLM → explanation
```



### Local LLM Setup

* **Runs via Ollama**
* **Model**: `gemma4:e4b`
* Fully **local inference** (no API dependency)



### Prompt Engineering (Controlled Generation)

```python
def build_prompt(row):
    return f"""
You are a biomedical image analysis assistant.

Density: {row['density_label']}
Nucleus Size: {row['size_label']}
Profile: {row['tissue_profile']}

Generate a concise structural explanation.
Avoid speculation. Be precise.
"""
```

### LLM Invocation

```python
import ollama

response = ollama.chat(
    model="gemma4:e4b",
    messages=[{"role": "user", "content": prompt}]
)
```

---

### Key Design Decisions

#### 1. LLM does NOT decide — it explains

* rule engine → controls truth
* LLM → generates interpretation
* **avoids hallucination**
* **preserves determinism**

---

#### 2. Structured Input → Controlled Output

The LLM is not given raw data—it receives:

```text
Density: High
Size: Small
Profile: Crowded & Small Cells
```

* ensures consistent
* grounded explanations

---


### Example Output

> “The tissue exhibits a densely packed arrangement of small nuclei, suggesting a compact cellular structure with limited intercellular space.”

---
