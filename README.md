# BALF White Blood Cell Detection Using YOLOv12

## 1. About

This project focuses on the automated detection and classification of white blood cells (WBCs) in bronchoalveolar lavage fluid (BALF) microscopic images using YOLOv12.

The model is designed to detect and classify four major BALF cell types:

- Macrophages
- Lymphocytes
- Neutrophils
- Eosinophils

The primary objective is to identify an effective YOLO model configuration and optimize data augmentation strategies to improve cell detection performance.

Based on the YOLOv12 implementation, additional scripts were developed to automate model comparison, data augmentation experiments, performance evaluation, and detection visualization.

## 2. Key Contributions

### 2.1. Automated YOLO Model Comparison

Developed an automated pipeline to evaluate different YOLO model versions and model sizes for BALF cell detection.

- Automated training and evaluation across different YOLO model versions.
- Performance comparison across different model sizes.
- Additional scripts to handle models that could not be executed through the main automation script due to differences in model naming conventions.

### 2.2. Automated Data Augmentation Experiments

Developed an automated experimental pipeline to investigate the effects of different data augmentation techniques and combinations on object detection performance.

- Automated experiments using various combinations of data augmentation techniques.
- Performance comparison across dataset expansion scales ranging from 2× to 5×.
- Identification of effective augmentation combinations and dataset expansion strategies.

### 2.3. Experimental Analysis and Visualization

Developed additional scripts for analyzing experimental results and visualizing the final detection performance.

- Extraction of the top five augmentation combinations based on mAP50.
- Cell detection and visualization using the selected model weights.
- Generation of graphs for experimental analysis and research paper figures.

## 3. Repository Structure

The following scripts were additionally developed for automated experimentation and result analysis.

```text
.
├── yolov12/
│   ├── yolo_metric.py
│   ├── yolo_per_version.py
│   ├── additialnal_yolo_version.py
│   └── additialnal_yolov12x.py
│
├── top5_map50.py
├── detect_yolov12l_simple.py
└── make_graph.ipynb
```

### 3.1. Additional Scripts Inside the YOLOv12 Repository

| File | Description |
|---|---|
| `yolo_metric.py` | Automates training and evaluation across different data augmentation combinations and expansion scales to identify effective augmentation strategies. |
| `yolo_per_version.py` | Automates training and evaluation across different YOLO model versions and sizes to compare model performance. |
| `additialnal_yolo_version.py` | Runs additional YOLO model configurations that could not be executed through the main automation script due to differences in model naming conventions, including capitalization and size-specific naming. |
| `additialnal_yolov12x.py` | Performs additional experiments with YOLOv12x, which requires separate handling due to its model naming convention. |

### 3.2. Additional Analysis and Visualization Scripts

| File | Description |
|---|---|
| `top5_map50.py` | Analyzes the results of automated augmentation experiments and extracts the top five combinations based on mAP50. |
| `detect_yolov12l_simple.py` | Loads the selected model weights, performs cell detection, and visualizes the detection results. |
| `make_graph.ipynb` | Visualizes experimental data and generates graphs for research paper figures and result analysis. |

## 4. Experimental Workflow

The overall experimental workflow consists of the following steps:

### Step 1. YOLO Model Selection

- Automated training and evaluation across different YOLO model versions and sizes.
- Additional experiments for model configurations requiring separate handling.
- Comparison of model performance to select a suitable architecture.

### Step 2. Data Augmentation Experiments

- Automated training using various combinations of data augmentation techniques.
- Evaluation of different dataset expansion scales ranging from 2× to 5×.
- Comparison of detection performance across different augmentation configurations.

### Step 3. Performance Analysis

- Analysis of experimental results based on mAP50.
- Extraction of the top five augmentation combinations.
- Identification of the best-performing augmentation strategy.

### Step 4. Final Detection and Visualization

- Selection of the best-performing model weights.
- Cell detection using the selected weights.
- Visualization of predicted bounding boxes and cell classifications.

### Step 5. Research Result Visualization

- Processing and analysis of experimental results.
- Generation of graphs for quantitative comparison and research paper figures.

## 5. Experimental Results

The experiments demonstrated that detection performance varied depending on the combination of augmentation techniques and dataset expansion scale.

Key findings include:

- Achieved a **mAP50 of 0.893** with the best-performing selective augmentation combination.
- Selective augmentation outperformed the configuration that applied all available augmentation techniques.
- Excessive data augmentation did not necessarily improve detection performance.
- **2× data augmentation** was identified as the most effective dataset expansion strategy among the evaluated scales.

These findings suggest that selecting augmentation techniques according to dataset characteristics is more effective than indiscriminately applying all available augmentation methods.

## 6. Pre-trained Weights

The final trained model weights are provided through Google Drive due to their large file size. <br>
(The weight has not been uploaded yet, but we will upload it if needed.)

| Item | Description |
|---|---|
| Model | YOLOv12l |
| Weight File | `best.pt` |
| Download | [Google Drive - Download Weights](GOOGLE_DRIVE_LINK) |

### Download Instructions

1. Access the Google Drive link above.
2. Download the model weight file (`.pt`).
3. Place the downloaded file in the appropriate directory.
4. Run `detect_yolov12l_simple.py` to perform cell detection and visualize the results.

## 7. Technologies

- Python
- PyTorch
- Ultralytics
- YOLOv12
- OpenCV
- Jupyter Notebook

## 8. Framework and Implementation

This project utilizes the [Ultralytics](https://github.com/ultralytics/ultralytics) framework for YOLO-based model training and inference.

The original YOLOv12 implementation was used as the foundation, with additional scripts independently developed for automated model comparison, data augmentation experiments, performance analysis, and result visualization.

- Framework: [Ultralytics](https://github.com/ultralytics/ultralytics)
- Model: YOLOv12

## 9. Associated Publication

This project was conducted as part of the research presented in the following publication:

**Automated detection and classification of bronchoalveolar lavage fluid cells using YOLOv12: Optimization of data augmentation**

The repository contains additional scripts developed for model comparison, automated data augmentation experiments, and experimental result visualization.
