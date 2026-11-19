# Employee Workforce Segmentation

Unsupervised segmentation of employees using **PCA, K-Means and Hierarchical (Ward) clustering**, with analysis of high-risk clusters, department segments, performance groups and retention risk.

## Structure
- `src/pipeline.py` – full pipeline (data -> scaling -> PCA -> K-Means -> hierarchical -> profiling -> reports)
- `app.py` – Streamlit **Workforce Analytics Dashboard**
- `outputs/segmentation_report.md` – **Employee Segmentation Report**
- `outputs/pca_summary.md` – **PCA Summary**
- `outputs/figures/` – scree plot, elbow/silhouette, dendrogram, PCA cluster plots

## Run
```bash
pip install -r requirements.txt
python src/pipeline.py                 # uses data/employees.csv (synthetic if missing)
python src/pipeline.py my_hr_data.csv  # or your own dataset (e.g. IBM HR Analytics)
streamlit run app.py
```

## Method
1. StandardScaler on numeric features (Department excluded from clustering, analysed afterwards).
2. PCA, keeping components for >= 90% cumulative variance.
3. K-Means with K chosen by elbow + silhouette (K = 3-6); Ward hierarchical for comparison (ARI reported).
4. Attrition is NOT used for clustering; it is only used to label retention-risk tiers.

## Data note
The bundled dataset is **synthetic** (seeded, 1,500 employees). Swap in a real HR dataset with similar columns.
