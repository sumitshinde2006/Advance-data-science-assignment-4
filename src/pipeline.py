"""Employee Workforce Segmentation: K-Means + Hierarchical + PCA.
Run: python src/pipeline.py [path/to/your.csv]
If no CSV is given/found, a synthetic HR dataset is generated (data/employees.csv)."""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, dendrogram

SEED = 42
OUT, FIG = "outputs", "outputs/figures"
DEPTS = ["Sales", "Engineering", "HR", "Finance", "Operations", "Support"]


def make_data(n=1500):
    rng = np.random.default_rng(SEED)
    seg = rng.choice(4, n, p=[.30, .25, .25, .20])  # hidden archetypes
    p = lambda a, b, c, d: np.select([seg == 0, seg == 1, seg == 2], [a, b, c], d)
    df = pd.DataFrame({
        "EmployeeID": range(1, n + 1),
        "Department": rng.choice(DEPTS, n, p=[.22, .26, .08, .12, .18, .14]),
        "Age": np.clip(rng.normal(p(41, 28, 36, 33), 5), 21, 60).round(),
        "MonthlyIncome": np.clip(rng.normal(p(9500, 3800, 6500, 5000), 1200), 2000, 20000).round(-1),
        "YearsAtCompany": np.clip(rng.normal(p(11, 2, 6, 4), 2.5), 0, 35).round(),
        "YearsSinceLastPromotion": np.clip(rng.normal(p(2, 1.5, 3, 5), 1.5), 0, 15).round(),
        "PerformanceRating": np.clip(rng.normal(p(3.8, 3.0, 4.2, 2.6), .4), 1, 5).round(1),
        "JobSatisfaction": np.clip(rng.normal(p(3.6, 3.0, 3.4, 2.0), .6), 1, 4).round(1),
        "WorkLifeBalance": np.clip(rng.normal(p(3.3, 3.0, 2.0, 2.4), .6), 1, 4).round(1),
        "OverTimeHours": np.clip(rng.normal(p(3, 4, 14, 8), 3), 0, 30).round(),
        "TrainingHours": np.clip(rng.normal(p(25, 40, 20, 8), 8), 0, 80).round(),
        "DistanceFromHome": rng.integers(1, 30, n),
    })
    z = (-3.2 + .45 * (3 - df.JobSatisfaction) + .35 * (3 - df.WorkLifeBalance)
         + .05 * df.OverTimeHours + .15 * df.YearsSinceLastPromotion
         - .00008 * df.MonthlyIncome + .01 * df.DistanceFromHome)
    df["Attrition"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def load(path):
    if path and os.path.exists(path):
        return pd.read_csv(path)
    p = "data/employees.csv"
    if not os.path.exists(p):
        make_data().to_csv(p, index=False)
    return pd.read_csv(p)


def main():
    os.makedirs(FIG, exist_ok=True)
    df = load(sys.argv[1] if len(sys.argv) > 1 else None)
    num = [c for c in df.select_dtypes("number").columns if c not in ("EmployeeID", "Attrition")]
    X = df[num].astype(float)  # Department kept out of clustering so it can be analysed afterwards
    Xs = StandardScaler().fit_transform(X)

    # ---- PCA
    pca_full = PCA(random_state=SEED).fit(Xs)
    cum = pca_full.explained_variance_ratio_.cumsum()
    n_comp = int(np.argmax(cum >= 0.90) + 1)
    pca = PCA(n_components=n_comp, random_state=SEED)
    Z = pca.fit_transform(Xs)
    plt.figure(figsize=(7, 4))
    k = range(1, len(cum) + 1)
    plt.bar(k, pca_full.explained_variance_ratio_, alpha=.6, label="Individual")
    plt.plot(k, cum, "r-o", label="Cumulative")
    plt.axhline(.9, ls="--", c="gray")
    plt.axvline(n_comp, ls="--", c="green")
    plt.xlabel("Component"); plt.ylabel("Explained variance"); plt.title("PCA Scree Plot")
    plt.legend(); plt.tight_layout(); plt.savefig(f"{FIG}/pca_scree.png", dpi=130); plt.close()
    ld = pd.DataFrame(pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"])
    ld.loc[ld.abs().max(axis=1).sort_values(ascending=False).index].to_csv(f"{OUT}/pca_loadings.csv")

    # ---- K-Means: elbow + silhouette
    ks = list(range(3, 7))  # business-friendly range
    wcss, sil = [], []
    for kk in ks:
        km = KMeans(kk, n_init=10, random_state=SEED).fit(Z)
        wcss.append(km.inertia_)
        sil.append(silhouette_score(Z, km.labels_))
    best_k = ks[int(np.argmax(sil))]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(ks, wcss, "o-"); ax[0].set_title("Elbow Method (WCSS)"); ax[0].set_xlabel("K")
    ax[1].plot(ks, sil, "o-g"); ax[1].set_title("Silhouette Score"); ax[1].set_xlabel("K")
    plt.tight_layout(); plt.savefig(f"{FIG}/kmeans_k_selection.png", dpi=130); plt.close()
    km = KMeans(best_k, n_init=10, random_state=SEED).fit(Z)
    df["KMeans_Cluster"] = km.labels_

    # ---- Hierarchical (Ward)
    sample = np.random.default_rng(SEED).choice(len(Z), 150, replace=False)
    plt.figure(figsize=(10, 4))
    dendrogram(linkage(Z[sample], "ward"), no_labels=True)
    plt.title("Dendrogram (Ward, 150-employee sample)"); plt.ylabel("Distance"); plt.tight_layout()
    plt.savefig(f"{FIG}/dendrogram.png", dpi=130); plt.close()
    hc = AgglomerativeClustering(best_k, linkage="ward").fit(Z)
    df["Hier_Cluster"] = hc.labels_
    ari = adjusted_rand_score(df.KMeans_Cluster, df.Hier_Cluster)
    sil_h = silhouette_score(Z, hc.labels_)

    # ---- PCA scatter
    df["PC1"], df["PC2"] = Z[:, 0], Z[:, 1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for a, col, t in zip(ax, ["KMeans_Cluster", "Hier_Cluster"], ["K-Means", "Hierarchical (Ward)"]):
        a.scatter(df.PC1, df.PC2, c=df[col], cmap="tab10", s=8, alpha=.7)
        a.set_title(t)
        a.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
        a.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    plt.tight_layout(); plt.savefig(f"{FIG}/pca_clusters.png", dpi=130); plt.close()

    # ---- Profiling + risk (Attrition is used only to validate/label risk, not to cluster)
    g = df.groupby("KMeans_Cluster")
    prof = g.agg(Size=("EmployeeID", "count"), Age=("Age", "mean"), Income=("MonthlyIncome", "mean"),
                 Tenure=("YearsAtCompany", "mean"), YrsSincePromo=("YearsSinceLastPromotion", "mean"),
                 Performance=("PerformanceRating", "mean"), JobSat=("JobSatisfaction", "mean"),
                 WorkLife=("WorkLifeBalance", "mean"), OT_Hours=("OverTimeHours", "mean"),
                 Training=("TrainingHours", "mean"), AttritionRate=("Attrition", "mean")).round(2)
    r = prof.AttritionRate.rank(method="first")
    prof["RiskTier"] = np.where(r > len(prof) * 2 / 3, "High", np.where(r > len(prof) / 3, "Medium", "Low"))

    def persona(row):
        if row.RiskTier == "High" and row.Performance < prof.Performance.median():
            return "Disengaged & At-Risk"
        if row.OT_Hours > prof.OT_Hours.median() and row.Performance >= prof.Performance.median():
            return "Burnout-Prone High Performers"
        if row.Tenure > prof.Tenure.median():
            return "Stable Veterans"
        return "Early-Career Learners"
    prof["Persona"] = prof.apply(persona, axis=1)
    prof.to_csv(f"{OUT}/cluster_profiles.csv")
    df["Persona"] = df.KMeans_Cluster.map(prof.Persona)
    df["RiskTier"] = df.KMeans_Cluster.map(prof.RiskTier)
    dept = pd.crosstab(df.Department, df.Persona, normalize="index").round(3)
    dept.to_csv(f"{OUT}/department_segments.csv")
    df["PerfGroup"] = pd.cut(df.PerformanceRating, [0, 2.9, 3.9, 5], labels=["Low", "Core", "High"])
    perf = pd.crosstab(df.Persona, df.PerfGroup, normalize="index").round(3)
    df.to_csv(f"{OUT}/employees_clustered.csv", index=False)

    # ---- Reports
    md = lambda t: t.to_markdown()
    open(f"{OUT}/pca_summary.md", "w").write(f"""# PCA Summary
- Features used: {X.shape[1]} (numeric only; Department analysed after clustering), standardised with StandardScaler.
- Components kept: **{n_comp}** (cumulative variance {cum[n_comp-1]:.1%}; threshold 90%).
- PC1 / PC2 explain {pca.explained_variance_ratio_[0]:.1%} / {pca.explained_variance_ratio_[1]:.1%}.

Top loadings (full table in `pca_loadings.csv`):

{md(pd.read_csv(f"{OUT}/pca_loadings.csv", index_col=0).head(8).round(2))}

![scree](figures/pca_scree.png)
""")
    open(f"{OUT}/segmentation_report.md", "w").write(f"""# Employee Segmentation Report
**Data:** {len(df)} employees, {len(num)} numeric features (Department profiled after clustering).
**Method:** StandardScaler -> PCA ({n_comp} components) -> K-Means and Ward hierarchical clustering.

## Choosing K
Silhouette peaked at K={best_k} (score {max(sil):.3f}); elbow plot in `figures/kmeans_k_selection.png`.
Hierarchical (Ward, K={best_k}) silhouette {sil_h:.3f}; agreement with K-Means (ARI) = {ari:.3f}.

## Cluster profiles (K-Means)
{md(prof)}

## High-risk / retention-risk clusters
{md(prof[prof.RiskTier == "High"][["Persona", "Size", "AttritionRate", "JobSat", "WorkLife", "OT_Hours", "YrsSincePromo"]])}

## Department-level segments (share of each department)
{md(dept)}

## Performance groups by persona (share of persona)
{md(perf)}

## Recommendations (edit using your own reading of the numbers)
1. Target the high-risk persona with stay interviews, promotion review and workload rebalancing.
2. Cap overtime for burnout-prone high performers; they are costly to lose.
3. Invest in training and career paths for early-career learners to build tenure.

![clusters](figures/pca_clusters.png)
![dendrogram](figures/dendrogram.png)
""")
    print(f"K={best_k}, PCA comps={n_comp}, silK={max(sil):.3f}, silH={sil_h:.3f}, ARI={ari:.3f}")
    print(prof.to_string())


if __name__ == "__main__":
    main()
