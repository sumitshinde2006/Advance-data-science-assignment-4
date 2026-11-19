"""Workforce Analytics Dashboard. Run: streamlit run app.py (after src/pipeline.py)"""
import pandas as pd, plotly.express as px, streamlit as st
st.set_page_config(page_title="Workforce Analytics", layout="wide")
df = pd.read_csv("outputs/employees_clustered.csv"); prof = pd.read_csv("outputs/cluster_profiles.csv")
st.title("Workforce Analytics Dashboard")
depts = st.sidebar.multiselect("Department", sorted(df.Department.unique()), default=sorted(df.Department.unique()))
risk = st.sidebar.multiselect("Risk tier", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
d = df[df.Department.isin(depts) & df.RiskTier.isin(risk)]
c = st.columns(4)
c[0].metric("Employees", len(d)); c[1].metric("Attrition rate", f"{d.Attrition.mean():.1%}")
c[2].metric("Avg performance", f"{d.PerformanceRating.mean():.2f}"); c[3].metric("High-risk share", f"{(d.RiskTier=='High').mean():.1%}")
a, b = st.columns(2)
a.plotly_chart(px.scatter(d, x="PC1", y="PC2", color="Persona", title="PCA view of segments", opacity=.7), use_container_width=True)
b.plotly_chart(px.histogram(d, x="Department", color="Persona", barmode="stack", title="Segments by department"), use_container_width=True)
a, b = st.columns(2)
a.plotly_chart(px.bar(d.groupby("Persona").Attrition.mean().reset_index(), x="Persona", y="Attrition", title="Retention risk by segment"), use_container_width=True)
b.plotly_chart(px.box(d, x="Persona", y="PerformanceRating", title="Performance groups"), use_container_width=True)
st.subheader("Cluster profiles"); st.dataframe(prof)
