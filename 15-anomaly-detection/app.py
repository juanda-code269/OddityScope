import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score,recall_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

st.set_page_config(page_title="Outlier Explorer",page_icon="🔎",layout="wide")
st.title("Outlier Explorer — unsupervised anomaly detection")
st.warning("An unusual observation is not automatically fraud, error, danger, or wrongdoing.")


@st.cache_data
def demo(n=1800):
    rng=np.random.default_rng(36);normal=rng.multivariate_normal([50,30,12,80],[[100,55,5,20],[55,80,3,18],[5,3,9,4],[20,18,4,120]],n)
    anomalies=rng.uniform([5,1,0,10],[120,95,45,170],45);x=np.vstack([normal,anomalies]);d=pd.DataFrame(x,columns=["volume","frequency","error_rate","latency"]);d["known_synthetic_anomaly"]=[0]*n+[1]*len(anomalies);return d


upload=st.sidebar.file_uploader("Numeric CSV",type="csv")
try:
    df=pd.read_csv(upload) if upload else demo();source=f"Uploaded data: {upload.name}" if upload else "Synthetic operational data with known injected anomalies"
    known=next((c for c in df if c.lower() in {"known_synthetic_anomaly","anomaly","label","target"}),None)
    features=[c for c in df.select_dtypes(include=np.number) if c!=known]
    if len(df)<50 or len(features)<2:raise ValueError("Need 50 rows and at least two numeric feature columns.")
    clean=df[features].replace([np.inf,-np.inf],np.nan).fillna(df[features].median())
except Exception as exc:st.error(str(exc));st.stop()
contamination=st.sidebar.slider("Flagged fraction",.01,.25,.05,.01);scaler=RobustScaler();z=scaler.fit_transform(clean)
iso=IsolationForest(n_estimators=300,contamination=contamination,random_state=42).fit(z);iso_score=-iso.score_samples(z);iso_flag=iso.predict(z)==-1
lof=LocalOutlierFactor(n_neighbors=min(35,len(df)-1),contamination=contamination);lof_flag=lof.fit_predict(z)==-1;lof_score=-lof.negative_outlier_factor_
result=df.copy();result["Isolation score"]=iso_score;result["LOF score"]=lof_score;result["Isolation flagged"]=iso_flag;result["LOF flagged"]=lof_flag;result["Agreement"]=iso_flag&lof_flag
tabs=st.tabs(["Overview","Projection","Most unusual","Compare methods","Explanation & limits"])
with tabs[0]:
    st.info(source);a,b,c=st.columns(3);a.metric("Rows",f"{len(df):,}");b.metric("Isolation flags",int(iso_flag.sum()));c.metric("Both methods flag",int((iso_flag&lof_flag).sum()))
    feature=st.selectbox("Inspect feature",features);st.plotly_chart(px.histogram(result,x=feature,color="Isolation flagged",marginal="box",barmode="overlay"),width="stretch")
with tabs[1]:
    pca=PCA(2).fit_transform(z);projection=pd.DataFrame({"PC1":pca[:,0],"PC2":pca[:,1],"Flagged":np.where(iso_flag,"Flagged","Typical"),"Score":iso_score})
    st.plotly_chart(px.scatter(projection,x="PC1",y="PC2",color="Flagged",size="Score",opacity=.65,title="PCA projection"),width="stretch")
    st.caption(f"The two components preserve {PCA(2).fit(z).explained_variance_ratio_.sum():.1%} of scaled variance. Projection can hide or distort high-dimensional relationships.")
with tabs[2]:
    method=st.selectbox("Rank by",["Isolation score","LOF score"]);st.dataframe(result.nlargest(40,method)[features+[method,"Isolation flagged","LOF flagged"]],width="stretch")
with tabs[3]:
    agreement=pd.crosstab(result["Isolation flagged"],result["LOF flagged"],rownames=["Isolation"],colnames=["LOF"]);st.dataframe(agreement,width="stretch")
    if known:
        truth=df[known].astype(bool);metrics=pd.DataFrame({"Method":["Isolation Forest","LOF"],"Precision":[precision_score(truth,iso_flag,zero_division=0),precision_score(truth,lof_flag,zero_division=0)],"Recall":[recall_score(truth,iso_flag),recall_score(truth,lof_flag)]});st.dataframe(metrics.style.format({"Precision":"{:.1%}","Recall":"{:.1%}"}),width="stretch")
with tabs[4]:
    selected=st.number_input("Explain row index",0,len(df)-1,int(np.argmax(iso_score)));row_z=np.abs(z[int(selected)]);explain=pd.DataFrame({"Feature":features,"Robust distance magnitude":row_z}).sort_values("Robust distance magnitude")
    st.plotly_chart(px.bar(explain.tail(12),x="Robust distance magnitude",y="Feature",orientation="h",title="Features farthest from the scaled center"),width="stretch")
    st.markdown("""Isolation Forest scores points by how quickly random splits isolate them; LOF compares local density with neighboring points. Scaling, feature choice, contamination, correlated variables, neighborhood size, and high dimensionality can radically change results. Robust distance is a simple explanation aid, not an exact decomposition of either algorithm. Review flagged records with domain experts before drawing conclusions.""")

