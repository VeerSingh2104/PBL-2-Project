import io
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from xai_agent import profile_data, train_models, explain_instance, evaluate_attributions, rank_explainers, METRIC_DIRECTIONS

st.set_page_config(page_title="XAIEvalAgent", page_icon="🔎", layout="wide")
st.title("XAIEvalAgent")
st.caption("Automated XAI method selection, multi-metric evaluation, comparison and recommendation")
st.info("Research prototype based on the PBL-2 report. Rankings are conditional on the selected data, model, metrics and weights—not a universal explainer ranking.")

@st.cache_data(show_spinner=False)
def load_uci(dataset_id):
    from ucimlrepo import fetch_ucirepo
    data = fetch_ucirepo(id=dataset_id)
    X = data.data.features.copy()
    y = data.data.targets.copy()
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    X.columns = [str(c).strip().replace(" ", "_") for c in X.columns]
    return X.assign(__target__=y.astype(str).values)

with st.sidebar:
    st.header("1 · Dataset")
    source = st.radio("Data source", ["Upload CSV", "Heart Failure (UCI 519)", "German Credit (UCI 144)", "Built-in demo"])
    uploaded = st.file_uploader("Choose CSV", type=["csv"]) if source == "Upload CSV" else None
    if source == "Heart Failure (UCI 519)":
        st.caption("299 patient records · 12 clinical features · DEATH_EVENT target")
    elif source == "German Credit (UCI 144)":
        st.caption("1,000 applicants · 20 attributes · credit risk target")
    st.divider()
    st.header("2 · Evaluation")
    test_size = st.slider("Test split", .15, .4, .25, .05)
    methods = st.multiselect("Explainers", ["SHAP", "LIME"], default=["SHAP", "LIME"])
    st.caption("Anchors, Integrated Gradients and Grad-CAM are extension candidates; this initial tabular version does not yet execute them.")
    st.divider()
    st.header("3 · Metric weights")
    weights = {}
    defaults = {"Faithfulness Correlation":1,"Max-Sensitivity":1,"Local Lipschitz Estimate":1,"Cross-Explainer Agreement":1,"Sparseness (Gini)":1,"Runtime (s)":1}
    for metric in METRIC_DIRECTIONS:
        weights[metric] = st.slider(metric, 0.0, 3.0, 1.0, .25, key="w_"+metric)

try:
    if source == "Upload CSV":
        if uploaded is None: st.warning("Upload a CSV to begin."); st.stop()
        df = pd.read_csv(uploaded)
    elif source == "Heart Failure (UCI 519)":
        df = load_uci(519)
    elif source == "German Credit (UCI 144)":
        df = load_uci(144)
    else:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(as_frame=True)
        df = d.frame.rename(columns={"target":"__target__"})
        df["__target__"] = df["__target__"].map({0:"malignant",1:"benign"})
except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

if df.empty or len(df.columns) < 2:
    st.error("Dataset must contain at least one feature and one target column."); st.stop()
st.subheader("Dataset overview")
target_default = "__target__" if "__target__" in df.columns else str(df.columns[-1])
target = st.selectbox("Target column", list(df.columns), index=list(df.columns).index(target_default))
profile = profile_data(df, target)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Rows", f"{profile['rows']:,}")
c2.metric("Features", profile["features"])
c3.metric("Numeric / categorical", f"{profile['numeric_features']} / {profile['categorical_features']}")
c4.metric("Missing values", profile["missing_values"])
with st.expander("Preview dataset"):
    st.dataframe(df.head(20), use_container_width=True)
st.write("**Detected profile:**", f"{profile['task']} · {profile['modality']} · {len(profile['classes'])} classes")
if len(profile["classes"]) < 2:
    st.error("The selected target must have at least two classes."); st.stop()

if st.button("Train models & evaluate explainers", type="primary", disabled=not methods):
    try:
        with st.spinner("Training models and evaluating candidate explainers…"):
            models = train_models(df, target, test_size)
            result_sets = {}
            for model_name, model in models.items():
                instance = model.X_test.iloc[[0]].copy()
                attrs, runtimes = explain_instance(model, instance, methods)
                if not attrs:
                    st.warning(f"No explanations were generated for {model_name}.")
                    continue
                metrics = evaluate_attributions(model, instance, attrs, runtimes)
                result_sets[model_name] = rank_explainers(metrics, weights)
        st.session_state["results"] = result_sets
        st.session_state["models"] = {k: {"accuracy":v.accuracy} for k,v in models.items()}
    except Exception as e:
        st.exception(e)

results = st.session_state.get("results", {})
if results:
    st.subheader("Evaluation results")
    for model_name, ranked in results.items():
        st.markdown(f"### {model_name}")
        st.caption(f"Test accuracy (context only): {st.session_state['models'][model_name]['accuracy']:.3f}")
        best = ranked.iloc[0]
        st.success(f"Recommended explainer for this run: {best['Explainer']} · XAIScore {best['XAIScore']:.3f}")
        st.write("Recommendation is derived from the displayed metrics and current weights.")
        visible = [c for c in ranked.columns if c != "Attributions"]
        st.dataframe(ranked[visible].style.format(precision=4), use_container_width=True)
        fig, ax = plt.subplots(figsize=(8,3))
        ax.bar(ranked["Explainer"], ranked["XAIScore"])
        ax.set_ylabel("XAIScore (0–1)")
        ax.set_ylim(0,1)
        ax.set_title("Explainer comparison")
        st.pyplot(fig)
        with st.expander("Feature attribution details"):
            for _, row in ranked.iterrows():
                vals = pd.DataFrame({"Feature":df.drop(columns=[target]).columns,"Attribution":row["Attributions"]})
                vals["Absolute importance"] = vals["Attribution"].abs()
                st.markdown(f"**{row['Explainer']}**")
                st.dataframe(vals.sort_values("Absolute importance",ascending=False).drop(columns="Absolute importance"), use_container_width=True)
        csv = ranked[visible].to_csv(index=False).encode()
        st.download_button(f"Download {model_name} metrics (CSV)", csv, file_name=f"{model_name.lower().replace(' ','_')}_metrics.csv", mime="text/csv", key="csv_"+model_name)
    payload = {name: ranked.drop(columns=["Attributions"]).to_dict(orient="records") for name,ranked in results.items()}
    st.download_button("Download complete evaluation (JSON)", json.dumps(payload,indent=2,default=float), file_name="xaievalagent_report.json", mime="application/json")
else:
    st.markdown("### How to use")
    st.markdown("1. Load one of the report datasets or upload a tabular CSV.\n2. Select the target column.\n3. Choose candidate explainers and metric weights.\n4. Run the evaluation and inspect the ranking, metric table and report exports.")
