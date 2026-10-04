import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import prepare_tabular_data
from shap_explainer import run_shap_experiment

st.set_page_config(page_title="XAIEvalAgent | Phase 1", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#0b1020;color:#e8edf8}
[data-testid="stSidebar"]{background:#10172a;border-right:1px solid #202b43}
[data-testid="stMetric"]{background:#111a2e;border:1px solid #26334e;padding:16px 18px;border-radius:14px}
[data-testid="stMetricLabel"]{color:#9ba9c5}
[data-testid="stMetricValue"]{color:#f2f5ff}
div[data-testid="stVerticalBlockBorderWrapper"]{border-color:#27344f!important;border-radius:14px}
.hero{padding:28px 30px;border:1px solid #293956;border-radius:20px;background:radial-gradient(ellipse at 85% 0%,#20335a 0%,#131e36 42%,#10182b 100%);margin-bottom:18px}
.eyebrow{font-family:'DM Mono',monospace;color:#82a9ff;font-size:12px;letter-spacing:2px;text-transform:uppercase}
.hero h1{font-size:38px;line-height:1.15;margin:9px 0;color:#f3f6ff;font-weight:800}
.hero p{font-size:15px;color:#b4c1d9;max-width:800px}
.pill{display:inline-block;border:1px solid #35507d;background:#172747;color:#a9c6ff;border-radius:99px;padding:5px 10px;font-size:11px;font-family:'DM Mono',monospace;margin-right:6px}
.section-kicker{font-family:'DM Mono',monospace;color:#8aa5d5;font-size:11px;letter-spacing:1.5px;text-transform:uppercase}
.metric-card{padding:17px 18px;border:1px solid #27344f;border-radius:14px;background:#111a2e;height:100%}
.metric-card h3{font-size:16px;color:#f2f5ff;margin:5px 0 8px}
.metric-card p{font-size:13px;color:#aab8d2;line-height:1.55}
.formula{font-family:'DM Mono',monospace;background:#0a1223;color:#a9d8ff;border:1px solid #263752;border-radius:9px;padding:12px;font-size:13px;line-height:1.65;overflow-wrap:anywhere}
.muted{color:#9eacc6;font-size:13px}
.smallcap{font-family:'DM Mono',monospace;font-size:11px;color:#85a1d1}
div.stButton>button[kind="primary"]{background:#6288f5;border:0;color:white;border-radius:10px;font-weight:700}
.stTabs [data-baseweb="tab"]{color:#aebbd4}
hr{border-color:#26334c}
</style>
""",unsafe_allow_html=True)

DATASETS={
"Heart Failure Clinical Records":{"domain":"Healthcare","source":"UCI Machine Learning Repository · ID 519","rows":"299 records","target":"DEATH_EVENT","features":[
("age","Patient age (years)","Numerical"),("anaemia","Decrease of red blood cells or haemoglobin","Binary"),("creatinine_phosphokinase","CPK enzyme level in blood (mcg/L)","Numerical"),("diabetes","Whether the patient has diabetes","Binary"),("ejection_fraction","Percentage of blood leaving the heart per contraction","Numerical"),("high_blood_pressure","Whether the patient has hypertension","Binary"),("platelets","Platelet count (kiloplatelets/mL)","Numerical"),("serum_creatinine","Serum creatinine level (mg/dL)","Numerical"),("serum_sodium","Serum sodium level (mEq/L)","Numerical"),("sex","Sex attribute","Binary"),("smoking","Whether the patient smokes","Binary"),("time","Follow-up period (days)","Numerical"),("DEATH_EVENT","Death during follow-up (target)","Target · Binary")]},
"Statlog German Credit":{"domain":"Finance","source":"UCI Machine Learning Repository · ID 144","rows":"1,000 records","target":"Credit Risk","features":[
("checking_status","Status/balance of checking account","Categorical"),("duration","Credit duration (months)","Numerical"),("credit_history","Past credit repayment behaviour","Categorical"),("purpose","Purpose of loan","Categorical"),("credit_amount","Loan amount requested","Numerical"),("savings_status","Savings account/bonds","Categorical"),("employment","Duration of current employment","Categorical"),("installment_rate","Installment as % of disposable income","Numerical"),("personal_status_sex","Personal status and sex","Categorical"),("other_debtors","Other debtors/guarantors","Categorical"),("residence_since","Years at current residence","Numerical"),("property","Property type","Categorical"),("age","Applicant age (years)","Numerical"),("other_installment_plans","Other installment plans","Categorical"),("housing","Housing situation","Categorical"),("existing_credits","Number of existing credits at bank","Numerical"),("job","Employment/skill category","Categorical"),("people_liable","Number of dependents","Numerical"),("telephone","Registered telephone status","Categorical"),("foreign_worker","Foreign worker status","Categorical"),("Credit Risk","Good/bad credit risk (target)","Target · Binary")]}
}
METRICS=[
{"name":"Faithfulness Correlation","dimension":"Fidelity","direction":"Higher is better","range":"−1 to 1","formula":"corr( attribution_mass(S), | f(x) − f(x with S replaced by baseline) | )","meaning":"Tests whether feature importance aligns with the change in model output when features are removed or replaced. Sample feature subsets S, compare attribution mass with prediction change, then correlate.","caveat":"Depends on the baseline, feature subsets and perturbation protocol. Correlation is not proof of causal influence."},
{"name":"Max-Sensitivity","dimension":"Robustness","direction":"Lower is better","range":"0 and upward","formula":"max_{δ ∈ N(x), ||δ|| ≤ r} || Φ(x) − Φ(x + δ) ||","meaning":"Measures the largest change in the explanation under small input perturbations within a defined radius r. In practice, approximate the maximum by sampling a fixed set of perturbations.","caveat":"The perturbation radius, distance metric and sampling strategy must be fixed and reported."},
{"name":"Local Lipschitz Estimate","dimension":"Stability","direction":"Lower is better","range":"0 and upward","formula":"max_{z ∈ N(x), z ≠ x}  || Φ(x) − Φ(z) || / || x − z ||","meaning":"Measures the rate at which an explanation changes relative to the size of a nearby input change. Unlike Max-Sensitivity, it is a change-per-distance estimate.","caveat":"A neighborhood and compatible input/attribution distance functions are required; estimate depends on sampled neighbors."},
{"name":"Cross-Explainer Agreement","dimension":"Consistency","direction":"Higher is better","range":"−1 to 1","formula":"Agreement_j = mean_{k ≠ j} τ( rank(|Φ_j(x)|), rank(|Φ_k(x)|) )","meaning":"Uses Kendall’s rank correlation τ to measure whether candidate explainers agree on the relative importance of features for the same input. Average agreement against the other applicable explainers.","caveat":"Agreement is relative, not ground truth. Methods can agree and still be unfaithful; rankings need aligned feature spaces."},
{"name":"Sparseness (Gini Index)","dimension":"Interpretability proxy","direction":"Higher is better","range":"0 to 1","formula":"Gini(a) = [2 Σᵢ₌₁ⁿ i·|a|_(i)] / [n Σᵢ₌₁ⁿ |a|_(i)] − (n+1)/n","meaning":"Calculates concentration of absolute attributions, where |a|_(i) are sorted in ascending order. A higher Gini value means a smaller number of features dominate the explanation.","caveat":"Compactness is only a proxy for interpretability; it does not establish correctness or human understandability. Define the zero-attribution case as 0."},
{"name":"Runtime","dimension":"Computational cost","direction":"Lower is better","range":"0 and upward (seconds)","formula":"Runtime = (1 / M) Σⱼ₌₁ᴹ (tⱼ_end − tⱼ_start)","meaning":"Measures the average elapsed time to generate an explanation across M explained instances, using a consistent timing protocol.","caveat":"Record hardware, software, warm-up, sample count and whether preprocessing is included."}
]
st.sidebar.markdown('<div class="eyebrow">XAIEVALAGENT</div><h2 style="color:#eef3ff;margin:4px 0 0">Phase 1</h2><p class="muted">Research prototype · No model execution</p>',unsafe_allow_html=True)
page=st.sidebar.radio("WORKSPACE",["Overview","Dataset Profiler","Preprocessing Lab","SHAP Explainer","Metric Lab","Explainer Selector","System Blueprint"],label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.markdown('<span class="pill">PHASE 1</span><span class="pill">UI + FORMULAS</span>',unsafe_allow_html=True)
st.sidebar.caption("Model training and live XAI evaluation are intentionally out of scope for this phase.")

def hero(kicker,title,desc):
 st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><h1>{title}</h1><p>{desc}</p><span class="pill">DESIGN & METHODOLOGY</span><span class="pill">INTERACTIVE PROTOTYPE</span></div>',unsafe_allow_html=True)
def metric_card(m):
 st.markdown(f'<div class="metric-card"><div class="section-kicker">{m["dimension"]} · {m["direction"]}</div><h3>{m["name"]}</h3><p>{m["meaning"]}</p><div class="formula">{m["formula"]}</div><p style="margin-bottom:0"><b>Range:</b> {m["range"]}<br><span class="muted">{m["caveat"]}</span></p></div>',unsafe_allow_html=True)

if page=="Overview":
 hero("Automated explainability evaluation","From model profile to explainability insight.","A transparent workspace for identifying applicable XAI methods, defining evaluation criteria, and planning evidence-based comparison.")
 a,b,c=st.columns(3); a.metric("6","Evaluation metrics","Proposed in the report"); b.metric("5","Candidate explainers","Across tabular and deep visual models"); c.metric("2","Reference datasets","Healthcare + finance")
 st.markdown("### Phase 1 workflow")
 stages=[("01","Configure","Choose data modality, task, model access and explanation need."),("02","Profile","Extract dataset and model characteristics into a profile."),("03","Select","Apply transparent rules to identify potentially applicable explainers."),("04","Evaluate","Review proposed metrics, definitions, directions and formulas."),("05","Compare","Explore a clearly illustrative scoring example and weighting."),("06","Recommend","Understand how later evidence can support a contextual recommendation.")]
 for ix in range(0,6,3):
  cols=st.columns(3)
  for col,item in zip(cols,stages[ix:ix+3]):
   with col: st.markdown(f'<div class="metric-card"><div class="smallcap">STEP {item[0]}</div><h3>{item[1]}</h3><p>{item[2]}</p></div>',unsafe_allow_html=True)
 st.markdown("### What Phase 1 delivers")
 st.markdown("- Interactive frontend and dataset profiling for the two report datasets and user-uploaded CSV files.\n- Explainer applicability guidance based on the selected profile.\n- Feature definitions and a metric-by-metric formula reference.\n- Transparent, configurable score demonstration using illustrative values only.\n- Architecture and clear boundary between implemented interface and future model-backed evaluation.")
 st.warning("No model is trained and no real explanation or metric result is produced in Phase 1. Any example score is labelled illustrative, not an experiment result.")
elif page=="Dataset Profiler":
 hero("Input & configuration module","Dataset profiler","Inspect feature schema, data types, missingness and basic distributions without fitting a predictive model.")
 choice=st.selectbox("Reference dataset",list(DATASETS)+["Upload your own CSV"])
 if choice=="Upload your own CSV":
  uploaded=st.file_uploader("Upload a tabular CSV",type=["csv"])
  if uploaded is None: st.info("Upload a CSV to inspect its columns and descriptive statistics."); st.stop()
  try: df=pd.read_csv(uploaded)
  except Exception as e: st.error(f"Unable to read CSV: {e}"); st.stop()
  target=st.selectbox("Target column (optional for profiling)",["— Not specified —"]+df.columns.tolist())
  features=[c for c in df.columns if c!=target]
  profile=pd.DataFrame({"Feature":features,"Type":[str(df[c].dtype) for c in features],"Missing":[int(df[c].isna().sum()) for c in features],"Unique":[int(df[c].nunique(dropna=True)) for c in features]})
  domain="Custom upload"; src="User-provided CSV"; rows=len(df)
  numeric=df[features].select_dtypes(include=np.number)
  cat=df[features].select_dtypes(exclude=np.number)
 else:
  spec=DATASETS[choice]; domain=spec["domain"]; src=spec["source"]; rows=spec["rows"]; target=spec["target"]
  features=[f[0] for f in spec["features"] if not f[2].startswith("Target")]
  profile=pd.DataFrame(spec["features"],columns=["Feature","Description","Type"])
  numeric=cat=None
  st.caption(f"{src} · {spec['rows']} · target: {target}")
 a,b,c,d=st.columns(4)
 a.metric("Rows",f"{rows:,}" if isinstance(rows,int) else rows); b.metric("Features",len(features))
 b2=len(df[features].select_dtypes(include=np.number).columns) if choice=="Upload your own CSV" else sum(x[2]=="Numerical" for x in spec["features"])
 c.metric("Numeric",b2); d.metric("Categorical / binary",len(features)-b2)
 st.markdown("### Extracted feature inventory"); st.dataframe(profile,use_container_width=True,hide_index=True)
 if choice=="Upload your own CSV":
  q1,q2,q3=st.columns(3); q1.metric("Missing cells",int(df[features].isna().sum().sum())); q2.metric("Duplicate rows",int(df.duplicated().sum())); q3.metric("Target",str(target))
  with st.expander("Descriptive statistics"): st.dataframe(df[features].describe(include="all").transpose(),use_container_width=True)
  with st.expander("Data preview"): st.dataframe(df.head(25),use_container_width=True)
  with st.expander("Per-feature distribution"):
   col=st.selectbox("Choose feature",features)
   if pd.api.types.is_numeric_dtype(df[col]): st.bar_chart(df[col].dropna().value_counts(bins=20).sort_index())
   else: st.bar_chart(df[col].astype("string").fillna("Missing").value_counts().head(20))
 else: st.info("Reference schema is shown from the PBL report. Upload a local CSV to calculate live missingness, uniqueness and distributions.")
 st.markdown("### Profile extracted by Phase 1")
 st.code(f"modality: tabular\\ndomain: {domain}\\ntask: classification (reference target)\\ntarget: {target}\\nfeature_count: {len(features)}\\nmodel_loaded: false\\nprediction_interface: unavailable",language="yaml")
elif page=="Preprocessing Lab":
 hero("Data preparation module","Preprocessing lab","Clean and transform an uploaded tabular dataset, then export train/test feature matrices for a later model-backed phase.")
 st.markdown("### 1 · Upload and configure")
 upload=st.file_uploader("Upload dataset (CSV)",type=["csv"],key="prep_upload")
 if upload is None:
  st.info("Upload a CSV to begin. Preprocessing is performed on the uploaded data only; the reference dataset schemas remain available in Dataset Profiler.")
  st.stop()
 try:
  raw=pd.read_csv(upload)
 except Exception as e:
  st.error(f"Unable to read CSV: {e}"); st.stop()
 if raw.empty:
  st.error("The CSV contains no rows."); st.stop()
 raw.columns=[str(c).strip() for c in raw.columns]
 if len(set(raw.columns))!=len(raw.columns):
  st.error("Column names are duplicated after trimming whitespace."); st.stop()
 target=st.selectbox("Target column",raw.columns.tolist(),key="prep_target")
 p1,p2,p3=st.columns(3)
 with p1: test_pct=st.slider("Test split (%)",10,40,20,5)
 with p2: scaling=st.selectbox("Numeric scaling",["StandardScaler","MinMaxScaler","None"])
 with p3: dupes=st.checkbox("Remove duplicate rows",value=True)
 p4,p5=st.columns(2)
 with p4: num_imp=st.selectbox("Numeric missing values",["median","mean"],help="Imputation is fitted on the training split only.")
 with p5: cat_imp=st.selectbox("Categorical missing values",["most_frequent","constant"],help="Constant imputation fills missing categories with a dedicated placeholder.")
 random_seed=st.number_input("Random seed",min_value=0,max_value=99999,value=42,step=1)
 st.caption("Categorical features are one-hot encoded. The target is retained in its original form and is not scaled or encoded by this feature-preparation step.")
 if st.button("Run preprocessing",type="primary"):
  try:
   train,test,audit=prepare_tabular_data(raw,target,test_pct/100,scaling,num_imp,cat_imp,dupes,int(random_seed))
   st.session_state["prep_train"]=train
   st.session_state["prep_test"]=test
   st.session_state["prep_audit"]=audit
  except Exception as e:
   st.session_state.pop("prep_train",None); st.session_state.pop("prep_test",None); st.session_state.pop("prep_audit",None)
   st.error(f"Preprocessing could not be completed: {e}")
 if "prep_audit" in st.session_state:
  audit=st.session_state["prep_audit"]; train=st.session_state["prep_train"]; test=st.session_state["prep_test"]
  st.markdown("### 2 · Preprocessing audit")
  a,b,c,d=st.columns(4)
  a.metric("Rows after cleaning",audit["rows_after_cleaning"]); b.metric("Train rows",audit["train_rows"]); c.metric("Test rows",audit["test_rows"]); d.metric("Output features",audit["output_feature_count"])
  audit_df=pd.DataFrame([("Duplicate rows found",audit["duplicate_rows_found"]),("Duplicate rows removed",audit["duplicate_rows_removed"]),("Rows missing target removed",audit["rows_missing_target_removed"]),("Numeric columns",len(audit["numeric_columns"])),("Categorical columns",len(audit["categorical_columns"])),("Numeric imputation",audit["numeric_imputation"]),("Categorical imputation",audit["categorical_imputation"]),("Scaling",audit["scaling"]),("Encoding",audit["encoding"])],columns=["Step","Result"])
  st.dataframe(audit_df,use_container_width=True,hide_index=True)
  st.markdown("### 3 · Prepared output")
  t1,t2=st.tabs(["Training split","Test split"])
  with t1: st.dataframe(train.head(30),use_container_width=True); st.download_button("Download train.csv",train.to_csv(index=False).encode("utf-8"),"train.csv","text/csv")
  with t2: st.dataframe(test.head(30),use_container_width=True); st.download_button("Download test.csv",test.to_csv(index=False).encode("utf-8"),"test.csv","text/csv")
  st.download_button("Download preprocessing audit (JSON)",__import__("json").dumps(audit,indent=2,default=str).encode("utf-8"),"preprocessing_audit.json","application/json")
  st.markdown("### Processing sequence")
  st.markdown("1. Validate tabular input and selected target.\n2. Optionally remove exact duplicate rows and drop rows with missing target values.\n3. Split into training and test partitions.\n4. Fit numeric imputation/scaling and categorical imputation/one-hot encoding on training data only.\n5. Apply the fitted transformations to both partitions and export feature matrices.")
  st.warning("This prepares data; it does not train a model. For a final experiment, review domain-specific cleaning, outliers, target semantics and split strategy with your supervisor. Fit transformations only on training data to avoid leakage.")
elif page=="SHAP Explainer":
 hero("Model-backed explanation module","SHAP explainer","Train a baseline Random Forest on the preprocessed training split and generate real SHAP feature attributions for held-out records.")
 st.markdown("### Experiment input")
 if "prep_audit" not in st.session_state:
  st.info("First upload a CSV and run the Preprocessing Lab. This page uses its training and test outputs.")
  st.stop()
 audit=st.session_state["prep_audit"]; train=st.session_state["prep_train"]; test=st.session_state["prep_test"]; target=audit["target"]
 a,b=st.columns(2)
 a.metric("Training rows",len(train)); b.metric("Test rows",len(test))
 st.caption("A Random Forest baseline is trained here for the selected uploaded dataset. The target is not encoded or scaled by the preprocessing step; scikit-learn handles class labels.")
 explain_count=st.slider("Held-out instances to explain",min_value=1,max_value=min(50,len(test)),value=min(20,len(test)),step=1)
 if st.button("Train model and generate SHAP values",type="primary"):
  try:
   with st.spinner("Training Random Forest and calculating SHAP attributions..."):
    result=run_shap_experiment(train,test,target,explain_count,42)
   st.session_state["shap_result"]=result
   st.session_state["shap_target"]=target
  except Exception as e:
   st.session_state.pop("shap_result",None)
   st.error(f"SHAP experiment failed: {e}")
 if "shap_result" in st.session_state and st.session_state.get("shap_target")==target:
  result=st.session_state["shap_result"]
  st.markdown("### Calculated experiment results")
  a,b,c,d=st.columns(4)
  a.metric("Explained instances",result["explained_count"])
  b.metric("SHAP runtime",f'{result["runtime_seconds"]:.4f} s')
  c.metric("Faithfulness correlation",f'{result["faithfulness_correlation"]:.4f}' if np.isfinite(result["faithfulness_correlation"]) else "N/A")
  d.metric("Sparseness (Gini)",f'{result["sparseness_gini"]:.4f}')
  st.caption("These values are computed from the current uploaded dataset, train/test split, Random Forest and SHAP run. Runtime depends on hardware and configuration. Faithfulness is an approximate perturbation-based estimate for the first explained instance.")
  st.markdown("### Global feature importance")
  st.bar_chart(result["global_importance"].head(15).set_index("feature")["mean_abs_shap"])
  st.dataframe(result["global_importance"],use_container_width=True,hide_index=True)
  st.markdown("### Local explanation")
  instance=st.selectbox("Explained test instance",list(range(1,result["explained_count"]+1)),format_func=lambda n:f"Test instance {n}")
  local=result["local"][result["local"]["instance"]==instance].sort_values("abs_shap",ascending=False)
  st.caption(f'Predicted class: {local["predicted_class"].iloc[0]} · Predicted probability: {local["predicted_probability"].iloc[0]:.4f}')
  st.bar_chart(local.head(15).set_index("feature")["shap_value"])
  st.dataframe(local[["feature","feature_value","shap_value","abs_shap"]],use_container_width=True,hide_index=True)
  export={k:v for k,v in result.items() if k not in ["model","explainer","local","global_importance"]}
  st.download_button("Download SHAP results (JSON)",__import__("json").dumps(export,indent=2,default=str).encode("utf-8"),"shap_results.json","application/json")
  st.download_button("Download local SHAP values (CSV)",result["local"].to_csv(index=False).encode("utf-8"),"shap_local_values.csv","text/csv")
  st.warning("This is a baseline experiment, not a clinically or financially validated model. Review target meaning, class balance, preprocessing and model settings before interpreting results. The score does not establish causal influence or a universally best explainer.")
elif page=="Metric Lab":
 hero("Evaluation engine · methodology","Metric lab","Explore proposed evaluation dimensions and review the metrics now calculated by the SHAP experiment.")
 st.caption("Formula reference follows the evaluation methodology in the PBL report. Final implementation choices and metric applicability remain subject to validation.")
 for ix in range(0,len(METRICS),2):
  cols=st.columns(2)
  for col,m in zip(cols,METRICS[ix:ix+2]):
   with col: metric_card(m)
 st.markdown("### Proposed normalization")
 st.markdown("Metrics use different scales, so raw values should not be added directly. The report proposes min–max normalization within the candidate set, with direction correction so larger normalized values always indicate better performance.")
 st.latex(r"\displaystyle z_{e,k}=\frac{r_{e,k}-\min(r_{:,k})}{\max(r_{:,k})-\min(r_{:,k})}")
 st.latex(r"z^{*}_{e,k}=\begin{cases}z_{e,k},&\text{higher is better}\\1-z_{e,k},&\text{lower is better}\end{cases}")
 st.caption("If all candidates have the same value for a metric, define a consistent neutral handling policy (for example, assign 1 to all for that run). Scores are relative to the current candidate set.")
 st.markdown("### Proposed weighted XAIScore")
 st.latex(r"\mathrm{XAIScore}(e)=\frac{\sum_{k=1}^{6}w_k z^{*}_{e,k}}{\sum_{k=1}^{6}w_k},\quad w_k\geq 0,\;\sum_k w_k>0")
 st.caption("The report suggests equal weights by default and application-specific weighting as an option. Always show raw metrics alongside the aggregate score.")
 st.markdown("### Interactive score demonstration")
 st.warning("The table below remains a clearly labelled illustrative comparison. For calculated SHAP results, run the SHAP Explainer page on an uploaded dataset. Only SHAP is implemented as a model-backed explainer in this phase.")
 demo=pd.DataFrame({"Explainer":["SHAP","LIME","Anchors"],"Faithfulness Correlation":[.72,.55,.38],"Max-Sensitivity":[.18,.42,.30],"Local Lipschitz Estimate":[.95,1.60,1.25],"Cross-Explainer Agreement":[.68,.52,.41],"Sparseness (Gini)":[.61,.48,.70],"Runtime (s)":[2.40,.90,3.10]})
 st.dataframe(demo,use_container_width=True,hide_index=True); weights={}; cols=st.columns(3)
 for i,m in enumerate(METRICS):
  with cols[i%3]: weights[m["name"]]=st.slider(m["name"],0.0,3.0,1.0,.25,key="demo_"+str(i))
 score=demo.copy()
 for m in METRICS:
  n=m["name"]; v=score[n].astype(float); lo,hi=v.min(),v.max(); z=(v-lo)/(hi-lo) if hi>lo else pd.Series(np.ones(len(v)))
  score[n+" · normalized"]=z if m["direction"]=="Higher is better" else 1-z
 active=[m for m in METRICS if weights[m["name"]]>0]
 if not active: st.error("Set at least one weight above zero.")
 else:
  total=sum(weights[m["name"]] for m in active); score["Illustrative XAIScore"]=sum(score[m["name"]+" · normalized"]*weights[m["name"]]/total for m in active)
  score=score.sort_values("Illustrative XAIScore",ascending=False).reset_index(drop=True); score.insert(0,"Illustrative rank",np.arange(1,len(score)+1))
  st.dataframe(score[["Illustrative rank","Explainer","Illustrative XAIScore"]],use_container_width=True,hide_index=True)
  st.bar_chart(score.set_index("Explainer")["Illustrative XAIScore"])
  st.caption("Changing weights can change the ranking. This preview demonstrates aggregation only, not explainer performance.")
elif page=="Explainer Selector":
 hero("Rule-based applicability","Explainer selector","Explore which candidate explanation methods may fit a model/data profile. This is guidance, not a claim that an explainer has run.")
 modality=st.selectbox("Data modality",["Tabular","Image","Text"]); model=st.selectbox("Model family",["Tree-based","Neural network","Other black-box","Not decided"])
 diff=st.radio("Are gradients available?",["Yes","No","Unknown"],horizontal=True); internals=st.radio("Model internals accessible?",["Yes","No","Unknown"],horizontal=True)
 need=st.multiselect("Explanation requirement",["Local feature attribution","Global feature importance","Visual heatmap","Rule-based explanation"],default=["Local feature attribution"])
 methods={"SHAP":("Tabular, image and other supported model settings","Prediction function or supported model-specific implementation","Broad attribution framework; method and runtime depend on model."),"LIME":("Tabular, image and text","Prediction interface","Model-agnostic local surrogate explanation."),"Anchors":("Tabular, image and text (with suitable implementation)","Prediction interface","Produces local high-precision rules."),"Integrated Gradients":("Differentiable neural models","Gradient access","Gradient-based attribution; requires differentiability."),"Grad-CAM":("Deep visual models with suitable convolutional layers","Model internals and gradients","Produces a class-discriminative visual localization map.")}
 candidate=[]
 for name,(data,access,desc) in methods.items():
  eligible=(modality=="Tabular" and name in ["SHAP","LIME","Anchors"]) or (modality=="Image" and name in ["SHAP","LIME","Anchors","Integrated Gradients","Grad-CAM"]) or (modality=="Text" and name in ["SHAP","LIME","Anchors"])
  if name=="Integrated Gradients" and (model!="Neural network" or diff!="Yes"): eligible=False
  if name=="Grad-CAM" and (modality!="Image" or model!="Neural network" or diff!="Yes" or internals!="Yes"): eligible=False
  if eligible: candidate.append((name,data,access,desc))
 st.markdown("### Candidate methods")
 if candidate:
  for name,data,access,desc in candidate:
   st.markdown(f'<div class="metric-card" style="margin-bottom:10px"><div class="section-kicker">POTENTIALLY APPLICABLE</div><h3>{name}</h3><p>{desc}</p><p><b>Data:</b> {data}<br><b>Requirement:</b> {access}</p></div>',unsafe_allow_html=True)
 else: st.info("No candidates matched the current rule set. Adjust the profile or mark it as undecided.")
 st.caption("Candidate routing is a Phase 1 rule-based illustration. Exact compatibility depends on input representation, library support, model interface and implementation testing.")
 st.markdown("### Profile summary"); st.json({"modality":modality,"model_family":model,"gradients_available":diff,"internals_accessible":internals,"explanation_requirements":need,"candidate_explainers":[x[0] for x in candidate],"model_loaded":False})
elif page=="System Blueprint":
 hero("Architecture & scope","System blueprint","A staged plan aligned with the report's proposed evaluation pipeline.")
 modules=[("01","Input & configuration","Dataset, model metadata, task and application context","Structured configuration"),("02","Model/dataset profiler","Data modality, task, model family, differentiability and prediction interface","Profile dictionary"),("03","Explainer selection","Rule-based compatibility and explanation requirements","Candidate explainer list"),("04","Explanation generation","Candidate explainer + same model + relevant inputs","Attribution, rule or heatmap"),("05","Evaluation engine","Explanation outputs and standardized protocol","Metric results per explainer"),("06","Normalization & aggregation","Raw metrics, metric direction and weights","Normalized score / XAIScore"),("07","Ranking","Scores with individual metric values","Ranked candidate table"),("08","Recommendation & report","Ranking, constraints and metric evidence","Recommendation with reasoning and report")]
 for n,title,inp,out in modules:
  i=int(n); status="PHASE 1 UI" if i in [1,2,3,6,7,8] else "FUTURE INTEGRATION"
  st.markdown(f'<div class="metric-card" style="margin-bottom:9px"><div class="smallcap">MODULE {n}</div><h3>{title}</h3><p><b>Input:</b> {inp}<br><b>Output:</b> {out}</p><span class="pill">{status}</span></div>',unsafe_allow_html=True)
 st.markdown("### Phase boundaries"); left,right=st.columns(2)
 with left:
  st.markdown("#### Phase 1 · This prototype"); st.markdown("- Interactive Streamlit interface\n- Reference dataset schemas and CSV inspection\n- Model/data profile configuration\n- Rule-based candidate explainer guidance\n- Metric catalogue with formulas and caveats\n- Illustrative normalization, weights and score preview")
 with right:
  st.markdown("#### Later phases · Not active"); st.markdown("- Load/train models and generate explanations\n- Execute metrics under controlled perturbation protocols\n- Aggregate measurements over test instances\n- Validate metric definitions and applicability\n- Evidence-based recommendation from actual results\n- Persist experiment runs and export full evaluation reports")
 st.info("The report describes the system as a selection, evaluation, comparison and recommendation layer—not a new predictive model. Phase 1 makes the methodology inspectable while keeping model-backed claims for later experimentation.")
st.markdown("<hr><p class='muted' style='text-align:center'>XAIEvalAgent · PBL-2 · Phase 1 prototype · No model execution or experimental claims</p>",unsafe_allow_html=True)
