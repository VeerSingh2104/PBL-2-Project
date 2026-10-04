# XAIEvalAgent — Phase 1

XAIEvalAgent is a proposed automated workflow for selecting, evaluating, comparing, ranking, and recommending Explainable AI (XAI) methods. This Phase 1 implementation focuses on an interactive frontend, dataset profiling, explainer applicability guidance, a transparent formula reference, and an interactive tabular preprocessing workflow and a model-backed SHAP experiment page. **No machine-learning model is trained or executed in this phase.**

## Phase 1 capabilities
- Streamlit workspace with Overview, Dataset Profiler, Metric Lab, Explainer Selector, and System Blueprint sections.
- Reference schemas from the PBL report: UCI Heart Failure Clinical Records (ID 519) and Statlog German Credit (ID 144).
- Upload and inspect tabular CSVs: schema, types, missing values, unique counts, descriptive statistics, preview, and simple feature distributions.\n- Preprocess CSVs: optional duplicate removal, missing-target row removal, train/test split, training-only numeric imputation/scaling, categorical imputation and one-hot encoding, audit summary, and CSV/JSON downloads.
- Rule-based candidate guidance for SHAP, LIME, Anchors, Integrated Gradients, and Grad-CAM based on user-selected modality/model access.
- SHAP explainer page: trains a baseline Random Forest on the preprocessed training split, generates SHAP attributions on held-out rows, and calculates SHAP runtime, a perturbation-based faithfulness estimate, and Gini sparseness.
- Interactive, explicitly illustrative min-max normalization and weighted XAIScore demonstration; non-SHAP methods and the full six-metric benchmark remain future work.

## Run locally
Python 3.10+ recommended.

```bash
git clone https://github.com/VeerSingh2104/PBL-2-Project.git
cd PBL-2-Project
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Phase boundaries
**Included:** frontend, configuration inputs, static reference-dataset feature dictionaries, live profiling and preprocessing of uploaded CSVs, transparent explainer routing guidance, metric/formula documentation, and a score calculator using fixed illustrative data.

**Included after SHAP integration:** a baseline Random Forest and TreeSHAP workflow for user-uploaded preprocessed tabular data, with local/global attributions and limited computed diagnostics. **Not included:** a multi-explainer benchmark, all six metric implementations, domain validation, validated model performance, or evidence-based automated recommendation. The weighted score example remains illustrative and must not be reported as an experimental result.

## Proposed evaluation dimensions
| Metric | Dimension | Direction |
|---|---|---|
| Faithfulness Correlation | Fidelity | Higher |
| Max-Sensitivity | Robustness | Lower |
| Local Lipschitz Estimate | Stability | Lower |
| Cross-Explainer Agreement | Consistency | Higher |
| Sparseness (Gini Index) | Interpretability proxy | Higher |
| Runtime | Computational cost | Lower |

Metric definitions and exact evaluation protocols should be validated and finalized before Phase 2 experiments. Normalized scores are candidate-set-relative and should be shown alongside raw metric values.

## Technology
Python, Streamlit, Pandas, NumPy. XAI/model libraries are intentionally deferred until model-backed phases.
