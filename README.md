# XAIEvalAgent — Phase 1

XAIEvalAgent is a proposed automated workflow for selecting, evaluating, comparing, ranking, and recommending Explainable AI (XAI) methods. This Phase 1 implementation focuses on an interactive frontend, dataset profiling, explainer applicability guidance, and a transparent formula reference. **No machine-learning model is trained or executed in this phase.**

## Phase 1 capabilities
- Streamlit workspace with Overview, Dataset Profiler, Metric Lab, Explainer Selector, and System Blueprint sections.
- Reference schemas from the PBL report: UCI Heart Failure Clinical Records (ID 519) and Statlog German Credit (ID 144).
- Upload and inspect tabular CSVs: schema, types, missing values, unique counts, descriptive statistics, preview, and simple feature distributions.
- Rule-based candidate guidance for SHAP, LIME, Anchors, Integrated Gradients, and Grad-CAM based on user-selected modality/model access.
- Explanations of six proposed evaluation metrics, formula, direction, range, interpretation, and caveats.
- Interactive, explicitly illustrative min-max normalization and weighted XAIScore demonstration.

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
**Included:** frontend, configuration inputs, static reference-dataset feature dictionaries, live profiling of uploaded CSVs, transparent explainer routing guidance, metric/formula documentation, and a score calculator using fixed illustrative data.

**Not included:** loading or training predictive models, generating real explanations, computing real XAI evaluation metrics, validating recommendations experimentally, or claiming that any explainer is empirically best. The score example is only a demonstration of the aggregation interface.

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
