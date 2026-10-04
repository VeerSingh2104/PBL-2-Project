# XAIEvalAgent

An automated Explainable AI evaluation prototype based on the PBL-2 report. It profiles tabular classification data, trains comparable models, generates candidate explanations, evaluates them on multiple dimensions, ranks candidates with a configurable weighted score, and produces a transparent recommendation.

## Scope
- Domains in the report: UCI Heart Failure Clinical Records (ID 519) and Statlog German Credit (ID 144).
- Upload a CSV and choose its target column, or load either UCI dataset from the application.
- Train Random Forest and a feed-forward neural network (MLP) on the same split.
- Candidate explainers: SHAP and LIME. Anchors is listed as an optional extension and is surfaced only when installed and compatible. Integrated Gradients and Grad-CAM are not yet implemented in this tabular prototype.
- Metrics: Faithfulness Correlation, Max-Sensitivity, Local Lipschitz Estimate, Cross-Explainer Agreement, Sparseness (Gini), and runtime.
- Min-max normalization with direction correction; configurable metric weights; metric values remain visible beside XAIScore.

## Run locally
Python 3.10+ recommended.

```bash
git clone https://github.com/VeerSingh2104/PBL-2-Project.git
cd PBL-2-Project
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Workflow
1. Load a report dataset or upload a tabular CSV.
2. Select the target and classification task.
3. Train Random Forest and MLP with a stratified train/test split.
4. Select a test instance and generate SHAP/LIME explanations.
5. Evaluate explanations under the same model and instance set.
6. Review normalized metrics, weighted score, ranking, and recommendation.
7. Export results as CSV/JSON.

## Metric interpretation
Higher is better for Faithfulness Correlation, Cross-Explainer Agreement, and Sparseness. Lower is better for Max-Sensitivity, Local Lipschitz Estimate, and Runtime. Sparseness measures compactness only; it does not prove correctness. Agreement is relative to the candidate explainers included in the same run. Normalized scores are comparable only within that run.

## Important limitations
This is a research prototype, not a clinical or lending decision system. The recommendation is conditional on the selected model, data, instance, metric definitions, and weights; it is not a universal “best explainer” claim. Dataset access requires internet when using the UCI loader. Small datasets and expensive model-agnostic explainers can require additional runtime.

## Project structure
- `app.py` — Streamlit interface
- `xai_agent.py` — profiling, model training, explanations, metrics, scoring
- `requirements.txt` — dependencies
- `tests/test_metrics.py` — metric sanity checks
