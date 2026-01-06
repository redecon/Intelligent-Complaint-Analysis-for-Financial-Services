# Intelligent Complaint Analysis for Financial Services

A repository for scalable, explainable complaint classification and analysis for financial services. Combines data ingestion, preprocessing, model training (NLP + tabular), evaluation, and a lightweight inference API.

## Features
- Complaint ingestion and cleansing pipeline
- Text classification (intent / product / routing) with explainability (SHAP/LIME)
- Named entity recognition and key-phrase extraction
- Dashboard-ready outputs and evaluation metrics
- REST inference endpoint for batch/real-time scoring

## Quick start

Prerequisites:
- Python 3.9+
- git, virtualenv or venv
- (Optional) CUDA for GPU training

Clone and prepare environment:
```bash
git clone <repo-url>
cd Intelligent-Complaint-Analysis-for-Financial-Services
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Unix
source .venv/bin/activate
pip install -r requirements.txt
```

Run data preparation, training and evaluation (examples):
```bash
python -m scripts.prepare_data --input data/raw --output data/processed
python -m scripts.train --config configs/train.yaml
python -m scripts.evaluate --model artifacts/model.pt --data data/processed/test.csv
```

Start API:
```bash
uvicorn api.app:app --reload --port 8000
# then POST to http://localhost:8000/predict
```

## Project layout (recommended)
- data/               # raw and processed datasets
- notebooks/          # exploratory analysis and demos
- src/
    - ingestion/        # data loaders
    - preprocessing/    # cleaning & feature engineering
    - models/           # model definitions and trainers
    - explainability/   # SHAP/LIME wrappers
    - api/              # FastAPI/Flask inference app
- scripts/            # CLI utilities (prepare_data, train, evaluate)
- configs/            # YAML configurations
- requirements.txt
- README.md

## Data
- Expected: CSV/JSON with columns like id, text, product, sub_product, issue, company, state, submitted_at
- Keep PII removal and anonymization as part of preprocessing (scripts/preprocess.py)

## Modeling & Evaluation
- Support for transformers (Hugging Face) and classical models (scikit-learn)
- Use cross-validation, class-weighting or sampling for imbalanced targets
- Save artifacts (tokenizers, encoders, model weights) under artifacts/

## Inference / API
- /predict endpoint accepts JSON { "id": "...", "text": "..." } and returns predictions + confidence + explanations
- Bulk scoring supports CSV upload

## Configuration
- Use YAML files in configs/ for reproducible experiments (hyperparams, data splits, model type)

## Testing
- Minimal unit & integration tests under tests/
- Run:
```bash
pytest -q
```

## Contributing
- Open issues for feature requests/bugs
- Follow branch -> PR workflow; include tests for new logic

## License
MIT License — see LICENSE

---

For usage examples and sample data, check notebooks/ and scripts/ for runnable demos.
