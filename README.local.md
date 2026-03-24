# DevloCare ML Pipeline

This project implements an end-to-end ML pipeline for DevloCare:

- **Load dataset**
- **Preprocess features**
- **Train model**
- **Evaluate performance**
- **Log experiments to MLflow / DagsHub**
- **Version data and models with DVC**
- **Output a trained model** (`.pkl`)

## Project structure

```text
DevloCare_ML_Pipeline/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   ├── pipelines/
│   │   └── training_pipeline.py
│   │
│   └── utils/
│
├── data/                # DVC tracked
├── models/              # DVC tracked
│
├── dvc.yaml
├── params.yaml
├── mlruns/              # MLflow
│
├── requirements.txt
└── README.md
```

## Quickstart

1. **Install dependencies**

```bash
pip install -r requirements.txt
```

2. **Initialize DVC (if not already)**

```bash
dvc init
git add .dvc .gitignore
git commit -m "Initialize DVC"
```

3. **Run the training pipeline**

```bash
python -m src.pipelines.training_pipeline
```

This will:

- Load and preprocess the dataset (path configured in `params.yaml`)
- Train the model and evaluate metrics
- Log metrics and artifacts to MLflow / DagsHub
- Save the trained model to the `models/` directory as a `.pkl` file

