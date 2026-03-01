# MLOps Final Project: Diabetes Prediction API

**Team**: Albane Coiffe, Louise Lavergne, Maelwenn Labidurie  
**Deadline**: 04/03/2026

Production-grade ML application with full CI/CD pipeline, model versioning, and automated quality gates.

---

## 📋 Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Git Workflow](#git-workflow)
3. [CI/CD Pipelines](#cicd-pipelines)
4. [Model Promotion Pipeline](#model-promotion-pipeline)
5. [Setup & Local Development](#setup--local-development)
6. [Production Deployment](#production-deployment)
7. [Testing](#testing)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub (Version Control)                  │
│  feature/* → dev → staging → main (CI/CD Pipelines)         │
└──────────┬────────────────────────────────────────────────┬─┘
           │                                                │
           ▼                                                ▼
    ┌─────────────┐                            ┌─────────────────┐
    │  Local Dev  │                            │  GitHub Actions │
    │  (DVC Pull) │                            │   (3 Workflows) │
    └─────────────┘                            └────────┬────────┘
                                                        │
           ┌────────────┬────────────┬──────────────────┘
           ▼            ▼            ▼
    ┌─────────────┐ ┌─────────────┐ ┌──────────────────┐
    │   DVC S3    │ │   MLflow    │ │   Docker Image   │
    │ (Data v.)   │ │  DagsHub    │ │    Registry      │
    │             │ │ (Model v.)  │ │                  │
    └─────────────┘ └─────────────┘ └──────────────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         ┌────────┐  ┌──────────┐  ┌────────────┐
         │Staging │  │Production│  │  API Logs  │
         │ (MLflow)  │(MLflow)  │  │            │
         └────────┘  └──────────┘  └────────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Cloud Deploy    │
                  │ (Railway/Render) │
                  └──────────────────┘
```

---

## 📦 Tech Stack

- **Model Training**: Scikit-learn (Logistic Regression)
- **Data Versioning**: DVC (S3 remote)
- **Model Registry & Tracking**: MLflow + DagsHub
- **API**: FastAPI
- **Testing**: Pytest
- **Container**: Docker
- **CI/CD**: GitHub Actions
- **Deployment**: Railway/Render (Cloud)

---

## 🌳 Git Workflow & Branching Strategy

```
feature/xxx → [PR, tests] → dev → [integration] → staging → [validation] → main
      ↑                                                                        ↓
      └────────────────────────── (hotfix branches) ─────────────────────────┘
```

### Branch Purposes

| Branch | Purpose | Protection |
|--------|---------|-----------|
| `main` | Production environment | ✓ Requires PR, passing tests, quality gates |
| `staging` | Pre-production testing | ✓ Requires PR, passing tests |
| `dev` | Integration branch | ✓ Requires PR, passing tests |
| `feature/*` | Feature development | ✗ Open for development |

---

## 🔄 CI/CD Pipelines

The project implements **3 GitHub Actions workflows** for automated testing and deployment:

### **Pipeline 1: PR → dev** (Unit + Integration + Docker Build)

**Trigger**: Pull Request to `dev` branch  
**File**: `.github/workflows/ci.yml`

**Steps**:
1. ✅ Checkout code
2. ✅ Setup Python 3.10
3. ✅ Install dependencies (`requirements.txt`)
4. ✅ Run **unit tests only**:
   - Unit tests (`test_unit_schema.py`)
   - Sanity test (`test_dummy.py`)
5. ✅ Run **integration tests only**:
   - Integration tests (`test_integration_mlflow.py`)
   - Integration tests (`test_integration_api_mlflow.py`)
6. ✅ Build Docker image (no push)

**Success Criteria**:
- All tests pass ✓
- No lint/type errors ✓

**Failure**: PR blocked, cannot merge to `dev`

---

### **Pipeline 2: dev → staging** (Full Test Suite + Deploy Staging)

**Trigger**: Push to `staging` branch  
**File**: `.github/workflows/deploy-staging.yml`

**Steps**:
1. ✅ Checkout code
2. ✅ Setup Python 3.10
3. ✅ Install dependencies (`requirements.txt`)
4. ✅ Run **full test suite** (`pytest -v`)
5. ✅ Build Docker image (`docker build -t mlops-staging .`)
6. ✅ Start container with staging environment variables
7. ✅ Wait for API readiness (health check endpoint)
8. ✅ Run smoke tests:
   - `/health` endpoint
   - `/predict` with sample data
9. ✅ Stop container

**Environment Variables**:
- `MLFLOW_TRACKING_URI`: DagsHub instance
- `MLFLOW_MODEL_URI`: Points to `@Staging` version

**Purpose**: Validates that the containerized app works with the latest staging model

---

### **Pipeline 3: staging → main** (Promotion Gate + Production Deploy)

**Trigger**: Push to `main` branch  
**File**: `.github/workflows/deploy-prod.yml`

**Full Production Pipeline**:

1. **Setup & Dependencies**
   ```bash
   pip install mlflow
   pip install -r requirements.txt
   ```

2. **Quality Gates Validation** (BLOCKING STEP)
   ```bash
   python ml/quality_gates.py
   ```
   - Fetches latest model version with `@Staging` alias
   - Validates metrics against thresholds:
     - ✅ `accuracy >= 0.74`
   - **If FAIL**: Pipeline stops, model stays in Staging, production NOT updated
   - **If PASS**: Automatically promotes model to `@Production` alias

3. ✅ Build production Docker image
4. ✅ Start container with production credentials
5. ✅ Run final smoke tests (health + predict endpoints)

**Key Advantage**: Production deploys the already validated model candidate from registry flow (`@Staging` -> `@Production`), without retraining on `main`.

**Result**: Only models passing quality gates reach production with `@Production` alias

---

## 🚀 Model Promotion Pipeline

The model follows a **3-stage lifecycle** in MLflow Registry:

```
Push to main
   ↓
deploy-prod.yml runs:
   1. Read candidate model from @Staging
   2. Quality Gates check (accuracy >= 0.74)
   3. If ✅ Pass → Promote to @Production
   4. If ❌ Fail → Stay in @Staging (prod unchanged)
   ↓
API serves only @Production models
```

### **Stage 1: Candidate Model in Staging**

**When**: Candidate model has been registered and assigned to `@Staging` upstream (before production promotion).

```python
# Candidate exists in MLflow Registry:
mlops-model@Staging
```

**Result**: Staging version ready for promotion checks.

---

### **Stage 2: Staging → Production** (Quality Gates Check)

**When**: `deploy-prod.yml` runs after merge to `main`

**Quality Gates Validation** (`ml/quality_gates.py`):
```python
# Validates that @Staging model meets criteria:
accuracy >= 0.74  ← Current model: 0.7468 ✓ PASSES

# If ALL checks pass:
  → Promote to @Production alias
  → Only this version served in production
  
# If ANY check fails:
  → Model stays in @Staging
  → Previous @Production version still serving
  → Pipeline fails (preventing bad deployment)
```

**Security**: Production ONLY ever serves models with `@Production` alias

---

### **Stage 3: Production Runtime**

**In API** (`backend/app/predict.py`):
```python
MLFLOW_MODEL_URI = os.getenv(
    "MLFLOW_MODEL_URI", 
    "models:/mlops-model@Production"
)
model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
```

**Behavior**:
- Always loads from `@Production` alias
- If no `@Production` version exists → API fails (safe-fail)
- No automatic rollback (manual intervention needed if reverting)

---

---

## 📊 Testing Strategy

### **Test Types & Coverage**

| Type | File | Count | Purpose |
|------|------|-------|---------|
| **Unit** | `test_unit_schema.py` | 1 | Pydantic validation of input schema |
| **Integration** | `test_integration_mlflow.py` | 1 | MLflow model loading & inference |
| **Integration** | `test_integration_api_mlflow.py` | 1 | API output matches model direct output |
| **E2E** | `test_e2e.py` | 1 | Full prediction flow via HTTP |

### **Example: Quality Gate Threshold**

Given current model accuracy **0.7468**:
- ✅ Passes threshold (0.7468 >= 0.74)
- ✅ Will be promoted to Production
- ✅ Will be served in production API

If accuracy **< 0.74**:
- ❌ Fails quality gate in `deploy-prod.yml`
- ❌ Model stays in Staging
- ❌ Production keeps serving previous model

---

## 🛠️ Setup & Local Development

### **Prerequisites**

```bash
# Install system dependencies
- Python 3.10+
- Docker
- Git
- DVC
```

### **Local Development Setup**

```bash
# 1. Clone repo
git clone https://github.com/Maelwennlbdr/MLOps_project.git
cd MLOps_project

# 2. Create feature branch
git checkout -b feature/your-feature dev

# 3. Install ML dependencies
pip install -r requirements.txt
pip install "dvc[s3]"

# 4. Pull data from DVC remote (S3)
dvc pull -r myremote

# 5. Set MLflow credentials (DagsHub)
export MLFLOW_TRACKING_URI="https://dagshub.com/louiseLV/MLOps_project-dagshub.mlflow"
export MLFLOW_TRACKING_USERNAME="your_dagshub_username"
export MLFLOW_TRACKING_PASSWORD="your_dagshub_token"

# 6. Train model (creates new version in @Staging)
python ml/train.py

# 7. Run tests locally
pytest -v
```

### **Local API Testing**

```bash
# 1. Install backend
cd backend
pip install -r requirements.txt

# 2. Start API
uvicorn app.main:app --reload

# 3. In another terminal, test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 0,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50
  }'
```

---

## 🌐 Production Deployment

### **Automated Production Pipeline Flow**

```
1. Code ready on staging branch
   ↓
2. Create PR: staging → main
   ↓
3. Merge PR to main (GitHub Actions triggers)
   ↓
4. deploy-prod.yml executes:
   a. ✅ Run quality gates on @Staging model
   b. 🚀 If OK: promote to @Production
   c. 🐳 Build & deploy Docker container
   d. 🧪 Run smoke tests on live API
   ↓
5. API serves predictions from @Production model
```

### **Prerequisites for Deployment**

Add these **GitHub Secrets** to your repo:
- `MLFLOW_TRACKING_URI` - DagsHub MLflow instance
- `MLFLOW_TRACKING_USERNAME` - DagsHub username
- `MLFLOW_TRACKING_PASSWORD` - DagsHub token

### **Merging to Production**

```bash
# 1. On staging, all tests passed
git checkout staging
git status

# 2. Create PR or merge
git checkout main
git merge staging
git push origin main

# 3. GitHub Actions automatically:
#    - Validates quality gates
#    - Deploys if OK
```

### **Monitoring Deployment**

- **GitHub Actions Logs**: Check workflow progress
- **MLflow Dashboard**: [https://dagshub.com/louiseLV/MLOps_project](https://dagshub.com/louiseLV/MLOps_project)
  - View all model versions
  - Check @Production vs @Staging aliases
  - Track metrics & parameters

### **If Quality Gates Fail**

```
Error: accuracy: 0.73 < 0.74
→ Model stays in @Staging
→ Production continues serving previous @Production version
→ Fix model/training pipeline upstream, then merge to main again
```

---

## 🔍 Troubleshooting

### **Quality Gate Fails**

```
❌ accuracy: 0.73 < 0.74 ✗
   Model stays in @Staging, production NOT updated
   Previous @Production model continues serving
```

**Solution**: 
1. Improve model training (better hyperparameters)
2. Check data quality → might need more/better features
3. Retrain/register candidate model upstream: `python ml/train.py`
4. Push changes and merge to main again
5. Quality gates will run automatically with new model

---

### **API Cannot Load Model**

```
ERROR: Could not find model 'mlops-model@Production'
```

**Cause**: No model has been promoted to @Production (all training/gates failed)

**Solution**: 
1. Check MLflow credentials are correct
2. View MLflow dashboard to see available versions
3. Ensure at least one model passed quality gates
4. Manually check: `python ml/quality_gates.py`

---

### **"Aucun modèle trouvé avec l'alias 'Staging'"**

```
❌ Aucun modèle trouvé avec l'alias 'Staging'
```

**Cause**: No candidate model was available in registry under `@Staging`

**Solution**:
1. Check training workflow/logs that create/register the model
2. Common issues:
   - Model registration failed
   - Python dependency missing
   - Dataset unavailable/corrupted in training environment
3. Fix the training issue locally first (`python ml/train.py`)
4. Commit fix and retry merge to main

---

### **Tests Fail in CI (PR → dev)**

```
FAILED tests/test_unit_schema.py
FAILED tests/test_integration_api_mlflow.py
```

**Solution**:
1. Run tests locally: `pytest -v`
2. Check error messages in GitHub Actions logs
3. Common issues:
   - Package not installed
   - MLflow credentials missing locally
   - API not responding
4. Fix locally, commit, re-push to feature branch

---

### **Docker Build Fails**

```
ERROR: requirements.txt not found
ERROR: Dockerfile issue
```

**Solution**:
1. Verify Dockerfile exists at project root
2. Check requirements.txt has all dependencies
3. Test locally: `docker build .`

---

## 📁 Project Structure

```
MLOps_project/
├── .github/workflows/          # CI/CD Pipelines
│   ├── ci.yml                  # Test PR → dev
│   ├── deploy-staging.yml      # Deploy to staging
│   └── deploy-prod.yml         # Quality gates + Production
├── backend/                    # FastAPI Application
│   └── app/
│       ├── main.py            # API endpoints
│       ├── predict.py         # Model inference (MLflow)
│       └── schemas.py         # Pydantic models
├── ml/                         # ML Pipeline
│   ├── train.py               # Model training
│   ├── quality_gates.py        # Quality validation
│   ├── evaluate.py            # Model evaluation
│   ├── config.yaml            # ML config
│   └── models/                # Trained models (joblib)
├── data/                       # Data versioning (DVC)
│   └── raw/
│       └── diabetes.csv.dvc   # Data pointer
├── tests/                      # Test suite
│   ├── test_unit_schema.py
│   ├── test_integration_*.py
│   └── test_e2e.py
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Git workflow with protected branches
- ✅ Automated testing in CI/CD
- ✅ Data versioning with DVC
- ✅ Model registry & MLflow
- ✅ Automated quality gates
- ✅ Safe production deployments
- ✅ Containerization & cloud deployment
- ✅ Reproducibility & traceability

---

## 👥 Team

- **Albane Coiffe** 
- **Louise Lavergne** 
- **Maelwenn Labidurie** 

**Repository**: [https://github.com/Maelwennlbdr/MLOps_project](https://github.com/Maelwennlbdr/MLOps_project)
