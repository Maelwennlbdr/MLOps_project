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

### **Pipeline 1: PR → dev** (Test & Build)

**Trigger**: Pull Request to `dev` branch  
**File**: `.github/workflows/ci.yml`

**Steps**:
1. ✅ Checkout code
2. ✅ Setup Python 3.10
3. ✅ Install dependencies (`requirements.txt`)
4. ✅ Run **all test suite**:
   - Unit tests (`test_unit_schema.py`)
   - Integration tests (`test_integration_*.py`)
   - End-to-end tests (`test_e2e.py`)
5. ✅ Build Docker image (no push)

**Success Criteria**:
- All tests pass ✓
- No lint/type errors ✓

**Failure**: PR blocked, cannot merge to `dev`

---

### **Pipeline 2: dev → staging** (Deploy & Validate)

**Trigger**: Push to `staging` branch  
**File**: `.github/workflows/deploy-staging.yml`

**Steps**:
1. ✅ Checkout code
2. ✅ Build Docker image (`docker build -t mlops-staging .`)
3. ✅ Start container with staging environment variables
4. ✅ Wait for API readiness (health check endpoint)
5. ✅ Run smoke tests:
   - `/health` endpoint
   - `/predict` with sample data
6. ✅ Stop container

**Environment Variables**:
- `MLFLOW_TRACKING_URI`: DagsHub instance
- `MLFLOW_MODEL_URI`: Points to `@Staging` version

**Purpose**: Validates that the containerized app works with the latest staging model

---

### **Pipeline 3: staging → main** (Quality Gates + Production Deploy)

**Trigger**: Push to `main` branch  
**File**: `.github/workflows/deploy-prod.yml`

**Critical Steps**:

1. **Quality Gates Validation** (BLOCKING STEP)
   ```bash
   python ml/quality_gates.py
   ```
   - Fetches latest model version in `@Staging`
   - Checks metrics against thresholds:
     - `accuracy >= 0.74` ✓
   - **If FAIL**: Pipeline stops, model stays in Staging, production not updated
   - **If PASS**: Model automatically promoted to `@Production` stage

2. ✅ Build production Docker image
3. ✅ Start container with production credentials
4. ✅ Run final smoke tests
5. ✅ Validate API endpoints

**Environment Variables**:
- `MLFLOW_MODEL_URI`: Points to `@Production` version (highest priority)

**Result**: Only models that pass quality gates reach production

---

## 🚀 Model Promotion Pipeline

The model follows a **3-stage lifecycle** in MLflow Registry:

```
Training → Staging → Production
   ↓          ↓           ↓
train.py  [Quality   Served in
registers  Gates]      Prod API
```

### **Stage 1: Training → Staging** (Automatic)

**When**: `ml/train.py` runs (manual trigger or scheduled)

```python
# train.py automatically:
1. Train LogisticRegression on diabetes dataset
2. Log metrics (accuracy, params) to MLflow
3. Register model as "mlops-model"
4. Assign to @Staging alias (always gets latest trained model)
5. Log: git commit + DVC version for reproducibility
```

**Result**: New model version in `@Staging` (not served anywhere yet)

---

### **Stage 2: Staging → Production** (Quality Gates)

**When**: Code is merged to `main` branch (triggers `deploy-prod.yml`)

**Quality Gates Check** (`ml/quality_gates.py`):
```python
# Validates that @Staging model meets criteria:
- accuracy >= 0.74  ← Current model: 0.7468 ✓ PASSES

# If ALL checks pass:
  → Promote to @Production
  → Log: "Model vX promoted to Production"
  
# If ANY check fails:
  → Model stays in @Staging
  → Pipeline fails
  → Production keeps serving previous @Production version
```

**Security**: Production only ever serves models from `@Production` stage

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
- If no `@Production` version exists, API fails (safe-fail)
- No automatic rollback (manual if needed)

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

### **Architecture**

```
GitHub main branch → GitHub Actions
                     ↓
                 Quality Gates Check
                     ↓
                 Docker Build & Push
                     ↓
                 Cloud Platform (Railway/Render)
                     ↓
                 Public HTTPS URL
```

### **Deployment Steps**

1. **Environment Setup**
   - Add GitHub Secrets (MLFLOW_*, AWS_*, etc.)
   - Configure cloud platform credentials

2. **Push to main**
   ```bash
   git checkout staging
   git pull
   # Merge PR from dev
   
   git checkout main
   git merge staging
   git push origin main
   ```

3. **GitHub Actions Runs**
   - Quality gates validation
   - Docker image build
   - Cloud deployment
   - Health checks

4. **Verify Production**
   ```bash
   curl https://your-app.railway.app/health
   curl -X POST https://your-app.railway.app/predict ...
   ```

### **Monitoring**

- **MLflow Dashboard**: [https://dagshub.com/louiseLV/MLOps_project](https://dagshub.com/louiseLV/MLOps_project)
  - View all model versions
  - Model stages (@Staging, @Production)
  - Metrics & parameters

- **Production Logs**: Cloud platform (Railway/Render) dashboard

---

## 🔍 Troubleshooting

### **Quality Gate Fails**

```
❌ accuracy: 0.73 < 0.74 ✗
   Model stays in Staging, production not updated
```

**Solution**: Improve model, retrain, and rerun workflow

### **API Cannot Load Model**

```
ERROR: Could not find model 'mlops-model@Production'
```

**Solution**: 
1. Check MLflow credentials
2. Run `ml/quality_gates.py` to promote a Staging model
3. Ensure model version exists in registry

### **Tests Fail in CI**

Check logs in GitHub Actions:
- Unit test: schema validation
- Integration: MLflow connectivity
- E2E: API endpoint availability

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

## 📅 Timeline

| Date | Milestone |
|------|-----------|
| 28/02 | Quality gates implementation |
| 01/03 | Final testing & validation |
| 03/03 | Production deployment |
| 04/03 | **Project Submission** |

---

## 👥 Team

- **Albane Coiffe** - Backend & API
- **Louise Lavergne** - ML & MLflow
- **Maelwenn Labidurie** - DevOps & CI/CD

**Repository**: [https://github.com/Maelwennlbdr/MLOps_project](https://github.com/Maelwennlbdr/MLOps_project)