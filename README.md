# End-to-End MLOps Sentiment Analysis Pipeline

An end-to-end **MLOps project for binary sentiment classification** using movie reviews. The project starts with NLP experimentation and model tracking, then evolves into a reproducible ML pipeline with DVC, cloud storage, Docker, CI/CD, AWS EKS deployment, and production monitoring with Prometheus and Grafana.

The repository is organized around the workflow:

**Data → Preprocessing → Feature Engineering → Model Training → Experiment Tracking → DVC Pipeline → Flask API → Docker → AWS ECR → AWS EKS → Monitoring**

---

## 🚀 Project Highlights

- Binary sentiment classification: `positive` vs `negative`
- NLP text preprocessing with:
  - Lowercasing
  - Stop-word removal
  - Number removal
  - Punctuation removal
  - URL removal
  - Lemmatization
- Bag of Words (BoW) feature extraction
- TF-IDF feature extraction
- Baseline Logistic Regression model
- Comparison of multiple ML algorithms
- Logistic Regression hyperparameter tuning with `GridSearchCV`
- Experiment tracking and model logging with **MLflow**
- Remote experiment tracking through **DagsHub**
- Dataset and pipeline versioning with **DVC**
- Local and AWS S3 DVC remotes
- Flask-based model serving
- Docker containerization
- GitHub Actions CI/CD
- AWS ECR container registry
- AWS EKS Kubernetes deployment
- AWS LoadBalancer exposure
- Prometheus metrics collection
- Grafana dashboards
- AWS resource cleanup workflow

---

## 🏗️ High-Level Architecture

```text
                         ┌──────────────────────┐
                         │      IMDB Reviews     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Data Ingestion / DVC │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Text Preprocessing   │
                         │ • lowercase          │
                         │ • stopwords          │
                         │ • numbers            │
                         │ • punctuation        │
                         │ • URLs               │
                         │ • lemmatization      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │      Feature Engineering     │
                    │      • BoW / CountVectorizer │
                    │      • TF-IDF                 │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │       Model Experiments       │
                    │ • Logistic Regression         │
                    │ • Multinomial Naive Bayes     │
                    │ • XGBoost                     │
                    │ • Random Forest               │
                    │ • Gradient Boosting            │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │ MLflow + DagsHub               │
                    │ Parameters / Metrics / Models  │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                    ┌───────────────────────────────┐
                    │        Flask Application       │
                    └──────────────┬────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │      Docker Image    │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       AWS ECR        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       AWS EKS        │
                         │  Kubernetes Cluster  │
                         └──────────┬───────────┘
                                    │
                              LoadBalancer
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │   Production Flask   │
                         │        API           │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                  ┌──────────────┐      ┌──────────────┐
                  │  Prometheus  │ ───► │   Grafana    │
                  │   Metrics    │      │  Dashboard   │
                  └──────────────┘      └──────────────┘
```

---

# 📌 1. Project Setup

The project follows the Cookiecutter Data Science structure and uses a dedicated Python environment.

### Create the environment

```bash
conda create -n atlas python=3.10
conda activate atlas
```

### Install Cookiecutter

```bash
pip install cookiecutter
```

### Create the project structure

```bash
cookiecutter -c v1 https://github.com/drivendata/cookiecutter-data-science
```

The original setup notes also rename:

```text
src.models → src.model
```

The initial repository is then committed and pushed to Git.

---

# 🧪 2. NLP Experiments

The project contains three experimental stages.

## Experiment 1 — Baseline Model

The first experiment creates a small working dataset from `IMDB.csv`, performs text preprocessing, applies a Bag of Words representation, and trains a Logistic Regression classifier.

The experiment samples **500 reviews** and uses:

```python
CountVectorizer(max_features=100)
```

The train/test split uses:

```python
test_size=0.25
random_state=42
```

The experiment tracks:

- Accuracy
- Precision
- Recall
- F1 Score

The trained model is also logged to MLflow.

---

## Experiment 2 — BoW vs TF-IDF

The second experiment expands the comparison by evaluating:

### Vectorizers

```text
BoW
TF-IDF
```

### Algorithms

```text
Logistic Regression
Multinomial Naive Bayes
XGBoost
Random Forest
Gradient Boosting
```

This produces a matrix of model/vectorizer combinations.

For example:

```text
Logistic Regression + BoW
Logistic Regression + TF-IDF

Naive Bayes + BoW
Naive Bayes + TF-IDF

XGBoost + BoW
XGBoost + TF-IDF

Random Forest + BoW
Random Forest + TF-IDF

Gradient Boosting + BoW
Gradient Boosting + TF-IDF
```

Each run records:

- Vectorizer
- Algorithm
- Test size
- Model-specific parameters
- Accuracy
- Precision
- Recall
- F1 Score
- Trained model artifact

MLflow nested runs are used to organize the experiments.

---

## Experiment 3 — Logistic Regression Hyperparameter Tuning

The third experiment focuses on Logistic Regression using TF-IDF.

The hyperparameter search space is:

```python
param_grid = {
    "C": [0.1, 1, 10],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"]
}
```

Five-fold cross-validation is performed using:

```python
GridSearchCV(
    LogisticRegression(),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
```

The experiment logs:

- Hyperparameters
- Accuracy
- Precision
- Recall
- F1 Score
- Mean cross-validation score
- Standard deviation of cross-validation score
- Best F1 score
- Best trained model

---

# 🧹 3. Text Preprocessing

The NLP pipeline applies the following transformations to review text:

```text
Raw Review
    │
    ▼
Lowercase
    │
    ▼
Remove Stop Words
    │
    ▼
Remove Numbers
    │
    ▼
Remove Punctuation
    │
    ▼
Remove URLs
    │
    ▼
Lemmatization
    │
    ▼
Clean Review
```

The dataset is restricted to the two supported sentiment labels:

```text
positive → 1
negative → 0
```

The preprocessing implementation uses NLTK's:

- `stopwords`
- `WordNetLemmatizer`

---

# 📊 4. Model Evaluation

The project evaluates models using four classification metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | Correct positive predictions among predicted positives |
| Recall | Correct positive predictions among actual positives |
| F1 Score | Harmonic mean of precision and recall |

The experiments intentionally log these metrics to MLflow so different model configurations can be compared.

> The source experiments do not contain a fixed final benchmark table, so this README does not invent performance values. The actual results should be read from the MLflow/DagsHub experiment runs.

---

# 🧪 5. MLflow + DagsHub

MLflow is used for experiment tracking and model logging.

DagsHub is configured as the remote MLflow tracking location.

The experiment setup follows this pattern:

```python
mlflow.set_tracking_uri("<DAGSHUB_MLFLOW_TRACKING_URI>")

dagshub.init(
    repo_owner="<DAGSHUB_REPO_OWNER>",
    repo_name="<DAGSHUB_REPO_NAME>",
    mlflow=True
)

mlflow.set_experiment("<EXPERIMENT_NAME>")
```

### Experiments

The project contains experiment tracking for:

```text
Logistic Regression Baseline
BoW vs TfIdf
LoR Hyperparameter Tuning
```

### What MLflow stores

```text
Experiment
 ├── Parameters
 ├── Metrics
 ├── Nested Runs
 └── Model Artifacts
```

This makes it possible to compare model configurations and identify the best-performing experiment.

---

# 📦 6. DVC Pipeline

After experimentation, the project moves toward a reproducible ML pipeline using DVC.

The planned source modules include:

```text
src/
├── logger
├── data_ingestion.py
├── data_preprocessing.py
├── feature_engineering.py
├── model_building.py
├── model_evaluation.py
└── register_model.py
```

The project also introduces:

```text
dvc.yaml
params.yaml
```

### Initialize DVC

```bash
dvc init
```

### Local DVC remote

A temporary local storage directory can be created:

```bash
mkdir local_s3
```

Then configure:

```bash
dvc remote add -d mylocal local_s3
```

### Reproduce the pipeline

```bash
dvc repro
```

### Check pipeline status

```bash
dvc status
```

---

# ☁️ 7. AWS S3 as DVC Remote Storage

For cloud-based dataset/artifact storage, an AWS S3 bucket is configured as a DVC remote.

Install the required packages:

```bash
pip install "dvc[s3]" awscli
```

Configure AWS credentials:

```bash
aws configure
```

Add the S3 remote:

```bash
dvc remote add -d myremote s3://<bucket-name>
```

Push DVC-tracked data:

```bash
dvc push
```

> Never commit AWS credentials, access keys, secret keys, or authentication tokens to Git.

---

# 🌐 8. Flask Application

A `flask_app` directory is introduced for serving the trained model.

The application is intended to expose the machine learning model through a web API.

Install Flask:

```bash
pip install flask
```

The Flask application runs on port `5000` in the deployment configuration.

---

# 🐳 9. Dockerization

The Flask application is containerized with Docker.

Generate application requirements:

```bash
pip install pipreqs
cd flask_app
pipreqs . --force
```

Build the Docker image from the project root:

```bash
docker build -t capstone-app:latest .
```

Run locally:

```bash
docker run -p 8888:5000 capstone-app:latest
```

The application can then be accessed through the mapped port.

If the application requires the project authentication environment variable, provide it through the environment rather than hard-coding it:

```bash
docker run -p 8888:5000 \
  -e CAPSTONE_TEST=<YOUR_SECRET> \
  capstone-app:latest
```

---

# 🔄 10. CI/CD with GitHub Actions

The project uses GitHub Actions for continuous integration and deployment.

Workflow location:

```text
.github/
└── workflows/
    └── ci.yaml
```

The CI/CD workflow is designed to progress through:

```text
Git Push
   │
   ▼
Run Tests
   │
   ▼
Build Application
   │
   ▼
Build Docker Image
   │
   ▼
Push Image to AWS ECR
   │
   ▼
Deploy to AWS EKS
```

The project also includes:

```text
tests/
scripts/
```

for CI-related testing and automation scripts.

---

# 🏗️ 11. AWS ECR

The Docker image is pushed to Amazon Elastic Container Registry.

The CI/CD environment requires values such as:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_REGION
ECR_REPOSITORY
AWS_ACCOUNT_ID
```

The IAM identity used for ECR operations needs appropriate ECR permissions.

---

# ☸️ 12. Kubernetes / AWS EKS Deployment

Before deploying to EKS, the following tools are required:

```text
AWS CLI
kubectl
eksctl
```

Verify installations:

```bash
aws --version
kubectl version --client
eksctl version
```

### Create the EKS cluster

The project setup uses:

```bash
eksctl create cluster \
  --name flask-app-cluster \
  --region us-east-1 \
  --nodegroup-name flask-app-nodes \
  --node-type t3.small \
  --nodes 1 \
  --nodes-min 1 \
  --nodes-max 1 \
  --managed
```

### Configure kubectl

```bash
aws eks --region us-east-1 update-kubeconfig \
  --name flask-app-cluster
```

### Verify the cluster

```bash
aws eks list-clusters
aws eks --region us-east-1 describe-cluster \
  --name flask-app-cluster \
  --query "cluster.status"
```

Check Kubernetes connectivity:

```bash
kubectl get nodes
kubectl get namespaces
kubectl get pods
kubectl get svc
```

---

# 🚢 13. Deploy the Flask Application to EKS

The deployment uses Kubernetes configuration files such as:

```text
deployment.yaml
```

and a Kubernetes service:

```text
flask-app-service
```

The CI/CD configuration, Dockerfile, and Kubernetes deployment configuration need to be kept consistent.

The project configuration also requires the relevant node security group rules to allow traffic to the Flask application port.

### Check the LoadBalancer

```bash
kubectl get svc flask-app-service
```

Once the LoadBalancer is provisioned, use its external endpoint to test the application.

Example:

```bash
curl http://<external-ip>:5000
```

---

# 📈 14. Prometheus Monitoring

Prometheus is deployed on an Ubuntu EC2 instance.

The project notes use:

```text
Instance type: t3.medium
Disk: 20 GB
Prometheus port: 9090
SSH port: 22
```

### Download Prometheus

```bash
wget https://github.com/prometheus/prometheus/releases/download/v2.46.0/prometheus-2.46.0.linux-amd64.tar.gz

tar -xvzf prometheus-2.46.0.linux-amd64.tar.gz

mv prometheus-2.46.0.linux-amd64 prometheus
```

Move the installation:

```bash
sudo mv prometheus /etc/prometheus
sudo mv /etc/prometheus/prometheus /usr/local/bin/
```

### Configure Prometheus

Edit:

```bash
sudo nano /etc/prometheus/prometheus.yml
```

Example configuration:

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "flask-app"
    static_configs:
      - targets: ["<FLASK_LOAD_BALANCER_HOST>:5000"]
```

Run Prometheus:

```bash
/usr/local/bin/prometheus \
  --config.file=/etc/prometheus/prometheus.yml
```

---

# 📊 15. Grafana Monitoring

Grafana is deployed on a separate Ubuntu EC2 instance.

The project notes use:

```text
Instance type: t3.medium
Disk: 20 GB
Grafana port: 3000
SSH port: 22
```

Download the specified Grafana package:

```bash
wget https://dl.grafana.com/oss/release/grafana_10.1.5_amd64.deb
```

Install:

```bash
sudo apt install ./grafana_10.1.5_amd64.deb -y
```

Start Grafana:

```bash
sudo systemctl start grafana-server
```

Enable startup on boot:

```bash
sudo systemctl enable grafana-server
```

Check the service:

```bash
sudo systemctl status grafana-server
```

Open:

```text
http://<EC2-PUBLIC-IP>:3000
```

Then configure Prometheus as a Grafana data source using the Prometheus server endpoint.

---

# 🔐 16. Security and Secrets

Secrets must be supplied through environment variables or CI/CD secret stores.

Examples include:

```text
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AWS_ACCOUNT_ID
CAPSTONE_TEST
```

### Important

Do **not** commit:

```text
.env
AWS credentials
DagsHub tokens
API keys
private SSH keys
Kubernetes secrets containing credentials
```

Use GitHub repository secrets/variables and AWS IAM instead.

If a credential has ever been committed to a repository, treat it as compromised and rotate/revoke it.

---

# 📁 17. Suggested Project Structure

The complete project can follow a structure similar to:

```text
project-root/
│
├── .github/
│   └── workflows/
│       └── ci.yaml
│
├── data/
│
├── notebooks/
│   ├── data.csv
│   └── experiments/
│       ├── exp1_baseline_model.ipynb
│       ├── exp2_bow_vs_tfidf.py
│       └── exp3_lor_bow_hp.py
│
├── src/
│   ├── logger/
│   ├── data_ingestion.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_building.py
│   ├── model_evaluation.py
│   └── register_model.py
│
├── flask_app/
│   ├── ...
│   └── requirements.txt
│
├── tests/
│   └── ...
│
├── scripts/
│   └── ...
│
├── dvc.yaml
├── params.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

The exact filenames may differ depending on the final repository implementation.

---

# 🧰 18. Technology Stack

| Category | Technology |
|---|---|
| Language | Python |
| Environment | Conda |
| Project Structure | Cookiecutter Data Science |
| Data Processing | Pandas, NumPy |
| NLP | NLTK |
| Feature Engineering | CountVectorizer, TF-IDF |
| ML | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow |
| Remote Experiment Tracking | DagsHub |
| Data Versioning | DVC |
| Cloud Storage | AWS S3 |
| API | Flask |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Container Registry | AWS ECR |
| Orchestration | Kubernetes / AWS EKS |
| Infrastructure Management | eksctl / CloudFormation |
| Monitoring | Prometheus |
| Visualization | Grafana |
| Cloud | AWS |

---

# 🧠 19. Important AWS / Kubernetes Concepts

## CloudFormation and EKS

`eksctl` uses AWS CloudFormation behind the scenes to provision AWS infrastructure for an EKS cluster.

The project notes describe CloudFormation stacks for:

```text
EKS control plane
Node groups
```

CloudFormation allows these resources and their dependencies to be managed as a logical stack.

---

## Fleet Requests

An EKS node group uses an Auto Scaling Group to provision EC2 instances.

AWS account-level Fleet Request limits can therefore affect node group creation.

An error such as:

```text
You've reached your quota for maximum Fleet Requests
```

can indicate that the AWS account has reached the relevant quota.

---

## PersistentVolumeClaim (PVC)

A Kubernetes `PersistentVolumeClaim` is a request for storage by an application.

The PVC can be bound to a `PersistentVolume`, while a `StorageClass` defines how the storage is provisioned.

In AWS environments, this can involve storage such as EBS-backed volumes.

---

# 🧹 20. AWS Resource Cleanup

Cloud resources should be removed after testing to avoid unnecessary costs.

### Delete Kubernetes deployment

```bash
kubectl delete deployment flask-app
```

### Delete Kubernetes service

```bash
kubectl delete service flask-app-service
```

### Delete Kubernetes secret

```bash
kubectl delete secret capstone-secret
```

### Delete EKS cluster

```bash
eksctl delete cluster \
  --name flask-app-cluster \
  --region us-east-1
```

### Verify deletion

```bash
eksctl get cluster --region us-east-1
```

Also verify that:

- ECR artifacts are removed if no longer required
- S3 data is removed if no longer required
- CloudFormation stacks are terminated
- EC2 monitoring instances are terminated
- Associated AWS resources are no longer generating charges

---

# 🔬 21. End-to-End Development Lifecycle

The project can be understood as the following MLOps lifecycle:

```text
1. Create Repository
        ↓
2. Create Development Environment
        ↓
3. Generate Project Structure
        ↓
4. Perform NLP Experiments
        ↓
5. Track Experiments with MLflow
        ↓
6. Connect MLflow to DagsHub
        ↓
7. Build Reproducible DVC Pipeline
        ↓
8. Configure S3 Data Storage
        ↓
9. Build Flask Model API
        ↓
10. Add Tests and CI
        ↓
11. Containerize with Docker
        ↓
12. Push Image to AWS ECR
        ↓
13. Deploy to AWS EKS
        ↓
14. Expose Application through LoadBalancer
        ↓
15. Monitor with Prometheus
        ↓
16. Visualize with Grafana
        ↓
17. Clean Up Cloud Resources
```

---

# 🎯 22. Learning Outcomes

This project demonstrates practical experience across the complete machine learning lifecycle:

### Machine Learning

- Text classification
- Feature engineering
- Model comparison
- Hyperparameter optimization
- Classification evaluation

### MLOps

- Experiment tracking
- Model artifact management
- Data versioning
- Reproducible pipelines
- Model serving
- CI/CD

### DevOps

- Docker
- GitHub Actions
- Container registries
- Kubernetes
- Infrastructure provisioning

### Cloud

- AWS S3
- AWS ECR
- AWS EKS
- EC2
- IAM
- CloudFormation

### Observability

- Prometheus
- Grafana
- Application metrics

---

# ▶️ 23. Quick Start

For the experimentation stage:

```bash
conda create -n atlas python=3.10
conda activate atlas

pip install mlflow dagshub
pip install pandas numpy scikit-learn nltk xgboost scipy
```

Prepare the dataset under:

```text
notebooks/data.csv
```

Then run the experiments:

```bash
python exp2_bow_vs_tfidf.py
python exp3_lor_bow_hp.py
```

The baseline experiment can be executed from its Jupyter notebook.

For the full MLOps workflow, continue with:

```text
DVC
 ↓
Flask
 ↓
Docker
 ↓
GitHub Actions
 ↓
AWS ECR
 ↓
AWS EKS
 ↓
Prometheus
 ↓
Grafana
```

---

# ⚠️ Notes

- The experiment scripts use the DagsHub/MLflow repository configuration from the original project notes.
- Replace repository-specific URLs, usernames, endpoints, bucket names, account IDs, and secrets with your own values before publishing the project.
- Do not expose authentication tokens or cloud credentials in source code.
- The ML experiments contain no fixed final accuracy/F1 results in the supplied source material; use MLflow/DagsHub to inspect the actual recorded runs.
- AWS resource names and versions shown in the setup are based on the original project notes and may need adjustment for your environment.

---

# 📚 Project References

The implementation and workflow are based on the project's experiment scripts and deployment notes, including the baseline experiment, BoW vs TF-IDF comparison, Logistic Regression hyperparameter tuning, DVC workflow, AWS deployment, and monitoring setup.
