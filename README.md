# AI SMS Spam Detector

An end-to-end Artificial Intelligence application that classifies SMS text as **spam** or **ham (legitimate)** using Natural Language Processing (NLP), exposes the model through a FastAPI REST API, provides a browser-based interface, and is prepared for cloud deployment on Render and Docker.

## 1. Problem Statement

Unwanted SMS spam is a common communication problem. Spam messages may contain unwanted promotions, misleading prize claims, suspicious links, or other unsolicited content. Manual filtering is inefficient at scale. This project develops an AI-based text classifier that automatically predicts whether an incoming SMS message is spam or legitimate.

## 2. Use Case

The application can be used as a prototype filtering component for:

- SMS or messaging applications
- Customer communication systems
- Mobile security tools
- Telecom message screening systems
- Educational demonstrations of NLP classification

A user enters a message in the web interface or sends it to the API. The system returns the predicted class and estimated probabilities.

## 3. Solution Overview

The solution follows an end-to-end AI application workflow:

1. Obtain the labelled SMS Spam Collection dataset from UCI.
2. Clean duplicate/missing rows.
3. Split the data into stratified training and test sets.
4. Convert text into numerical TF-IDF features using unigrams and bigrams.
5. Train a Logistic Regression binary classifier.
6. Evaluate the model using accuracy, precision, recall, F1 score and a confusion matrix.
7. Save the trained scikit-learn pipeline with Joblib.
8. Load the trained model in a FastAPI application.
9. Expose the model through a REST endpoint and browser user interface.
10. Deploy the application to Render or build/run it with Docker.

## 4. Dataset

**Dataset:** SMS Spam Collection  
**Source:** UCI Machine Learning Repository, dataset ID 228  
**Task:** Text classification  
**Instances reported by UCI:** 5,574  
**Labels:** `ham` and `spam`

The training script downloads and extracts the dataset automatically. After cleaning, the exact row count used by the model is written to `models/metrics.json`.

The UCI dataset page describes the collection as a public set of labelled SMS messages assembled for mobile-phone spam research.

## 5. AI/ML Approach

### 5.1 Text representation — TF-IDF

`TfidfVectorizer` converts SMS text into numerical feature vectors. The model uses:

- lowercasing
- Unicode accent normalization
- unigram and bigram features (`ngram_range=(1,2)`)
- sublinear term frequency

TF-IDF gives more weight to informative words/phrases and less weight to extremely common terms.

### 5.2 Classification — Logistic Regression

Logistic Regression is used because it is efficient and well suited to sparse, high-dimensional text features. `class_weight="balanced"` is enabled because spam and ham classes are not equally represented.

### 5.3 Evaluation

The model uses an 80/20 stratified train-test split with `random_state=42`. Evaluation includes:

- Accuracy
- Spam precision
- Spam recall
- Spam F1 score
- Confusion matrix
- Full classification report

Run the training script and inspect `models/metrics.json` for the measured results produced from the current dataset.

## 6. Application Architecture

```text
                +--------------------------+
                | UCI SMS Spam Collection  |
                +------------+-------------+
                             |
                             v
                +--------------------------+
                | scripts/train_model.py   |
                | Cleaning + Train/Test    |
                +------------+-------------+
                             |
                             v
                +--------------------------+
                | TF-IDF + Logistic        |
                | Regression Pipeline      |
                +------------+-------------+
                             |
                             v
                +--------------------------+
                | spam_classifier.joblib   |
                +------------+-------------+
                             |
                             v
     +------------------- FastAPI -------------------+
     |                                               |
     |  Web UI (/)          REST API (/api/predict)  |
     +----------------------+------------------------+
                            |
                            v
                +--------------------------+
                | Render Cloud / Docker    |
                +--------------------------+
```

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Programming language | Python 3.12 |
| AI / ML | scikit-learn |
| NLP feature extraction | TF-IDF |
| Model | Logistic Regression |
| Data handling | pandas |
| Model persistence | joblib |
| API / backend | FastAPI |
| Web server | Uvicorn |
| Web UI | HTML, CSS, JavaScript, Jinja2 |
| Testing | pytest + FastAPI TestClient |
| Cloud deployment | Render |
| Containerization | Docker |
| Source control | Git + public GitHub repository |

## 8. Project Structure

```text
ai_sms_spam_detector/
├── app/
│   ├── main.py
│   ├── model.py
│   ├── static/style.css
│   └── templates/index.html
├── data/
│   └── README.md
├── models/
│   └── .gitkeep
├── scripts/
│   └── train_model.py
├── tests/
│   └── test_api.py
├── .gitignore
├── Dockerfile
├── render.yaml
├── requirements.txt
└── README.md
```

## 9. Local Setup Instructions

### 9.1 Clone the repository

```bash
git clone <YOUR_PUBLIC_GITHUB_REPOSITORY_URL>
cd ai_sms_spam_detector
```

### 9.2 Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 9.3 Install dependencies

```bash
pip install -r requirements.txt
```

### 9.4 Train the model

```bash
python scripts/train_model.py
```

This downloads the UCI data, trains the classifier, and creates:

- `models/spam_classifier.joblib`
- `models/metrics.json`

### 9.5 Run tests

```bash
pytest -q
```

### 9.6 Start the application

```bash
uvicorn app.main:app --reload
```

Then open:

- Web UI: `http://127.0.0.1:8000/`
- Swagger API documentation: `http://127.0.0.1:8000/docs`
- Health endpoint: `http://127.0.0.1:8000/api/health`

## 10. API / Web Application Usage

### Browser UI

1. Open the root URL.
2. Enter an SMS message.
3. Click **Classify Message**.
4. Review the predicted label, spam probability, ham probability, and confidence.

### API endpoint

**POST** `/api/predict`

Request:

```json
{
  "message": "Congratulations! You have won a free prize. Call now."
}
```

Example response format:

```json
{
  "label": "spam",
  "spam_probability": 0.9234,
  "ham_probability": 0.0766,
  "confidence": 0.9234
}
```

The exact values depend on the trained model.

## 11. Cloud Deployment Details — Render

This repository contains `render.yaml` and can be deployed as a Render Python Web Service.

### Option A — Render Dashboard

1. Push this project to a **public GitHub repository**.
2. Sign in to Render.
3. Choose **New > Web Service**.
4. Connect the GitHub repository.
5. Use the following settings if Render does not detect them automatically:

**Build command**

```bash
pip install -r requirements.txt && python scripts/train_model.py
```

**Start command**

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

6. Deploy the service.
7. Open the assigned `onrender.com` URL and verify `/`, `/api/health`, and `/docs`.

### Option B — render.yaml

If Blueprint deployment is available in your Render account, use the included `render.yaml` to create the service from the repository configuration.

## 12. Docker Instructions

Docker is included as a deployment fallback even when direct cloud deployment is successful.

### Build

```bash
docker build -t ai-sms-spam-detector .
```

### Run

```bash
docker run --rm -p 8000:8000 ai-sms-spam-detector
```

Open `http://localhost:8000`.

### Push to Docker Hub (optional)

```bash
docker tag ai-sms-spam-detector <DOCKERHUB_USERNAME>/ai-sms-spam-detector:latest
docker login
docker push <DOCKERHUB_USERNAME>/ai-sms-spam-detector:latest
```

## 13. Model Limitations

- The training data is English-language SMS and may not generalize well to other languages.
- Spam patterns change over time, so model performance can degrade without retraining.
- An attacker may deliberately change spelling or wording to evade a classifier.
- The probability is a model estimate, not a security guarantee.
- The dataset may not represent current regional SMS patterns.

## 14. Ethical and Responsible AI Considerations

- Do not automatically delete messages solely because the model predicts spam.
- False positives can hide legitimate communication, so production systems should provide user review/recovery options.
- Avoid collecting or storing private message content unless necessary and authorized.
- Monitor model performance and class-specific errors after deployment.
- Retrain on appropriate, consented, and representative data when adapting the system to a new region or language.

## 15. Future Improvements

- Add multilingual support.
- Compare Logistic Regression with Naive Bayes, SVM, or transformer models.
- Add model monitoring and drift detection.
- Add user feedback for incorrect predictions.
- Add authentication/rate limiting for production API use.
- Add CI/CD tests with GitHub Actions.

## 16. References

1. UCI Machine Learning Repository — SMS Spam Collection, dataset ID 228.
2. Almeida, T. A., Hidalgo, J. M. G., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results.
3. scikit-learn documentation — `TfidfVectorizer`, `LogisticRegression`, `Pipeline`.
4. FastAPI documentation.
5. Render documentation — Deploy a FastAPI application.

## 17. Author

Replace this section before submission:

**Student Name:** `<YOUR NAME>`  
**Student ID:** `<YOUR STUDENT ID>`  
**Module:** AI Application Development and Cloud Deployment
