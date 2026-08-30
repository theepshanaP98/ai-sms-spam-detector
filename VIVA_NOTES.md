# Viva / Demonstration Notes

## One-sentence project explanation

This project is an NLP-based web application that converts SMS text into TF-IDF features and uses Logistic Regression to predict whether the message is spam or legitimate.

## Why is this AI?

The application does not rely on fixed hand-written spam rules. A supervised machine-learning model learns statistical relationships between labelled training messages and their classes, then generalizes those learned patterns to unseen messages.

## Why TF-IDF?

Machine-learning algorithms require numerical features. TF-IDF converts text into sparse numerical vectors while emphasizing informative terms and reducing the influence of extremely common terms.

## Why Logistic Regression?

It is fast, interpretable, produces class probabilities, and is effective for binary classification with sparse high-dimensional text features.

## What is the input and output?

Input: raw SMS text.  
Output: spam/ham label plus estimated class probabilities and confidence.

## How was the model evaluated?

The dataset is split into 80% training and 20% testing using stratification. Performance is reported using accuracy, spam precision, spam recall, spam F1 score, a confusion matrix, and a classification report.

## Why is recall important for spam?

Spam recall measures how many actual spam messages the system successfully detects. Low spam recall means many harmful or unwanted messages are missed.

## Why is precision important?

Spam precision measures how often messages labelled spam are actually spam. Low precision means legitimate messages are incorrectly blocked, which is undesirable.

## Why FastAPI?

FastAPI makes the trained model accessible as a REST API and also supports automatic interactive Swagger documentation. The same backend serves the browser interface.

## What happens in the cloud?

Render installs dependencies, runs the training script during the build, and starts the FastAPI application with Uvicorn. The service then receives HTTP requests through a public URL.

## Why Docker?

Docker packages the application, dependencies, model-training step, and server command into a reproducible container. It is also a deployment fallback required by the assignment if direct cloud deployment cannot be completed.

## Main limitation

The model is trained on an older English SMS dataset, so it may not generalize to modern or multilingual spam without additional representative training data.
