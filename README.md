# Rice Grain Classifier

Final project for the DTU course 02476 Machine Learning Operations

## Project Description

This project implements a complete MLOps pipeline for classifying five varieties of rice grains: **Arborio**, **Basmati**, **Ipsala**, **Jasmine**, and **Karacadag**. While the core task is a classification problem, the focus is on applying the full Machine Learning Operations lifecycle—from data versioning and experiment tracking to containerized cloud training and user-facing deployment.

## Objective

The objective is to build a production-ready ML system that is:

- **Reproducible:** Environment isolation via Docker and dependency locking with `uv.lock`, plus data versioning with DVC
- **Scalable:** Cloud training on Google Vertex AI with data stored in GCP Cloud Storage
- **Observable:** Full experiment tracking with Weights & Biases, logging metrics, artifacts, and visualizations
- **Deployable:** Containerized API and Gradio frontend deployed on Google Cloud Run

## Data

We use the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) from Kaggle.

- **Size:** 75,000 images (15,000 per class)
- **Classes:** Arborio, Basmati, Ipsala, Jasmine, Karacadag
- **Preprocessing:** Images resized to 224×224, normalized with ImageNet statistics, with data augmentation (random horizontal flip, rotation, color jitter) applied to the training set
- **Storage:** Data is versioned with DVC and stored in a GCP Cloud Storage bucket to keep the repository lightweight

## Model

We implemented a custom CNN architecture with:

- Two convolutional layers (3→16→32 channels) with ReLU activation and max pooling
- Fully connected layers (32×56×56 → 128 → 5 classes)
- Trained with Adam optimizer and CrossEntropyLoss
- Achieves **99%+ accuracy** on the test set across all five rice varieties

## Frameworks & Tools

| Tool                      | Purpose                                                      |
| :------------------------ | :----------------------------------------------------------- |
| **PyTorch**               | Deep learning framework for model architecture and training  |
| **DVC**                   | Data versioning linked to GCP Cloud Storage                  |
| **Docker**                | Containerization for reproducible environments               |
| **GitHub Actions**        | CI/CD for testing, linting, and automated cloud deployments  |
| **GCP Artifact Registry** | Docker image hosting                                         |
| **GCP Vertex AI**         | Managed cloud training infrastructure                        |
| **GCP Cloud Run**         | Serverless deployment for API and frontend                   |
| **FastAPI**               | Backend API for triggering cloud training jobs               |
| **Gradio**                | Frontend UI for image upload and inference                   |
| **Weights & Biases**      | Experiment tracking, metrics logging, and artifact storage   |
| **scikit-learn**          | Evaluation metrics (precision, recall, F1, confusion matrix) |

## Results

What we built:

- A CNN classifier achieving 99%+ test accuracy on rice grain classification
- Automated CI/CD pipeline with GitHub Actions for testing, linting, and deployment
- Containerized training that runs on GCP Vertex AI
- Data versioning with DVC backed by GCP Cloud Storage
- Experiment tracking with Weights & Biases (loss curves, accuracy, confusion matrices, sample predictions)
- FastAPI backend deployed on Cloud Run for triggering training jobs
- Gradio frontend for user-friendly image classification
- 71% code coverage across 19 tests

## How to Run

### Option 1: Use the Hosted Service

The easiest way to try the classifier is via our deployed Gradio frontend:

**[https://rice-frontend-681024937248.europe-west1.run.app](https://rice-frontend-681024937248.europe-west1.run.app)**

> **Note:** The service runs on Cloud Run and may take 30-60 seconds to cold start on first load. Be patient.

Upload a rice grain image and get classification predictions with confidence scores for all five varieties.

### Option 2: Run Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/adamajane/machine-learning-operations-02476-project.git
   cd machine-learning-operations-02476-project
   ```

2. Install dependencies using [uv](https://github.com/astral-sh/uv):

   ```bash
   uv sync
   ```

3. Run the Gradio frontend:
   ```bash
   uv run python -m rice_cnn_classifier.frontend
   ```

> **Note:** On first run, the frontend will download the trained model from GCP Cloud Storage, which may take a moment.

The frontend will be available at `http://0.0.0.0:7860`.
