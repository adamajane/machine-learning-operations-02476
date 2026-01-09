# machine-learning-operations-02476
Final project for the DTU course 02476 Machine Learning Operations

The primary objective of this project is to build a robust, production-ready MLOps pipeline for classifying five different varieties of rice grains: **Arborio, Basmati, Ipsala, Jasmine, and Karacadag**. 

While the core technical task is a computer vision classification problem, the overarching goal is to apply the full "Machine Learning Operations" lifecycle to a large-scale dataset. We aim to move beyond a simple, static training script and instead create a system that is:
* **Reproducible:** Using environment isolation and versioning so that any group member (and examiners) can replicate results.
* **Scalable:** Efficiently managing a dataset of 75,000 images without bloating the version control system.
* **Observable:** Tracking every experiment, hyperparameter tweak, and metric to make data-driven decisions about model selection.
* **Deployable:** Packaging the final solution so that it is ready for real-world inference.

## 2. Dataset
We will be utilizing the [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) available on Kaggle. 
* **Composition:** The dataset contains 75,000 high-resolution images of rice grains, divided equally (15,000 each) into the five target classes.
* **Storage & Versioning:** Since the dataset size is significant, we will not store the images directly in GitHub. Instead, we will use **DVC (Data Version Control)** to track data versions. This allows us to keep our repository lightweight while maintaining a clear history of data transformations and splits.
* **Preprocessing:** Our pipeline will include automated scripts for image resizing, normalization, and data augmentation to improve model generalization.

# Model:
We will implement a CNN architecture, utilizing either a custom-designed model or a pre-trained model. To ensure reproducibility and scalability, the experimentation pipeline will be managed using Hydra for configuration and orchestrated through bash scripts. All experimental results, including hyperparameters and performance metrics, will be logged to wandB.

# Tools:

| Tool | Purpose in this Project |
| :--- | :--- |
| **PyTorch** | **Deep Learning Framework:** We will use PyTorch as our core engine for building and training the CNN. Its dynamic computation graph makes it ideal for iterative experimentation and debugging. We will specifically leverage `torchvision` for pre-trained ResNet models and efficient image transformations. |
| **Docker** | **Environment Isolation:** We will containerize our training and inference code. This ensures the project runs identically across all group members' machines and avoids "dependency hell" |
| **Hydra** | **Configuration Management:** We will use Hydra to manage hyperparameters (learning rate, batch size, etc.) and model architectures via YAML files. This allows us to run different experiments from the command line without modifying the core source code. |
| **GitHub** | **Collaboration & CI/CD:** Beyond hosting our code, we will use GitHub Actions to automate code quality checks (linting) and unit tests every time a new feature is pushed to the repository. |
| **Weights & Biases (WandB)** | **Experiment Tracking:** Every training run will be logged to WandB. This provides our group with a centralized dashboard to visualize training curves, compare different model versions, and store the resulting model artifacts. |


