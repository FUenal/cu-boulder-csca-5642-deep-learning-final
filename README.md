
# Phishing Email Detection with Large Language Models

This repository contains a Jupyter notebook for a project investigating the effectiveness of Large Language Models (LLMs) in detecting phishing emails, particularly within few-shot learning scenarios. Inspired by the methodology outlined in the "Spam-T5: Benchmarking Large Language Models for Few-Shot Email Spam Detection" paper by Labonne and Moran, this project provides a self-contained environment for data preprocessing, model training, evaluation, and visualization.

## Table of Contents

1.  [Problem Description](#1-problem-description)
2.  [Project Scope & Enhancements](#2-project-scope--enhancements)
3.  [Dataset Provenance](#3-dataset-provenance)
4.  [Models Used](#4-models-used)
5.  [Setup and Dependencies](#5-setup-and-dependencies)
6.  [Running the Notebook](#6-running-the-notebook)
7.  [Results and Discussion](#7-results-and-discussion)
8.  [Conclusion](#8-conclusion)
9.  [Future Work](#9-future-work)
10. [License](#10-license)
11. [Acknowledgments](#11-acknowledgments)

---

### 1. Problem Description

Phishing attacks delivered via email continue to be a significant cybersecurity threat, leading to various forms of compromise. Traditional detection methods often struggle with the evolving sophistication of these attacks. This project explores the utility of Large Language Models (LLMs) in enhancing phishing email detection, especially in low-data (few-shot) environments where labeled examples are scarce. It aims to benchmark the performance of selected fine-tuned and non-fine-tuned LLMs against established machine learning baselines to determine their suitability for this critical task.

### 2. Project Scope & Enhancements

This project builds upon the methodology of the "Spam-T5" paper and includes the following specific adaptations and enhancements:

* **Focused LLM Comparison:** Instead of the original LLMs, this project specifically evaluates:
    * One pre-trained, **fine-tuned LLM**: `cybersectony/phishing-email-detection-distilbert_v2.4.1`
    * One **non-fine-tuned LLM**: `bert-base-uncased`
* **Reduced Baseline Models:** The comparison set of traditional machine learning baselines has been streamlined to focus on efficient and strong performers: XGBoost and LightGBM.
* **Few-Shot Learning Emphasis:** The training and evaluation primarily focus on few-shot learning scenarios, using small, specific numbers of training samples (4, 8, 16, 32, 64, 128, 256) across multiple datasets. The full dataset training (0.8 ratio) for LLMs has been excluded to optimize for computational efficiency.
* **Comprehensive Evaluation Metrics:** Performance is rigorously assessed using F1 score, Precision, Recall, and Accuracy.
* **EDA Visualizations:** Includes exploratory data analysis visualizations such as pie charts for dataset distribution and text length/word frequency plots.
* **Performance Benchmarking:** Detailed analysis and visualizations of training time and inference time across all models and training sample sizes.
* **Key Results Visualization:** A primary plot comparing F1 scores of selected LLMs and the best baseline model across varying training samples.

### 3. Dataset Provenance

The project utilizes four publicly available and widely-used email/SMS spam datasets, automatically downloaded and preprocessed within the notebook:

* **Enron Spam Data:** Derived from the Enron email dataset, manually labeled for spam detection.
    * Source: [https://github.com/MWiechmann/enron_spam_data](https://github.com/MWiechmann/enron_spam_data)
* **Ling-Spam Data:** A collection of emails from the Ling-Spam corpus, categorized as spam or ham.
    * Source: [https://github.com/oreilly-japan/ml-security-jp](https://github.com/oreilly-japan/ml-security-jp)
* **SMS Spam Collection:** A public dataset of SMS messages tagged as legitimate (ham) or spam.
    * Source: [https://archive.ics.uci.edu/ml/datasets/sms+spam+collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
* **SpamAssassin Public Corpus:** A well-known dataset used for spam filtering research.
    * Source: [https://spamassassin.apache.org/old/publiccorpus/](https://spamassassin.apache.org/old/publiccorpus/)

### 4. Models Used

**Large Language Models (LLMs):**
* **Fine-tuned:**
    * `cybersectony/phishing-email-detection-distilbert_v2.4.1` (DistilBERT-based for sequence classification)
* **Non-fine-tuned:**
    * `bert-base-uncased` (Standard BERT model for sequence classification)

**Baseline Machine Learning Models:**
* **XGBoost Classifier:** (`xgboost.XGBClassifier`)
* **LightGBM Classifier:** (`lightgbm.LGBMClassifier`)

### 5. Setup and Dependencies

This project is designed to run in a Jupyter environment (e.g., Kaggle, Google Colab, or local Jupyter/VS Code).

**Prerequisites:**
* Python 3.8+

**Installation:**
All required Python packages can be installed by running the first few code cells in the `csca-5642-deep-learning-final-4.ipynb` notebook. The core dependencies include:

```bash
pip install --upgrade \
    torch \
    torchvision \
    torchaudio \
    transformers \
    datasets \
    evaluate \
    setfit \
    accelerate \
    sentence-transformers \
    numpy==1.26.4 \
    scipy==1.13.0 \
    scikit-learn==1.4.2 \
    ipykernel==6.29.3 \
    matplotlib==3.8.4 \
    matplotlib-inline==0.1.6 \
    xgboost==2.0.3 \
    lightgbm==4.3.0 \
    catboost==1.2.2 \
    nltk==3.8.1 \
    pyarrow==16.1.0 \
    pandas==2.2.2 \
    scienceplots==2.0.1
````

### 6\. Running the Notebook

To run the project:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-github-username/your-repo-name.git](https://github.com/your-github-username/your-repo-name.git)
    cd your-repo-name
    ```

    (Remember to replace `your-github-username/your-repo-name.git` with your actual repository URL).

2.  **Open the notebook:**
    Open `csca-5642-deep-learning-final-4.ipynb` using Jupyter Lab, Jupyter Notebook, or VS Code.

3.  **Execute cells sequentially:**
    Run all cells in the notebook from top to bottom.

      * The notebook will automatically download and preprocess the datasets into the `data/` directory.
      * It will then train and evaluate the specified baseline and LLM models in few-shot settings.
      * Finally, it will generate and display performance plots and save them to the `outputs/png/` and `outputs/pdf/` directories.

    **Note on Computation Time:** The training of LLMs can be computationally intensive. This notebook has been optimized by reducing the number of LLM training epochs (to 2) and focusing solely on few-shot training sizes to expedite execution. A GPU (e.g., provided by Kaggle or Google Colab Pro) is highly recommended.

### 7\. Results and Discussion

*(This section will be populated after you've interpreted your results in the notebook. It will highlight key findings from your plots: e.g., how F1 score changes with data, training/inference time comparisons, LLM vs. Baseline performance, the efficiency of inference, etc.)*

### 8\. Conclusion

*(This section will summarize your findings, reiterate the suitability of LLMs for phishing detection, especially in few-shot settings, and briefly mention the trade-offs observed.)*

### 9\. Future Work

  * Explore more recent and efficient LLM architectures (e.g., Mistral, Gemma models for classification via prompting).
  * Integrate additional, more current phishing datasets that capture the latest attack vectors.
  * Investigate advanced fine-tuning techniques or domain adaptation methods for LLMs in cybersecurity.
  * Perform deeper error analysis on misclassified emails to understand model limitations.
  * Quantify the carbon footprint of training and inference for different model sizes.

### 10\. License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

### 11\. Acknowledgments

  * Inspired by "Spam-T5: Benchmarking Large Language Models for Few-Shot Email Spam Detection" by Labonne and Moran.
  * Datasets sourced from respective public repositories (links provided in the notebook).
  * Hugging Face Transformers library and ecosystem.

## Author 

**Fatih Uenal**

🎓 MSc Computer Science & AI, CU Boulder

🌐 Webpage: https://dataiq.netlify.app/

🔗 LinkedIn: https://www.linkedin.com/in/fatih-uenal/

📊 Kaggle Notebook: [https://www.kaggle.com/code/fatihuenal/csca-5632-unsupervised-algorithms-final-2](https://www.kaggle.com/code/fatihuenal/csca-5642-deep-learning-final-3)

<!-- end list -->

```
```
