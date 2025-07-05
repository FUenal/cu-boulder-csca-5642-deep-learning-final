
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
    git clone https://github.com/FUenal/cu-boulder-csca-5642-deep-learning-final.git
    cd your-repo-name
    ```

2.  **Open the notebook:**
    Open `csca-5642-deep-learning-final-4.ipynb` using Jupyter Lab, Jupyter Notebook, or VS Code.

3.  **Execute cells sequentially:**
    Run all cells in the notebook from top to bottom.

      * The notebook will automatically download and preprocess the datasets into the `data/` directory.
      * It will then train and evaluate the specified baseline and LLM models in few-shot settings.
      * Finally, it will generate and display performance plots and save them to the `outputs/png/` and `outputs/pdf/` directories.

    **Note on Computation Time:** The training of LLMs can be computationally intensive. This notebook has been optimized by reducing the number of LLM training epochs (to 2) and focusing solely on few-shot training sizes to expedite execution. A GPU (e.g., provided by Kaggle or Google Colab Pro) is highly recommended.

### 7\. Results, Discussion, and Conclusion

This section presents the results of the model training and evaluation, focusing on the performance of selected Large Language Models (LLMs) and traditional machine learning baselines in few-shot learning scenarios. The analysis covers F1 scores, training times, and inference times, averaged across the four diverse datasets.

### Analysis of Training and Inference Times

The plots for average training time and average inference time versus the number of training samples provide crucial insights into the computational efficiency of the models:

* **Average Training Time:**
    * As expected, training time generally increases with the number of training samples for all models.
    * The Large Language Models (Phishing-DistilBERT-v2.4.1 and bert-base-uncased) exhibit significantly higher training times compared to the traditional baseline models (XGBoost and LightGBM). This is primarily due to their much larger parameter counts and the complexity of backpropagation through deep neural networks. Even with reduced epochs (2 epochs in this study), LLMs demand substantially more computational resources for training.
    * Among the baselines, XGBoost and LightGBM demonstrate remarkable efficiency, training in fractions of a second even for larger few-shot sample sizes.

* **Average Inference Time:**
    * The average inference times for all models are remarkably fast, often in the order of milliseconds or less, even for the LLMs. This is a critical finding, as it indicates that once trained, even large models can provide near real-time predictions, which is essential for practical phishing detection systems.
    * While LLMs are slower than baselines for inference, their absolute inference times remain very low, making them viable for production environments where speed is important. This suggests that the primary computational bottleneck for LLMs is training, not deployment.

### Analysis of Test F1 Scores (Macro Average)

The F1 score comparison plot is a key indicator of model effectiveness, especially in potentially imbalanced datasets:

   * **Performance Trends:** All models generally show an improvement in F1 score as the number of training samples increases, highlighting the benefit of more data for learning robust patterns.
   * **LLMs in Few-Shot Settings:**
      * Both the fine-tuned Phishing-DistilBERT-v2.4.1 and the non-fine-tuned bert-base-uncased achieve high F1 scores, even with very limited training samples. This demonstrates the power of transfer learning; their pre-trained knowledge allows them to generalize effectively from few examples.
      * The fine-tuned Phishing-DistilBERT-v2.4.1 generally performs at a very high level, often slightly outperforming the generic bert-base-uncased, especially at lower sample counts. This suggests that domain-specific fine-tuning provides a tangible benefit for phishing detection.
   * **Baselines vs. LLMs:**
      * The "Best Baseline Model" (representing the top performance between XGBoost and LightGBM) also achieves very strong F1 scores, demonstrating that highly optimized traditional machine learning models remain competitive, particularly as the number of training samples increases.
       * In some few-shot scenarios, LLMs can surpass baselines due to their deeper contextual understanding. However, the gap might narrow or even reverse at higher data volumes depending on the dataset complexity and the specific LLM's fine-tuning.

These results underscore the potential of LLMs in phishing detection, particularly in data-scarce environments, while also reaffirming the strong performance and efficiency of well-tuned traditional machine learning algorithms.

### 8\. Limitations

While this project provides valuable insights into phishing email detection using LLMs and traditional baselines, it is subject to several limitations:

   * **Dataset Scope:** The study utilized four publicly available datasets, which, while diverse, may not fully represent the constantly evolving landscape of real-world phishing attacks. Modern phishing tactics, such as highly personalized spear phishing, QR code-based phishing (quishing), or advanced business email compromise (BEC) schemes, might not be adequately captured in these older or generalized datasets.
   * **Limited LLM Selection:** For computational efficiency, this study focused on a select few LLMs (one fine-tuned and one non-fine-tuned) and did not include the very latest or largest state-of-the-art models (e.g., Llama 3, newer Gemma variants, Mistral Large). Performance characteristics might vary significantly with different LLM architectures or sizes.
   * **Few-Shot Focus:** While a strength for specific use cases, the exclusive focus on few-shot training sizes (4 to 256 samples) means the study does not fully explore the performance of LLMs or baselines when exposed to very large, real-world scale training datasets (e.g., hundreds of thousands to millions of samples).
   * **Generalization to Real-world Deployment:** The performance observed in a controlled academic setting may not directly translate to real-world deployment, where factors like data drift, zero-day attacks, adversarial examples, and system latency can significantly impact effectiveness.
   * **Absence of Explainability (XAI):** The project does not delve into the interpretability of LLM decisions. Understanding *why* an LLM classifies an email as phishing is crucial for trust, debugging, and identifying novel attack patterns, but this was beyond the scope of this work.

### 9\. Future Workk

Building upon the findings of this project, several avenues for future research and development emerge:

   * **Expanded LLM Evaluation:** Explore a broader range of cutting-edge LLMs, including more recent decoder-only models (e.g., Mistral, Gemma series) and their larger variants. Investigate advanced prompting strategies (e.g., chain-of-thought, self-consistency) for zero-shot or few-shot classification with these models.
   * **Integration of Current Datasets:** Incorporate newer and more specialized phishing datasets that capture the latest attack vectors, such as QR code phishing (quishing), advanced social engineering, and various Business Email Compromise (BEC) schemes. This would ensure higher real-world relevance.
   * **Advanced Fine-Tuning and Domain Adaptation:** Investigate more sophisticated fine-tuning techniques (e.g., LoRA, QLoRA, adapter layers) to efficiently adapt LLMs to specific phishing domains with even less data or computational overhead.
   * **Explainable AI (XAI) for Transparency:** Implement and compare XAI techniques (e.g., LIME, SHAP, attention heatmaps) to provide transparency into how both LLMs and baselines make their classification decisions. This would enhance trust and aid in identifying novel phishing indicators.
   * **Adversarial Robustness Testing:** Conduct systematic studies on the models' robustness against adversarial attacks (e.g., adding typos, character substitutions, paraphrasing) specifically designed to evade detection.
   * **Multimodal Phishing Detection:** Extend the scope to include multimodal analysis, where models process not only text but also embedded images, attachments, and linked content, as phishing attacks increasingly leverage these elements.
   * **Real-time System Prototyping:** Develop a prototype real-time phishing detection system to evaluate the practical implications of integration, latency, and throughput in a live environment.
   * **Ethical Considerations and Bias Analysis:** Conduct a deeper analysis of potential biases in phishing detection, ensuring models do not disproportionately flag legitimate emails from specific demographics or communication styles.

### 10\. Acknowledgments

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
