# Customer Support Sentiment Analysis Pipeline & Architectural Reflection

A production-ready NLP and Sequence Modeling project designed to process text-based customer support requests, perform lexical normalization, evaluate robust multi-class baselines, and capture temporal semantics using Deep Learning architectures.

---

## 📂 Project Architecture & Workspace Layout

The repository follows an organized layout structure for experimental isolation and persistent logging:

```text
part-3-nlp-sequence-modeling/
│
├── README.md               # Complete project documentation and synthesis
├── notebook.ipynb          # End-to-end executable experimental pipeline
├── requirements.txt        # Pinned project dependencies
└── results/
    ├── task1_data_distribution.png     # Metric distributions (Class counts vs. Word density)
    ├── model_evaluation.png            # Visual performance diagnostics (Confusion Matrix/History)
    └── sample_predictions.txt          # Qualitative model validation spot-check log

## 🛠️ Pipeline Modules & Task Execution

### Task 1: Dataset Understanding & Exploration
* **Objective:** Conduct an automated structural and statistical audit on raw unstructured text features.
* **Technical Details:** * Ingested a synthetic dataset containing 1,500 categorical items.
  * Verified categorical structural balance across targeted evaluation dimensions: `positive` (479 records), `neutral` (524 records), and `negative` (497 records).
  * Determined a baseline maximum vector length limit with an observed average text density of 12.72 words per record.
  * Exported exploratory charts safely to disk using a **Save-Then-Show** memory execution protocol.

### Task 2: Text Preprocessing & Cleaning
* **Objective:** Clean raw customer text to lower downstream vector sparsity.
* **Technical Details:**
  * Implemented an isolated text cleanup function using regular expressions (`re`).
  * Enforced standard optimization techniques: forced lowercase mapping, non-alphabetic character removal, whitespace normalization, and full NLTK stop-word stripping.
  * Preserved the foundational semantic intent while removing noisy syntax, preparing clean strings for downstream mathematical encoding layers.

### Task 3: Feature Extraction (TF-IDF)
* **Objective:** Convert normalized text sequences into stable, continuous numerical representations.
* **Technical Details:**
  * Initialized scikit-learn's `TfidfVectorizer` capped with a `max_features=5000` boundary condition.
  * Calculated mathematical Term Frequency-Inverse Document Frequency weightings to penalize high-frequency context filler words while magnifying unique vocabulary markers.
  * Split the dataset into fixed training splits (`80%`) and test splits (`20%`) using a constant seed configuration to ensure stable performance benchmarks.

### Task 4: Baseline Model Training & Evaluation
* **Objective:** Establish a robust statistical baseline for sentence classification.
* **Technical Details:**
  * Trained a multi-class `LogisticRegression` algorithm using L2 regularization weight penalization.
  * Captured global evaluation criteria (Precision, Recall, F1-Score, and Accuracy) using scikit-learn metrics.
  * Rendered a clean `Confusion Matrix Heatmap` using Seaborn, saving the output array directly to `results/model_evaluation.png` to trace classification boundaries.

### Task 5: Deep Learning (LSTM) for Sequence Modeling
* **Objective:** Build an advanced deep learning sequence network to maintain text dependencies.
* **Technical Details:**
  * Constructed a Keras sequential network utilizing a structural text pipeline: `Tokenization` $\rightarrow$ `Sequencing` $\rightarrow$ `Zero-Padding`.
  * **Layer Architecture:** 1. `Embedding Layer`: Maps sparse word arrays into low-dimensional dense continuous spaces.
    2. `Bidirectional LSTM Layer`: Tracks hidden historical context vectors backward and forward across the phrase structure.
    3. `Dense Layer`: Applies a Softmax activation map over output logits to produce continuous probability values across categories.
  * Optimized using the `Adam` gradient descent optimizer paired with `SparseCategoricalCrossentropy` loss computation.

### Task 6: Theoretical Reflections & Paradigms
* **RNN Processing Limitations:** Vanishing gradients restrict classic Recurrent Networks on long strings, because multiplying the weight matrices repeatedly over time causes learning steps to shrink exponentially toward zero.
* **LSTM Gate Control:** LSTMs bypass this issue using a unique "Cell State" linear path regulated by sigmoidal Forget, Input, and Output gates, allowing information to travel long distances without losing signal strength.
* **Attention Solution:** Attention mechanisms eliminate the structural bottleneck of compressing entire long phrases into a single hidden vector by allowing the system to focus weights directly on key source terms at every step.
* **Transformer Scale:** Transformers replace sequential processing paths entirely with parallelized Multi-Head Self-Attention layers ($O(1)$ operations), enabling vast training scale on internet-sized data arrays.

---

## 📈 Performance Artifacts Summary

* **`results/model_evaluation.png`**: Contains a high-resolution performance plot tracking either the baseline classification confusion thresholds or the deep learning training vs. validation performance lines across optimization loops.
* **`results/sample_predictions.txt`**: Logs real-time text outputs showing input messages alongside cleaned strings, prediction probabilities, and true target categories to enable easy manual quality audits.

---

## 🚀 Installation & Reproduction Steps

To run and reproduce the entire experimental pipeline locally or within a cloud ## 🚀 How to Run the Project in Google Colab

Since this project relies on persistent Google Drive storage for datasets and artifact logging, follow these steps to execute the pipeline:

1. **Upload the Notebook:** Open Google Colab and upload the `notebook.ipynb` file.
2. **Prepare the Data Directory:** Ensure your source customer support dataset is uploaded to your Google Drive at the following path:
   `📂 /content/drive/My Drive/ai_project_synthetic_datasets/part_3_nlp_sequence_modeling/`
3. **Execute the Cells:** Select **Runtime > Run All** from the top menu. 
4. **Authorize Drive Access:** When prompted by the notebook, click "Connect to Google Drive" to allow the script to verify your directories and write your evaluation assets.
5. **Verify Outputs:** Once execution finishes, check your Drive's `results/` folder to view the freshly generated `model_evaluation.png` and `sample_predictions.txt` logs.

# Install the exact pinned operational software versions
pip install -r requirements.txt