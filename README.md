<p>
  <strong><span style="font-size:2em;">MobilePlantViT</span></strong>
</p>

---

## **Modules To Roll Out**

---

### 1. Dataset Acquisition and Preprocessing Engine ⚙️

This foundational module will handle the collection, organization, cleaning, and transformation of all image datasets (e.g., Plant Village, CCMT) required for the project.

---

### 2. Custom Architectural Blocks Implementation 🧱

This module involves coding the individual, reusable building blocks of the model, such as the:

- **DepthConv block**
- **GroupConv block**
- **CBAM attention module**
- **Lightweight Linear Self-Attention encoder**

---

### 3. MobilePlantViT Model Assembly 🏗️

Using the custom blocks from the previous module, this component focuses on assembling the complete end-to-end MobilePlantViT hybrid architecture as illustrated in the paper's diagrams and methodology.

---

### 4. Model Training and Validation Pipeline 🚀

This module creates the core script that loads the preprocessed data, feeds it to the model, implements the training loop (with loss calculation and optimization), and runs a validation cycle to monitor performance and save the best model weights.

---

### 5. Experimentation and Performance Evaluation Framework 📊

This component focuses on running the trained model on the test dataset to calculate key performance metrics like accuracy, precision, recall, and F1-score, replicating the result tables in the paper.

---

### 6. Comparative Baseline Analysis 🆚

To meet a key learning outcome, this module involves implementing or using pre-existing versions of MobileViTv1 and MobileViTv2 and running them through your evaluation framework for a direct performance comparison against your model.

---

### 7. Real-Time Inference Engine 💡

This module develops a streamlined function or class that can load your trained MobilePlantViT weights and efficiently perform a disease classification on a single, new input image.

---

### 8. Results Visualization and Reporting Dashboard 📈

This final module creates scripts to generate visual outputs like training/validation accuracy curves, confusion matrices, and misclassification examples, which are crucial for analyzing model behavior and for your final project report.

---