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

- **`DepthConv` block**
- **`GroupConv` block**
- **`CBAM` attention module**
- **Lightweight `Linear Self-Attention` encoder**

---

### 3. MobilePlantViT Model Assembly 🏗️

Using the custom blocks from the previous module, this component focuses on assembling the complete end-to-end `MobilePlantViT` hybrid architecture as illustrated in the paper's diagrams and methodology.

---

### 4. Model Training and Validation Pipeline 🚀

This module creates the core script that loads the preprocessed data, feeds it to the model, implements the training loop (with loss calculation and optimization), and runs a validation cycle to monitor performance and save the best model weights.

---

### 5. Experimentation and Performance Evaluation Framework 📊

This component focuses on running the trained model on the test dataset to calculate key performance metrics like accuracy, precision, recall, and F1-score, replicating the result tables in the paper.

---

### 6. Comparative Baseline Analysis 🆚

To meet a key learning outcome, this module involves implementing or using pre-existing versions of `MobileViTv1` and `MobileViTv2` and running them through your evaluation framework for a direct performance comparison against your model.

---

### 7. Real-Time Inference Engine 💡

This module develops a streamlined function or class that can load your trained `MobilePlantViT` weights and efficiently perform a disease classification on a single, new input image.

---

### 8. Results Visualization and Reporting Dashboard 📈

This final module creates scripts to generate visual outputs like training/validation accuracy curves, confusion matrices, and misclassification examples, which are crucial for analyzing model behavior and for your final project report.

---

## **Module Breakdown**

### 1. Dataset Acquisition and Preprocessing Engine ⚙️

This foundational module is responsible for fetching all raw image data and transforming it into a clean, augmented, and batched format ready for model consumption. A robust and consistent preprocessing pipeline is critical for reproducibility and achieving high performance.

**Key Features:**

- **Automated Data Downloader:**  
  A script to download and extract the required datasets mentioned in the paper: Plant Village, CCMT (Cashew, Cassava, Maize, Tomato), Sugarcane, and Coconut.

- **Standardized Directory Organizer:**  
  A utility to automatically organize the downloaded images into a consistent `train/validation/test` directory structure, applying the specified 70-15-15 split ratio for each dataset.

- **Image Transformation Pipeline:**  
  - Implements a series of transformations that are applied to every image:
  - **Resizing:** All images are uniformly resized to an input shape of `3×224×224`.
  - **Normalization:** Pixels are normalized using the standard ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.

- **Data Augmentation Engine:**  
  - Applies a series of random augmentations only to the training set to improve model generalization, as listed in the paper's experimental setup. 
  - Includes Horizontal Flip, Random 90° Rotation, Shift/Scale/Rotate, Random Gamma, Random Brightness/Contrast, RGB Shift, and CLAHE.

- **PyTorch Data Loaders:**  
  Utilizes PyTorch's `Dataset` and `DataLoader` classes to efficiently load data in batches, handle shuffling for the training set, and parallelize data loading.

---

### 2. Custom Architectural Blocks Implementation 🧱

This module focuses on coding the individual, self-contained building blocks of the MobilePlantViT architecture as distinct `PyTorch nn.Module` classes. Think of these as the custom Lego bricks you'll need before you can build the final model.

**Key Features:**

- **`DepthConv` Block:**  
  A module implementing Depthwise Separable Convolution, which consists of a DepthwiseConv2D layer followed by a PointwiseConv2D (1×1 convolution) layer. This block will also include Batch Normalization and the GELU activation function.

- **`GroupConv` Block:**  
  A module for Group Convolution, where input channels are divided into groups and convolved separately. This block must include an internal residual connection for effective information flow.

- **`CBAM` Module:**  
  A faithful implementation of the Convolutional Block Attention Module, containing two sequential sub-modules: a Channel Attention Module and a Spatial Attention Module, to refine feature maps.

- **`LinearSelfAttention` Block:**  
  A custom attention module implementing the efficient linear self-attention mechanism. It will compute Query (Q), Key (K), and Value (V) representations and derive the context vector with linear complexity, which is a core component of the model's encoder .

---

### 3. MobilePlantViT Model Assembly 🏗️

In this module, you will use the custom blocks from Module 2 to construct the complete, end-to-end MobilePlantViT architecture. This involves connecting the blocks in the correct sequence as specified in the paper's methodology and Figure 1.

**Key Features:**

- **Stem Layer:**  
  The model starts with a DepthConv block that functions as the stem, expanding the input image channels from 3 to 32.

- **Hierarchical Convolutional Body:**  
  A sequence of GroupConv and DepthConv blocks arranged in the specified 1x-2x-4x configuration. Each stage is responsible for downsampling the spatial dimensions while expanding the channel dimensions and is fused with a CBAM module.

- **Patch Embedding Layer:**  
  A DepthConv block is used to convert the final feature maps from the convolutional body into a sequence of flattened patches, which will serve as input to the Transformer encoder.

- **Lightweight Transformer Encoder:**  
  A single (x1) encoder block that applies the custom LinearSelfAttention, Layer Normalization, and a Feedforward Network (FFN) to the sequence of image patches.

- **Classification Head:**  
  The final part of the model that takes the output from the encoder and produces the final predictions. It consists of a Global Average Pooling (GAP) layer, a Dropout layer for regularization, and a final Linear layer for classification.

---

### 4. Model Training and Validation Pipeline 🚀

This module creates the executable engine for training your assembled model. It orchestrates the entire process, from feeding data to the model to updating its weights and periodically evaluating its performance on the validation set.

**Key Features:**

- **Main Training Script (`train.py`):**  
  An executable script that initializes the model, data loaders, optimizer, and loss function. It contains the main training and validation loops.

- **Optimizer & Loss Configuration:**  
  Implements the Adam optimizer with the specified weight decay and uses the Categorical Crossentropy loss function, as detailed in the experimental setup.

- **Learning Rate Scheduler:**  
  Implements a "reduce on plateau" scheduler that reduces the learning rate by half if the validation accuracy does not improve for 10 consecutive epochs.

- **Early Stopping Mechanism:**  
  A crucial feature to prevent overfitting and save time. The training process will be halted if validation accuracy fails to improve for 50 epochs, and the best-performing weights will be saved.

- **Live Performance Logging:**  
  Integrates a logger (e.g., TensorBoard or a simple console/file logger) to track and display key metrics like training/validation loss and accuracy in real-time for each epoch.

- **Model Checkpointing:**  
  Saves the model's state (weights) periodically, with a specific mechanism to always preserve the checkpoint that achieved the best validation accuracy.

---

### 5. Experimentation and Performance Evaluation Framework 📊

This module provides the tools to rigorously assess the performance of your final trained model on the unseen test dataset. It's focused on generating the quantitative results that will prove your model's effectiveness.

**Key Features:**

- **Main Evaluation Script (`evaluate.py`):**  
  A script that loads your best saved model checkpoint and the test data loader to perform a final evaluation.

- **Comprehensive Metric Calculation:**  
  Implements functions to compute and report all the key metrics used in the paper:
  - Overall Accuracy
  - Macro and Weighted averages for Precision, Recall, and F1-score
  - One-vs-Rest (OvR) Area Under the Curve (AUC) score

- **Confusion Matrix Computation:**  
  Generates a confusion matrix for each dataset to allow for in-depth error analysis, especially for identifying which classes are most often confused (e.g., different types of leaf spots).

- **Parameter Count Utility:**  
  A simple function to iterate through the model's layers and sum up all trainable parameters, confirming that your model is lightweight (the paper reports 0.69M parameters).

---

### 6. Comparative Baseline Analysis 🆚

A critical module for fulfilling the project's learning outcomes. Here, you will train and evaluate two other lightweight models (MobileViTv1 and MobileViTv2) under the exact same conditions as your MobilePlantViT to provide a fair and direct performance comparison.

**Key Features:**

- **Baseline Model Integration:**  
  Find and import standard, trusted implementations of MobileViTv1-XXS and MobileViTv2-050 (e.g., from a library like timm).

- **Standardized Training Protocol:**  
  You will train these baseline models using the exact same training pipeline (Module 4) and data splits/augmentations (Module 1) that you used for MobilePlantViT. This ensures the comparison is unbiased.

- **Comparative Results Generation:**  
  Use the evaluation framework from Module 5 to run the trained baselines on the test sets and gather all the same performance metrics.

- **Side-by-Side Reporting:**  
  Create scripts to automatically generate comparison tables, formatted similarly to Table V in the research paper, to clearly display the accuracy, parameter counts, and other metrics for all three models.

---

### 7. Real-Time Inference Engine 💡

This module focuses on creating a clean, callable interface for using your trained model to make predictions on new, individual images. This component is essential for any downstream application, like a web demo or mobile app.

**Key Features:**

- **Weight Loading Function:**  
  A utility that can load the MobilePlantViT architecture and populate it with your best saved weights from the training process.

- **Single-Image Preprocessing:**  
  A function that takes a raw image (e.g., a JPEG file) and applies the necessary transformations (resize to 224x224, normalize, convert to a PyTorch tensor) to make it model-ready.

- **Prediction Function:**  
  A core function that:
  - Takes a preprocessed image tensor as input.
  - Passes it through the loaded model in evaluation mode.
  - Applies a softmax function to the model's output logits to get class probabilities.
  - Returns the top predicted class label and its associated confidence score.

- **Class Label Mapping:**  
  Includes a simple mapping system (e.g., a JSON file) to convert the model's numerical output (e.g., class index 15) into a human-readable name (e.g., 'Tomato_Leaf_Curl_Virus').

---

### 8. Results Visualization and Reporting Dashboard 📈

The final module provides the tools needed to visually analyze your model's behavior and present your findings effectively. A good project is not just about a working model, but also about clearly communicating its results.

**Key Features:**

- **Training Curve Plotter:**  
  A script that uses your training logs to generate and save plots of training vs. validation accuracy and loss over epochs. This is crucial for visualizing model convergence and diagnosing issues like overfitting.

- **Confusion Matrix Plotter:**  
  Uses libraries like Matplotlib or Seaborn to create and save high-quality, labeled heatmaps of the confusion matrices, making it easy to see misclassification patterns.

- **Misclassification Showcase:**  
  A script that identifies images from the test set that were incorrectly classified. It should then generate a visual report showing the image, its true label, and the incorrect predicted label, which is very useful for error analysis.

- **Interactive Demo (Optional):**  
  A simple web application built using a framework like Streamlit or Flask. This app will provide a user interface where someone can upload their own plant leaf image and see the classification result from your inference engine (Module 7). This is an excellent way to demonstrate the practical value of your project.

---