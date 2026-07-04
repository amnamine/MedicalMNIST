# Medical MNIST AI

A deep learning project for classifying medical images by imaging modality. It includes a **Kaggle training notebook** that builds a CNN on the [Medical MNIST](https://doi.org/10.17632/8hdt269s7r) dataset, exports a Keras model, and a **Tkinter desktop app** for single-image inference with confidence scores.

The project is intended for **learning and experimentation**, not clinical diagnosis.

## Project Overview

Medical MNIST contains **58,954** images across **6 modalities**, preprocessed to **64×64** pixels. The training pipeline in `MedMNIST_TrainCode.ipynb` uses an 80/20 train/validation split (47,164 / 11,790 images), trains a custom CNN for 10 epochs, and reaches **99.91% validation accuracy**.

### Supported Classes

| Class | Description |
|-------|-------------|
| **AbdomenCT** | Abdominal CT scans |
| **BreastMRI** | Breast MRI images |
| **CXR** | Chest X-Ray images |
| **ChestCT** | Chest CT scans |
| **Hand** | Hand X-Ray images |
| **HeadCT** | Head CT scans |

---

## Repository Structure

```
MEDICAL_MNISTT/
├── MedMNIST_TrainCode.ipynb   # Training, evaluation, and model export
├── tkinterINTERFACE.py        # Desktop GUI for inference
├── medicalmnist.keras         # Exported model used by the GUI (~8 MB)
└── README.md
```

| File | Role |
|------|------|
| `MedMNIST_TrainCode.ipynb` | Loads the dataset, trains the CNN, plots metrics, saves the model |
| `tkinterINTERFACE.py` | Loads the model and runs predictions on user-selected images |
| `medicalmnist.keras` | Keras model loaded by the GUI (`tf.keras.models.load_model`) |

> **Model filename:** The notebook saves the trained weights as `medical_mnist_cnn_model.keras`. Rename or copy that file to `medicalmnist.keras` (or update the path in `tkinterINTERFACE.py`) before running the GUI.

---

## Requirements

### Inference (GUI)

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- **Tkinter** (included with most standard Python installations on Windows)

```bash
pip install tensorflow pillow numpy
```

### Training (notebook)

Additional libraries used in `MedMNIST_TrainCode.ipynb`:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

A **GPU** is recommended for training (the notebook was run on dual Tesla T4 GPUs on Kaggle). CPU training will work but be slower.

---

## Training the Model

Training logic lives in [`MedMNIST_TrainCode.ipynb`](MedMNIST_TrainCode.ipynb).

### 1. Get the dataset

Download the [Medical MNIST dataset](https://www.kaggle.com/datasets/andrewmvd/medical-mnist) from Kaggle, or point `data_dir` in the notebook to your local copy. The notebook expects a folder-per-class layout:

```
medical-mnist/
├── AbdomenCT/
├── BreastMRI/
├── CXR/
├── ChestCT/
├── Hand/
└── HeadCT/
```

On Kaggle, the notebook auto-detects:

- `/kaggle/input/medical-mnist`
- `/kaggle/input/datasets/andrewmvd/medical-mnist`

### 2. Run the notebook

Open and execute all cells in Jupyter or on [Kaggle Notebooks](https://www.kaggle.com/code). The notebook will:

1. Load images as **grayscale** 64×64 tensors and rescale pixels to `[0, 1]`
2. Split data 80% train / 20% validation (`seed=42`)
3. Train the CNN for **10 epochs** (`batch_size=256`)
4. Print a classification report and plot accuracy, loss, and a confusion matrix
5. Visualize predictions on 10 random validation images
6. Save the model as `medical_mnist_cnn_model.keras`

### Model Architecture

Custom **Sequential CNN** (TensorFlow/Keras):

| Layer | Details |
|-------|---------|
| Conv2D | 32 filters, 3×3, ReLU — input `(64, 64, 1)` |
| MaxPooling2D | 2×2 |
| Conv2D | 64 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Conv2D | 128 filters, 3×3, ReLU |
| MaxPooling2D | 2×2 |
| Flatten | — |
| Dense | 128 units, ReLU |
| Dropout | 0.3 |
| Dense | 6 units, Softmax |

- **Optimizer:** Adam  
- **Loss:** `sparse_categorical_crossentropy`  
- **Metric:** accuracy  

### Training Results

| Metric | Value |
|--------|-------|
| Training samples | 47,164 |
| Validation samples | 11,790 |
| Epochs | 10 |
| Batch size | 256 |
| Final validation accuracy | **99.91%** |
| Per-class precision / recall / F1 | 1.00 for all 6 classes |

After training, copy or rename the saved file to `medicalmnist.keras` for use with the GUI.

---

## Running the GUI

1. Ensure `medicalmnist.keras` is in the same folder as `tkinterINTERFACE.py`.
2. Start the app:

```bash
python tkinterINTERFACE.py
```

3. In the window:
   - **Load Image** — select an image file (preview shown at 300×300).
   - **Predict** — run inference and display the predicted modality with confidence (%).
   - **Reset** — clear the preview and result.

### Inference Pipeline

When you click **Predict**, the app:

1. Resizes the loaded image to **64×64**.
2. Normalizes pixel values to `[0, 1]` (divide by 255).
3. Adds a batch dimension and calls `model.predict()`.
4. Returns the class with the highest softmax probability.

---

## About the Medical MNIST Dataset

The [Medical MNIST dataset](https://doi.org/10.17632/8hdt269s7r) (Kus, 2022) contains 58,954 images derived from several medical imaging sources, preprocessed into a uniform 64×64 format suitable for classification benchmarks. It is widely used for teaching CNN-based medical image analysis.

This project is **not** affiliated with the dataset authors; it is a student/learning implementation built for that task.

---

## Limitations & Disclaimer

- **Not for clinical use.** Predictions are for educational purposes only and must not be used for medical decisions.
- **Single-image inference.** The GUI processes one image at a time; batch evaluation is only in the notebook.
- **Input format.** Training uses **grayscale** (`color_mode="grayscale"`). The GUI loads images as RGB via Pillow; for best results, use images similar to the Medical MNIST training set.
- **Generalization.** Performance on images outside the training distribution (different scanners, resolutions, or modalities) may be poor.

---

## License

If you publish or share this project, add an explicit license file. The Medical MNIST dataset has its own terms on [Mendeley Data](https://doi.org/10.17632/8hdt269s7r) and [Kaggle](https://www.kaggle.com/datasets/andrewmvd/medical-mnist); check those before redistributing data or derivatives.
