# Medical MNIST AI

A desktop application for classifying medical images by imaging modality. The project uses a pre-trained Keras model and a Tkinter graphical interface to load an image, run inference, and display the predicted class with a confidence score.

## Project Overview

This repository provides an inference-only workflow for **6-class medical image classification**. Images are resized to **64×64** pixels, normalized, and passed to a TensorFlow/Keras model trained on the [Medical MNIST](https://doi.org/10.17632/8hdt269s7r) dataset — a collection of ~59,000 MNIST-style medical images across six modalities.

The app is intended for **learning and experimentation**, not clinical diagnosis.

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
├── tkinterINTERFACE.py   # Desktop GUI application
├── medicalmnist.keras    # Pre-trained Keras model (~8 MB)
└── README.md
```

| File | Role |
|------|------|
| `tkinterINTERFACE.py` | Loads the model, opens the GUI, and runs prediction on user-selected images |
| `medicalmnist.keras` | Saved Keras model used at startup (`tf.keras.models.load_model`) |

> **Note:** Training code and the raw dataset are not included in this repository. Only the exported model and the inference interface are provided.

---

## Requirements

- Python 3.8+
- [TensorFlow](https://www.tensorflow.org/)
- [NumPy](https://numpy.org/)
- [Pillow](https://python-pillow.org/)
- **Tkinter** (included with most standard Python installations on Windows)

Install dependencies:

```bash
pip install tensorflow pillow numpy
```

---

## How to Use

1. Clone or download this repository.
2. Ensure `medicalmnist.keras` is in the same folder as `tkinterINTERFACE.py`.
3. Run the application:

```bash
python tkinterINTERFACE.py
```

4. In the window:
   - **Load Image** — pick an image file from your computer (displayed at 300×300 in the preview).
   - **Predict** — run the model and show the predicted modality and confidence (%).
   - **Reset** — clear the preview and result text.

### Inference Pipeline

When you click **Predict**, the app:

1. Resizes the loaded image to **64×64** (model input size).
2. Converts pixel values to the range `[0, 1]` by dividing by 255.
3. Adds a batch dimension and calls `model.predict()`.
4. Returns the class with the highest softmax probability.

---

## Model

- **Format:** Keras (`.keras`)
- **File:** `medicalmnist.keras`
- **Input:** 64×64 RGB image, values in `[0, 1]`
- **Output:** 6-class probability vector (AbdomenCT, BreastMRI, CXR, ChestCT, Hand, HeadCT)

The model is loaded once when the script starts. Keep the `.keras` file next to the script so the relative path resolves correctly.

---

## About the Medical MNIST Dataset

The [Medical MNIST dataset](https://doi.org/10.17632/8hdt269s7r) (Kus, 2022) contains 58,954 images derived from several medical imaging sources, preprocessed into a uniform 64×64 format suitable for classification benchmarks. It is widely used for teaching CNN-based medical image analysis.

This project is **not** affiliated with the dataset authors; it is a student/learning implementation built on top of a trained model for that task.

---

## Limitations & Disclaimer

- **Not for clinical use.** Predictions are for educational purposes only and must not be used for medical decisions.
- **Single-image inference.** The GUI processes one image at a time; there is no batch mode or dataset evaluation script in this repo.
- **No training included.** To retrain or fine-tune, you need the Medical MNIST dataset and a separate training notebook or script.
- **Generalization.** Performance on images outside the training distribution (different scanners, resolutions, or modalities) may be poor.

---

## License

If you publish or share this project, add an explicit license file. The Medical MNIST dataset has its own terms on [Mendeley Data](https://doi.org/10.17632/8hdt269s7r); check those before redistributing data or derivatives.
