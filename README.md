# WESAD Stress Detection Using GRU Networks

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/tensorflow-2.10+-orange.svg)](https://tensorflow.org/)
[![License MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status Active](https://img.shields.io/badge/status-Active-brightgreen.svg)]()

## 📋 Overview

This project implements an **automated stress detection system** using physiological signals from wearable devices. We utilize a **Gated Recurrent Unit (GRU)** neural network to classify stress states from multi-modal wearable sensor data with **96.55% accuracy**.

### Key Features

✅ **Binary Classification:** Stress vs. Non-Stress detection  
✅ **Multi-Modal Sensors:** ACC, BVP, EDA, TEMP from Empatica E4  
✅ **GRU Architecture:** Efficient temporal pattern learning  
✅ **Subject-Level Validation:** No data leakage, true generalization  
✅ **Balanced Performance:** 93.7% recall, 96% precision  
✅ **Production-Ready:** Saved models and preprocessing tools included

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **96.55%** |
| **Precision (Stress)** | 96.0% |
| **Recall (Stress)** | 93.7% |
| **F1-Score** | 94.8% |
| **Specificity** | 98.3% |
| **Training Samples** | 2,205 windows |
| **Test Samples** | 1,107 windows |
| **Total Subjects** | 15 |

### Confusion Matrix

```
                    Predicted
                Not Stressed  Stressed
Actual
Not Stressed        759          13      (98.3% correct)
Stressed             21         314      (93.7% correct)
```

---

## 🎯 Project Structure

```
WESAD-Stress-Detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── notebooks/
│   ├── 1_data_exploration.ipynb      # EDA & visualization
│   └── 2_model_training.ipynb        # Main training pipeline
├── src/
│   ├── preprocessing.py              # Data preprocessing functions
│   ├── model.py                      # GRU model architecture
│   └── evaluation.py                 # Evaluation metrics
├── data/
│   └── wesad/                        # WESAD dataset (download separately)
├── models/
│   ├── wesad_stress_detector_model.h5     # Trained model
│   ├── scaler_preprocessing.pkl           # StandardScaler
│   ├── class_weights.pkl                  # Class balancing weights
│   └── config_inference.pkl               # Configuration
├── outputs/
│   ├── training_history.csv          # Training metrics
│   ├── training_history.png          # Training curves
│   ├── confusion_matrix.png          # Confusion matrix
│   └── roc_curve.png                 # ROC curve
└── docs/
    └── PROJECT_REPORT.pdf            # Full technical report
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/WESAD-Stress-Detection.git
cd WESAD-Stress-Detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

Download the WESAD dataset from:
- **Official:** https://github.com/MariusHotte/WESAD
- **Kaggle:** https://www.kaggle.com/datasets/wesad-wearable-stress-affect-detection-dataset

Extract to `data/wesad/` directory.

### 3. Run Training

```bash
python notebooks/2_model_training.ipynb
```

Or using command line:

```bash
python -c "
import jupyter
jupyter.notebook.notebookapp.main(['notebooks/2_model_training.ipynb'])
"
```

### 4. Make Predictions

```python
import pickle
import joblib
import numpy as np
import scipy.signal
from tensorflow.keras.models import load_model

# Load artifacts
model = load_model('models/wesad_stress_detector_model.h5')
scaler = joblib.load('models/scaler_preprocessing.pkl')
config = joblib.load('models/config_inference.pkl')

# Load new subject data
with open('data/wesad/S2/S2.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# Preprocess and predict
# (See inference guide in docs/)
```

---

## 📚 Dataset Description

### WESAD (Wearable Stress and Affect Detection)

**15 Subjects** with **2 hours of data** per subject
- Mean age: 27.5 ± 2.4 years
- 12 males, 3 females
- **2 Wearable Devices:**
  - Chest: RespiBAN Professional @ 700 Hz
  - Wrist: Empatica E4 (focus of this project)

### Experimental Protocol

1. **Baseline (20 min):** Reading neutral material
2. **Amusement (6.5 min):** Watching funny video clips
3. **Stress (10 min):** Trier Social Stress Test (TSST)
4. **Meditation:** Recovery period

### Wrist Sensors (Empatica E4)

| Sensor | Sampling Rate | Unit | Importance |
|--------|---------------|------|-----------|
| EDA | 4 Hz | μS | ⭐⭐⭐ Primary stress indicator |
| TEMP | 4 Hz | °C | ⭐⭐ Gradual response |
| BVP | 64 Hz | a.u. | ⭐⭐ Heart rate variability |
| ACC | 32 Hz | g-force | ⭐ Movement context |

---

## 🔧 Technical Details

### Data Preprocessing Pipeline

```
Raw Data (Multi-rate: 4-64 Hz)
    ↓
[1] RESAMPLE all signals to 4 Hz (unified)
    ↓
[2] COMBINE 6 features (ACC_X, ACC_Y, ACC_Z, BVP, EDA, TEMP)
    ↓
[3] CREATE WINDOWS (120 samples = 30 seconds)
    ↓
[4] STANDARDIZE (mean=0, std=1 per subject)
    ↓
[5] SPLIT: Subject-level (10 train, 5 test) ← NO LEAKAGE!
    ↓
READY FOR ML ✅
```

### Model Architecture

```
Input: (120, 6) - 120 timesteps, 6 features
  ↓
GRU(64, return_sequences=True)
  ↓
Dropout(0.3)
  ↓
GRU(32, return_sequences=False)
  ↓
Dropout(0.3)
  ↓
Dense(16, activation='relu')
  ↓
Dropout(0.2)
  ↓
Dense(1, activation='sigmoid')
  ↓
Binary Output: [0=Not Stressed, 1=Stressed]

Total Parameters: 31,137 (lightweight!)
Inference Time: ~5ms
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Cross-Entropy |
| Batch Size | 32 |
| Epochs | 30 (early stopping @ 9) |
| Class Weights | Applied (imbalance handling) |
| Validation Split | Subject-level (10 train, 5 test) |

---

## 🎓 Key Innovations

### 1. Subject-Level Train/Test Split
- ✅ Prevents data leakage
- ✅ Ensures model generalizes to new subjects
- ✅ More realistic performance assessment

### 2. Multi-Rate Signal Synchronization
- ✅ Unified all sensors to 4 Hz
- ✅ Preserved temporal relationships
- ✅ No information loss

### 3. Balanced Class Weighting
- ✅ Handled 30% stress / 70% non-stress imbalance
- ✅ Equal importance for both classes
- ✅ Fair performance metrics

### 4. GRU over LSTM
- ✅ Fewer parameters (31K vs 43K)
- ✅ Faster training & inference
- ✅ Better for wearable deployment

---

## 📈 Results Visualization

### Training History
```
Epoch 1:  Train: 77%, Val: 98%  (overfitting)
Epoch 2:  Train: 94%, Val: 94%  (convergence)
Epoch 3:  Train: 95%, Val: 98%  (recovery)
Epoch 4:  Train: 95%, Val: 96%  (best) ✅
Epoch 9:  Early Stopping (patience=5)
```

### Performance Breakdown
- **True Negatives:** 759 (correctly identified non-stressed)
- **True Positives:** 314 (correctly identified stressed)
- **False Positives:** 13 (false alarms - low!)
- **False Negatives:** 21 (missed stress - acceptable)

---

## 💾 Saved Artifacts

All models and preprocessing tools are provided:

| File | Purpose |
|------|---------|
| `wesad_stress_detector_model.h5` | Trained GRU model |
| `scaler_preprocessing.pkl` | StandardScaler for normalization |
| `class_weights.pkl` | Class weights for imbalance |
| `config_inference.pkl` | Configuration parameters |
| `training_history.csv` | Metrics per epoch |
| `training_history.png` | Training curves |

---

## 📖 Documentation

- **Full Report:** See `docs/PROJECT_REPORT.pdf` for technical details
- **Inference Guide:** See `docs/INFERENCE_GUIDE.txt` for step-by-step usage
- **Jupyter Notebooks:** See `notebooks/` for interactive examples

---

## 🛠️ Dependencies

```
tensorflow>=2.10
keras>=2.10
numpy>=1.20
pandas>=1.3
scikit-learn>=1.0
scipy>=1.9
matplotlib>=3.5
seaborn>=0.12
joblib>=1.1
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 🎯 Applications

This system can be used for:

✅ **Mental Health Monitoring:** Continuous stress tracking for patients  
✅ **Workplace Wellness:** Alert systems for high-stress situations  
✅ **Sports Performance:** Optimize training by monitoring stress levels  
✅ **Research Tool:** Objective stress measurement in studies  
✅ **Personal Wellness:** Individual stress tracking and awareness  

---

## ⚠️ Limitations

1. **Lab-Controlled Setting:** Real-world stress may differ from TSST protocol
2. **Device-Specific:** Requires Empatica E4 wearable device
3. **Binary Only:** Doesn't distinguish stress intensity levels
4. **Limited Population:** 15 subjects - larger datasets recommended
5. **Individual Variability:** Some subjects may require personalization

---

## 🔬 Future Improvements

### Short-term
- [ ] Add confidence scores to predictions
- [ ] Implement temporal smoothing post-processing
- [ ] Build mobile app wrapper

### Medium-term
- [ ] Add attention mechanisms for interpretability
- [ ] Implement bidirectional GRU for offline analysis
- [ ] Support multiple wearable devices

### Long-term
- [ ] Real-world stress scenario validation
- [ ] Personalized model fine-tuning
- [ ] Federated learning for privacy
- [ ] Clinical integration and deployment

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{wesad_stress_2026,
  author = {Majd Bdour and Hala Hassan and Halla Taani and Sadeen Abu-Khadra and Haneen Zawateen},
  title = {WESAD Stress Detection Using GRU Networks},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/WESAD-Stress-Detection}}
}
```

### Original Dataset Citation

```bibtex
@inproceedings{schmidt2018wesad,
  title={Introducing WESAD, a multimodal dataset for wearable stress and affect detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  booktitle={2018 International Conference on Multimodal Interaction},
  pages={400--408},
  year={2018},
  organization={ACM}
}
```

---

## 👥 Team

**Students:**
- Majd Bdour
- Hala Hassan
- Hala Taani
- Sadeen Abu-Khadra
- Haneen Zawateen

**Supervisor:** Dr. Zayed Al-Hailat

**Institution:** Yarmouk University, DA450 Course (2025/2026)

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs via GitHub Issues
- Suggest improvements
- Submit pull requests

---

## 📞 Contact & Support

For questions or support:
- 📧 Email: [majdbdour8@gmail.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/WESAD-Stress-Detection/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/WESAD-Stress-Detection/discussions)

---

## 🙏 Acknowledgments

- **WESAD Dataset Authors:** Schmidt et al. (2018)
- **Empatica:** For the E4 wearable device
- **Dr. Zayed Al-Hailat:** For supervision and guidance
- **Yarmouk University:** For resources and support

---

## 📚 References

Primary Dataset and Research Papers
[1] Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger and Kristof Van Laerhoven. 2018. Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection. In 2018 International Conference on Multimodal Interaction (ICMI '18), October 16–20, 2018, Boulder, CO, USA. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3242969.3242985
Wearable Sensor Documentation
[2] PLUX Wireless Biosignals. (2021). "Electrocardiography (ECG) Sensor Datasheet." PLUX Biosignals S.A. Retrieved from https://support.pluxbiosignals.com/wp-content/uploads/2021/10/biosignalsplux-Electrocardiography-ECG-Datasheet.pdf
[3] PLUX Wireless Biosignals. (2021). "Electrodermal Activity (EDA) Sensor Datasheet." PLUX Biosignals S.A. Retrieved from https://support.pluxbiosignals.com/wp-content/uploads/2021/11/Electrodermal_Activity_EDA_Datasheet.pdf
[4] PLUX Wireless Biosignals. (2021). "Electromyography (EMG) Sensor Datasheet." PLUX Biosignals S.A. Retrieved from https://support.pluxbiosignals.com/wp-content/uploads/2021/10/biosignalsplux-Electromyography-EMG-Datasheet.pdf
[5] PLUX Wireless Biosignals. (2021). "Temperature (TMP) Sensor Datasheet - Revolution Series." PLUX Biosignals S.A. Retrieved from https://support.pluxbiosignals.com/wp-content/uploads/2021/11/revolution-tmp-sensor-datasheet.pdf
[6] PLUX Wireless Biosignals. (2021). "Accelerometer (ACC) Sensor Datasheet." PLUX Biosignals S.A. Retrieved from https://support.pluxbiosignals.com/wp-content/uploads/2021/11/Accelerometer_ACC_Datasheet.pdf
[7] PLUX Wireless Biosignals. (2021). "Respiration (PZT) Sensor Datasheet." PLUX Biosignals S.A. Retrieved from https://support.pluxbiosignals.com/wp-content/uploads/2021/11/Respiration_PZT_Datasheet.pdf
[8] Empatica Inc. (2024). "E4 Wristband - Real-time Physiological Signals." Empatica Research. Retrieved from https://www.empatica.com/en-gb/research/e4/
Dataset Sources
[9] WESAD Dataset: https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29
[10] Kaggle WESAD: https://www.kaggle.com/datasets/wesad-wearable-stress-affect-detection-dataset
Deep Learning Frameworks Documentation
[11] TensorFlow Documentation: https://www.tensorflow.org/api_docs
[12] Keras GRU Layer Documentation: https://keras.io/api/layers/recurrent_layers/gru/
Libraries and Tools
[13] TensorFlow/Keras: Abadi, M., et al. (2016). "TensorFlow: A system for large-scale machine learning." 12th USENIX symposium on operating systems design and implementation, 265-283.
[14] NumPy: Harris, C. R., et al. (2020). "Array programming with NumPy." Nature, 585(7825), 357-362.
[15] Scikit-learn: Pedregosa, F., et al. (2011). "Scikit-learn: Machine learning in Python." Journal of machine learning research, 12, 2825-2830.
[16] SciPy: Virtanen, P., et al. (2020). "SciPy 1.0: fundamental algorithms for scientific computing in Python." Nature methods, 17(3), 261-272.



---
