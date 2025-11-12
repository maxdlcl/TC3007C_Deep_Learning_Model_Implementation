# TC3007C Deep Learning Model Implementation

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.x-red.svg)](https://keras.io/)

## 📋 Table of Contents
- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [File Structure](#file-structure)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Author](#author)
- [References](#references)

## 🔬 About the Project

This project implements deep learning models, specifically Convolutional Neural Networks (CNNs), for the classification of medical chest X-ray images. The models are trained to identify one of four conditions:

- **COVID-19** 🦠
- **Pneumonia** 🫁
- **Lung Opacity** 🌫️
- **Normal (Healthy)** ✅

This work is particularly relevant in the current medical context, especially in the field of respiratory disease diagnosis through medical image interpretation. The ability to correctly classify these images has a significant impact on early and accurate disease detection, which can substantially improve treatment outcomes and patient care.

The development of deep learning models for medical image classification is a research area with great relevance and potential to improve access to accurate and timely diagnoses, which could otherwise be costly or inaccessible for certain patients due to lack of medical specialists.

### Objectives

The main objective of this project is to build and evaluate different CNN architectures for medical image classification, in order to identify the architecture that performs best in this specific task.

## 📊 Dataset

The dataset used in this project is the **COVID-19 Radiography Database** from Kaggle.

- **Source:** [COVID-19 Radiography Database on Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- **License:** Copyright Authors

### Dataset Distribution

The latest version of the dataset contains:
- **Normal/Healthy:** 10,192 images
- **COVID-19:** 3,616 images
- **Pneumonia:** 6,012 images
- **Lung Opacity:** 1,345 images

### Data Split

- **Training:** 80%
- **Validation:** 10%
- **Testing:** 10%

### Data Preprocessing

The following preprocessing and augmentation techniques were applied:
- Random rotations
- Horizontal and vertical shifts
- Random zoom
- Horizontal flipping
- Class weight calculation to handle dataset imbalance

## 🧠 Models Implemented

### 1. Simple CNN (Baseline Model)
A simple CNN architecture built from scratch with:
- 3 convolutional layers
- Pooling layers
- Dense layers at the end

**Purpose:** Establish a baseline for comparison with more complex models.

### 2. Deep CNN
An improved architecture with:
- More convolutional and pooling layers
- Intermediate dense layer with more neurons
- Better regularization techniques

**Result:** Slight improvement over the baseline model.

### 3. Transfer Learning with ResNet50V2 (Best Model) ⭐
Leveraging a pre-trained ResNet50V2 model:
- Fine-tuned on the COVID-19 dataset
- Last layers unfrozen for domain adaptation
- Custom dense layers added for the specific classification task

**Result:** Significantly better performance compared to models trained from scratch. This model was selected as the best performer and saved for future use.

### Training Techniques

All models were trained using:
- **Loss Function:** Categorical Cross-Entropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Regularization:**
  - Early Stopping
  - Learning Rate Reduction on Plateau
  - Class Weights for imbalanced data

## 💻 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/maxdlcl/TC3007C_Deep_Learning_Model_Implementation.git
cd TC3007C_Deep_Learning_Model_Implementation
```

2. Install required dependencies:
```bash
pip install tensorflow keras numpy pandas matplotlib scikit-learn kaggle
```

3. Download the dataset:
   - Set up your Kaggle API credentials (kaggle.json)
   - Use the notebook to download the dataset, or manually download from [Kaggle](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

## 🚀 Usage

### Running the Notebook

Open and run the Jupyter notebook:
```bash
jupyter notebook M2_Portafolio.ipynb
```

The notebook contains:
1. **Data loading and preprocessing** - Download and prepare the dataset
2. **Exploratory Data Analysis** - Visualize sample images from each class
3. **Model training** - Train three different CNN architectures
4. **Evaluation** - Assess model performance on test data
5. **Prediction application** - Use the trained model to classify new X-ray images

### Using the Pre-trained Model

The best model (ResNet50V2) is saved as `ResNet50V2_COVID19_Classifier.keras` and can be loaded for inference:

```python
from tensorflow import keras

# Load the model
model = keras.models.load_model('ResNet50V2_COVID19_Classifier.keras')

# Make predictions on new images
predictions = model.predict(your_image_data)
```

## 📈 Results

The ResNet50V2 transfer learning model achieved the best performance:

- **Strong performance** on COVID-19 and Normal classes with minimal confusion
- **Some confusion** between Normal and Lung Opacity classes
- **Solid overall performance** including on the minority Pneumonia class
- The class balancing strategy proved effective across all classes

The confusion matrix analysis shows that the model has good performance in classifying images of healthy patients and those with COVID-19, with few confusions between these classes. However, there are some notable confusions between the normal and lung opacity classes, indicating that the model has difficulty distinguishing between these two conditions.

## 📁 File Structure

```
TC3007C_Deep_Learning_Model_Implementation/
├── M2_Portafolio.ipynb              # Main Jupyter notebook with all implementations
├── ResNet50V2_COVID19_Classifier.keras  # Saved best model (ResNet50V2)
├── README.md                         # This file
└── .gitattributes                    # Git LFS configuration for model file
```

## 🛠️ Technologies Used

- **Python 3.8+** - Programming language
- **TensorFlow 2.x** - Deep learning framework
- **Keras** - High-level neural networks API
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning utilities
- **Kaggle API** - Dataset download
- **Jupyter Notebook** - Interactive development environment

## 🔮 Future Improvements

Potential areas for enhancement include:
- Experimentation with different architectures (EfficientNet, Vision Transformers, etc.)
- Hyperparameter optimization using techniques like Grid Search or Bayesian Optimization
- Implementation of advanced data augmentation techniques (MixUp, CutMix)
- Incorporation of additional training data
- Development of an ensemble model combining multiple architectures
- Deployment as a web application or API for real-world use
- Cross-validation for more robust performance estimation

## 👨‍💻 Author

**Maximiliano De La Cruz Lima**
- Student ID: A01798048
- Course: TC3007C - Inteligencia Artificial Avanzada para la Ciencia de Datos
- Instructor: Dr. Julio Guillermo Arriaga Blumenkron
- Module: M2. Técnicas y arquitecturas de deep learning

## 📚 References

Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C. S., Liang, H., Baxter, S. L., McKeown, A., Yang, G., Wu, X., Yan, F., Dong, J., Prasadha, M. K., Pei, J., Ting, M. Y. L., Zhu, J., Li, C., Hewett, S., Dong, J., Ziyar, I., Shi, A., Zhang, R., Zheng, L., Hou, R., Shi, W., Fu, X., Duan, Y., Zhang, E. D., Zhang, C. L., Li, O., Wang, X., Singer, M. A., Sun, X., Xu, J., Tafreshi, A., Lewis, M. A., Xia, H., & Zhang, K. (2018). Identifying medical diagnoses and treatable diseases by image-based deep learning. *Cell*, 172(5), 1122-1131.e9. https://doi.org/10.1016/j.cell.2018.02.010

---

**Note:** This project is for educational purposes as part of the TC3007C course. The models and predictions should not be used as a substitute for professional medical diagnosis.