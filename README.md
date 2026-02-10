# Disease Prediction from Medical Data

This project implements machine learning models to predict the possibility of various diseases based on patient data.

## 🚀 Overview
The system utilizes classification techniques on structured medical datasets to assist in early disease detection.

### Supported Diseases
- **Breast Cancer**: Predicts malignant vs benign tumors.
- **Heart Disease**: Detects presence of cardiovascular disease.
- **Diabetes**: Predicts onset of diabetes based on diagnostic measurements.

## 🧠 Algorithms
We compare four powerful classification algorithms:
1. **Logistic Regression**: A robust baseline for binary classification.
2. **Support Vector Machine (SVM)**: Effective in high-dimensional spaces.
3. **Random Forest**: An ensemble method that reduces overfitting.
4. **XGBoost**: Gradient boosted decision trees for state-of-the-art performance.

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/anshikatiwari1813/codealpha-disease-prediction.git
cd codealpha-disease-prediction

# Install dependencies
pip install -r requirements.txt
```

## 📈 Usage

### 1. Local Development (Flask)
Launch the Flask development server:
```bash
python api/index.py
```
Open `http://127.0.0.1:5000` in your browser.

### 2. Deployment (Vercel)
This project is pre-configured for **Vercel**. Just connect your GitHub repository to Vercel, and it will automatically deploy using the provided `vercel.json`.

## 🎨 Design
The web interface features a modern **Glassmorphism** design with:
- Dynamic form generation based on medical datasets.
- Real-time predictions with confidence scores.
- Responsive layout using Bootstrap 5.

---
*Built as part of the CodeAlpha Internship task.*
