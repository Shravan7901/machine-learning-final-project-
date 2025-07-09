#  Sleep Disorder Prediction Using Machine Learning

This project uses machine learning algorithms to predict whether a person has a sleep disorder (Insomnia or Sleep Apnea) based on lifestyle and health features. The objective is to create a binary classification model that can assist in early detection of sleep disorders.

---

##  Dataset Used

- **Dataset Name**: Sleep Health and Lifestyle Dataset
- **Target Variable**: Sleep Disorder (None, Insomnia, Sleep Apnea)
- **Data Source**: Provided as part of course project
- **Format**: CSV file (`Sleep_health_and_lifestyle_dataset.csv`)

We converted the multi-class target into binary:
- `0`: No disorder  
- `1`: Has disorder (Insomnia or Sleep Apnea)

---

##  Key Components of the Project

### 1. **Data Preprocessing**
- Removed missing values
- Encoded categorical columns using LabelEncoder
- Scaled numerical columns using StandardScaler
- Converted target variable into binary

### 2. **Model Training**
We trained the following classification models:
- Logistic Regression
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest

### 3. **Model Evaluation**
Used the following evaluation metrics:
- Accuracy
- Precision, Recall, F1-Score
- Confusion Matrix
- ROC AUC Score
- ROC Curve visualization
- Model comparison chart

### 4. **Visualization**
Visualized key findings using:
- Count plots
- Heatmaps
- Confusion matrices
- ROC curves
- Model comparison bar chart

---

## ▶ How to Run This Code

### Option 1: On Google Colab
1. Open the `.ipynb` notebook in Google Colab
2. Upload the dataset: `Sleep_health_and_lifestyle_dataset.csv`
3. Run the notebook from top to bottom — outputs, plots, and evaluations will be shown.

### Option 2: On Local Jupyter Notebook
1. Install required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

2. Run the notebook using Jupyter or VSCode
3. Make sure the dataset is in the same folder

---

##  Best Performing Model

- **Model**: Random Forest
- **Accuracy**: 88%
- **ROC AUC Score**: 96%
- **Most Important Features**: Stress level, BMI, Physical activity level

---

##  Team Members

- Nallavelli Shravan Kumar  
- Shyam Verma  
- Pooja Shinde  
- Professor: Dr. Itauama  
- Northwood University

---

##  Notes

- We did not apply hyperparameter tuning; default model parameters were used.
- The dataset is small but still gave strong results.
- This project was done purely for academic and learning purposes.

---

##  Report Format Reference

This submission follows guidelines inspired by academic ML conferences like:  
**[ICML](https://icml.cc/)** and **[AAAI](https://aaai.org/)**
