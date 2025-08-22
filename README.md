# Capstone_Project
# 🧬 Integrative Analysis of Breast Cancer Outcomes Using Machine Learning

**Term:** Spring 2025  
**Team:** Purple (Meghana Bodduluri, Sanjana Kanneganti, Sowmya Bathula, Swetha Nunemunthala)  
**Keywords:** Python, pandas, scikit-learn, TensorFlow, SHAP, Gradio  

---
## 📖 Project Overview:
<P>Breast cancer is one of the most common cancers among women worldwide. Early detection and personalized treatment planning are critical for improving patient outcomes.
This project analyzes breast cancer patient data to predict survival, treatment response, and infection risk using gene expression and clinical datasets (GSE datasets). The focus is on deriving actionable insights from complex biomedical data to support clinical decision-making.</p>
<p>In this project, we used gene expression and clinical datasets (GSE25066 & GSE19783 from NCBI GEO) to predict:
 
- Survival outcomes
  
- Treatment response
  
- Infection risk
We implemented Machine Learning (Random Forest) and Deep Learning (Neural Networks) models, applied SHAP explainability to identify genetic key features most associated with patient outcomes and deployed an interactive Gradio interface for real-time predictions. </P>


---

## 🔹 Objectives
- Predict survival, treatment response, and infection risk from high-dimensional gene expression data.  
- Compare the performance of **Random Forest** and **Neural Network** models.  
- Use SHAP values to identify important genetic features.  
- Deploy an interactive Gradio app for real-time predictions.

---

### 🛠️ Technical Design

<p>Languages: Python</p>
<p>Data Handling: pandas, numpy, GEOparse</p>
<p>ML/DL Models: scikit-learn (Random Forest), TensorFlow/Keras (Neural Networks)</p>
<p>Explainability: SHAP</p>
<p>Visualization: matplotlib, seaborn</p>
<p>Interface: Gradio</p>
<p>Version Control: GitHub</p>


---

## 🔹 Approach
## 1.Data Source 
The analysis depends on two distinct datasets.
* **GSE25066**: Which has gene expression profiles for breast cancer patients and is employed to predict survival outcomes and treatment responses.
* **GSE19783**:* Which provides gene expression data prominent to infection risk.
* Datasets were downloaded from the NCBI GEO repository in compressed format and converted to CSV for analysis.

## 2. Data Preprocessing
**Steps performed:**  

* **Data Retrieval:**  
  Downloaded and extracted datasets using GEOparse, then converted them to structured CSV files.  
  ![Data Retrieval](https://github.com/user-attachments/assets/e5f95cdc-b1db-4b33-936a-7ec09da47f84)  

* **Data Cleaning:**  
  Selected only relevant gene expression columns (GSM) and added a `Probe_Category` column.  
  Proxy labels for **Survival_Outcome, Treatment_Outcome, and Infection_Outcome** were created using KMeans clustering.  

* **Normalization & Splitting:**  
  Standardized the gene expression data using `StandardScaler` (mean=0, std=1). Split datasets 80% for training and 20% for testing.  
  ![Normalization](https://github.com/user-attachments/assets/7d542438-a666-4f1f-9571-7fdd48bd3e80)  

* **Challenges:**  
  - Proxy outcomes may not fully match clinical labels, introducing noise.  
  - Infection dataset had fewer features (115 vs 508), limiting model predictions.  

---

## 3. Exploratory Data Analysis (EDA)

**i. Mean Survival and Infection Outcome**  
Non-survivors had higher mean gene expression (~9.5) vs survivors (~6.0). High infection risk patients also showed higher expression (~11 vs 7).  
![Mean Survival & Infection](https://github.com/user-attachments/assets/6e0cef86-1776-460d-9dc8-c693e09e623c)  

**ii. Mean Expression Across GSM Columns**  
Distinct expression patterns observed between survivors and non-survivors.  
![Mean Expression GSM](https://github.com/user-attachments/assets/feab09f4-47e0-47bd-88ce-0eb062bfc90a)  

**iii. GSM Expression by Infection Outcome**  
Clear clustering of low-risk (blue) and high-risk (red) patients.  
![GSM Infection Outcome](https://github.com/user-attachments/assets/99caff90-7f3c-42e2-ac64-ae842c924d29)  

**iv. Density of Survival & Infection**  
Dataset-specific expression distributions highlighted differences between survival and infection datasets.  
![Density Survival & Infection](https://github.com/user-attachments/assets/18f45d88-48c5-4e79-aec1-7daf9b1c10ea)  

**v. Correlation Heatmap of GSM Features**  
High correlation among first 5 GSM features, indicating redundancy.  
![Correlation Heatmap](https://github.com/user-attachments/assets/dda555d6-7c8b-44c3-b4b8-0c2c6a23a191)  

**vi. GSM615096 vs GSM615097 Scatter Plot**  
Shows strong predictive potential for survival outcomes.  
![GSM Scatter Plot](https://github.com/user-attachments/assets/d1718bc1-af43-4146-912e-8e6da6d61365)  

---

## 4. Model Development & Evaluation

**Random Forest (RF) Models**  
- Trained for survival, treatment, and infection outcomes.  
- Hyperparameters: `n_estimators=50`, `max_depth=10`.  
- Saved models: `rf_survival_model.joblib`, `rf_treatment_model.joblib`, `rf_infection_model.joblib`.  

**Neural Network (NN) Models**  
- Architecture: Dense(64) → Dropout(0.2) → Dense(32) → Dense(1, sigmoid)  
- Optimizer: Adam, Loss: Binary Crossentropy  
- Epochs: 20, Validation split: 20%  
- Saved models: `nn_survival_model.h5`, `nn_treatment_model.h5`, `nn_infection_model.h5`.  

**Performance Highlights**  

- **Random Forest:**  
  - Survival: 98.41% accuracy, AUC 0.9991  
  - Treatment: 98.07% accuracy, AUC 0.9988  
  - Infection: 99.39% accuracy, AUC 0.9998  
![RF Evaluation](https://github.com/user-attachments/assets/555beed8-5240-4d7c-aff6-ac87a73f6416)  

 **Neural Network Loss & NN Accuracy Curves**  
![Survival NN](https://github.com/user-attachments/assets/a72f8988-b1cd-4892-b691-765945cab2fe)  
![Treatment NN](https://github.com/user-attachments/assets/8c9a1237-e9e5-42cd-985c-491d42a53eaa)  
![Infection NN](https://github.com/user-attachments/assets/964eb237-f4ad-45eb-af60-9d0a366bbda1)  

**NN Performance Comparison**  
![NN Comparison](https://github.com/user-attachments/assets/8639c715-a781-4319-b7d6-7dc5b6775722)  

**ROC Curves (Random Forest)**  
![RF ROC Curves](https://github.com/user-attachments/assets/b6317439-9823-46d7-bd56-81f5d49c8b65)  

**Model Comparison**  
![RF vs NN Comparison](https://github.com/user-attachments/assets/db5e0f0f-cff1-4c42-b11b-6512dbe777de)  

**SHAP Feature Importance (Survival Outcome)**  
![SHAP Summary](https://github.com/user-attachments/assets/5ec93798-c73e-4bff-8627-c289ba63e5e9)  

---

## 5. Gradio Deployment
Interactive interface to predict outcomes using five gene inputs (GSM615096–GSM615100).  
![Gradio Interface](https://github.com/user-attachments/assets/1100bc45-1fa5-417a-b5c0-a17e5a49797f)  

---

## 6. Key Findings
- RF models outperformed NN, with infection risk achieving **99.39% accuracy**.  
- SHAP analysis confirmed critical gene features for survival predictions.  
- EDA validated that survivors have higher gene expression patterns.  
- Gradio interface provides real-time predictions but is limited to five features.  

---

## 7. Recommendations / Future Work
- Use real clinical labels instead of proxy outcomes.  
- Expand infection dataset for better predictions.  
- Optimize NN architecture and reduce overfitting.  
- Extend Gradio to accept full feature set.  
- Validate models on external datasets.  

---

## 8. Conclusion
Random Forest and Neural Network models successfully predicted breast cancer outcomes, with RF models showing **superior performance**. This project highlights how **AI and explainable ML** can assist clinicians in personalized treatment planning.  


---

## 🔹 Results
- Random Forest consistently outperformed Neural Networks.  
- Infection risk model achieved **99.39% accuracy with near-perfect AUC**.  
- SHAP results aligned with clinical insights, making predictions interpretable.  
- The Gradio app provides real-time predictions, demonstrating potential for clinical decision support.

---

## 🔹 My Contribution (Meghana Bodduluri)
- Retrieved and preprocessed GEO datasets into structured formats.  
- Conducted **EDA**, visualizations, and correlation analysis(survival trends, heatmaps, scatter plots).
- Implemented and tuned **Random Forest models**.  
- Applied **SHAP explainability** to identify predictive features.  
- Assisted with **Gradio deployment** for interactive model use.

---


























 



   
     
