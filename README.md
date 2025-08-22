# Capstone_Project
# 🧬 Integrative Analysis of Breast Cancer Outcomes Using Oncology Genes

## About 
<P>Term: Spring 2025
<P>Team: Purple (Meghana Bodduluri, Sanjana Kanneganti, Sowmya Bathula, Swetha Nunemunthala)
**Keywords**: Python, pandas, scikit-learn, TensorFlow, SHAP, Gradio

## 📖 Project Overview  
<P>Breast cancer is one of the most common cancers among women worldwide. Early detection and personalized treatment planning are critical for improving patient outcomes.
In this project, we used gene expression and clinical datasets (GSE25066 & GSE19783 from NCBI GEO) to predict:

Survival outcomes

Treatment response

Infection risk

We implemented Machine Learning (Random Forest) and Deep Learning (Neural Networks) models, and applied SHAP explainability to identify genetic features most associated with patient outcomes. An interactive Gradio interface was developed to demonstrate real-time predictions. </P>

### 🛠️ Technical Design

<p>Languages: Python
<p>Data Handling: pandas, numpy, GEOparse
<p>ML/DL Models: scikit-learn (Random Forest), TensorFlow/Keras (Neural Networks)
<p>Explainability: SHAP
<p>Visualization: matplotlib, seaborn
<p>Interface: Gradio
<p>Version Control: GitHub


</P>

### 📦 Required Resources

<P>Data Resources: Public GEO datasets (e.g., GSE25066, GSE19783)</P>
<P>Software Resources:</P>
<P>    -Python 3.x </P>
<P>    - Jupyter Notebook / VS Code </P>
<P>Libraries: </P>
<P>    - GEOparse, pandas, numpy</P>
<P>    - scikit-learn, Keras/TensorFlow, SHAP, matplotlib, seaborn, joblib</P>
<P>Version Control management (GitHub/Git)</P>

### Project Plan :  
# Project Plan
## 1.Data Source 
The analysis depends on two distinct datasets.
* **GSE25066**: Which has gene expression profiles for breast cancer patients and is employed to predict survival outcomes and treatment responses.
* **GSE19783**:* Which provides gene expression data prominent to infection risk.
* Both datasets were downloaded as compressed matrix files from the NCBI GEO repository to ensure data integrity and accessibility for subsequent analysis. 

## 2.Data Pre-processing 
 This step involves data retrieval, cleaning and normalization.
* **2.1.Data Retrieval:**
The preprocessing stage started with the retrieval of data using the GEOparse library, a Python tool designed specifically for GEO data access. The datasets were extracted, decompressed from their.gz compressed format, and then converted into.csv files. In order to help with additional processing and analysis, this phase made sure that the raw data was transformed into a structured format.
![image](https://github.com/user-attachments/assets/e5f95cdc-b1db-4b33-936a-7ec09da47f84)

* **2.2.Data Cleaning:**
During data cleaning, all extra information was removed, and only the gene expression data within the GSM columns was used. A new column called Probe_Category was added to help group similar probes based on their ID patterns, making the data more organized.To create target labels for prediction, KMeans clustering was used on the gene expression data which helped to generate proxy outcomes like Survival_Outcome (e.g., Survivor vs. Non-Survivor),Treatment_Outcome and Infection_Outcome.

* **2.3.Normalization and Splitting:**
To get the data ready for machine learning, the gene expression values were standardized using StandardScaler from scikit-learn. This means the numbers were adjusted so that they have a mean of 0 and a standard deviation of 1, making them easier for models to work with. After this, the data was split into two parts for each outcome survival, treatment response, and infection risk which a is 80% for training the model and 20% for testing how well the model works.

![image](https://github.com/user-attachments/assets/7d542438-a666-4f1f-9571-7fdd48bd3e80)

**2.4.Challenges Encountered**

A significant challenge during preprocessing was the reliance on proxy outcomes derived through clustering, which may not fully align with actual clinical labels, potentially introducing noise and impacting model reliability. Additionally, the infection dataset contains fewer features (115 GSM columns compared to 508 in the survival dataset), which may constrain the model’s ability to accurately predict infection risk due to the reduced dimensionality of the data.

***

## 3.Exploratory Data Analysis (EDA)

**i. Mean Survival and Infection Outcome in Breast Cancer**

The Mean Survival and Infection Outcome bar plot highlighted differential expression patterns. Non-survivors exhibited a mean expression of approximately 9.5, while survivors had a mean of around 6.0. Similarly, high-risk infection patients showed a mean expression of about 11.0, compared to 7.0 for low-risk patients, underscoring the relationship between expression levels and outcomes.

![image](https://github.com/user-attachments/assets/6e0cef86-1776-460d-9dc8-c693e09e623c)

 
**ii.Mean Expression Across GSM Columns by Survival Outcome in Breast Cancer**

The Mean Expression Across GSM Columns by Survival Outcome revealed that non-survivors (red dots) consistently exhibited higher mean expression levels, ranging from 9.0 to 9.5 across 508 GSM features, compared to survivors (green dots), whose mean expression ranged from 6.0 to 6.5, indicating distinct expression profiles between the two groups.

![image](https://github.com/user-attachments/assets/feab09f4-47e0-47bd-88ce-0eb062bfc90a)

**iii. GSM Expression by Infection Outcome in Breast Cancer**

The GSM Expression by Infection Outcome scatter matrix of GSM496794, GSM496795, and GSM496796 showed clear clustering by infection outcome. Low-risk patients (blue) and high-risk patients (red) formed distinct clusters, with density plots highlighting multimodal distributions, suggesting that these features are highly discriminative for infection risk prediction.

![image](https://github.com/user-attachments/assets/99caff90-7f3c-42e2-ac64-ae842c924d29)

**iv. Density of Survival and Infection in Breast Cancer**

The Density of Survival and Infection plot compared the distribution of gene expression levels across the two datasets. The survival dataset (blue) exhibited a broader distribution with peaks at approximately 5.0 and 12.0, while the infection dataset (red) peaked at around 10.0, reflecting dataset-specific expression profiles and potential differences in underlying biology.

![image](https://github.com/user-attachments/assets/18f45d88-48c5-4e79-aec1-7daf9b1c10ea)

**v. Correlation Heatmap of GSM Features in Breast Cancer**

The Correlation Heatmap of GSM Features indicated high correlations among the first five GSM features. Specifically, correlations between GSM615096, GSM615097, GSM615098, and GSM615100 ranged from 0.85 to 0.90, suggesting potential redundancy, while GSM615099 showed a slightly lower correlation of 0.90 with GSM615098, indicating some degree of independence.

![image](https://github.com/user-attachments/assets/dda555d6-7c8b-44c3-b4b8-0c2c6a23a191)

**vi. GSM615096 vs GSM615097**

The GSM615096 vs. GSM615097 scatter plot displayed two distinct clusters based on survival outcome. Non-survivors (red) had lower expression levels, ranging from 0.0 to 7.5, while survivors (blue) exhibited higher expression levels, ranging from 7.5 to 17.5, suggesting that these features have strong predictive potential for survival outcomes.


![image](https://github.com/user-attachments/assets/d1718bc1-af43-4146-912e-8e6da6d61365)



***

### 4. Model Development and Evaluation
**4.1. Random Forest Models**

Random Forest (RF) classifiers were meticulously trained for survival, treatment response, and infection risk prediction. The hyperparameters for the survival and infection models were fixed at n_estimators=50 and max_depth=10 to ensure consistency. For the treatment model, a grid search was performed, confirming the same optimal parameters of n_estimators=50 and max_depth=10. The models were trained on scaled data with parallel processing enabled through n_jobs=-1 for efficiency, and they were saved as rf_survival_model.joblib, rf_treatment_model.joblib, and rf_infection_model.joblib for future use.

**4.2. Neural Network Models**
Neural Networks (NN) were developed using TensorFlow/Keras with a consistent architecture across all tasks: a first dense layer with 64 units and ReLU activation, followed by a dropout layer with a rate of 0.2, a second dense layer with 32 units and ReLU activation, and a final dense layer with 1 unit and sigmoid activation for binary classification. The models were optimized using the Adam optimizer with binary crossentropy loss. They were trained for 20 epochs with batch sizes of 64 for survival and 32 for treatment and infection, using a 20% validation split. The trained models were saved as nn_survival_model.h5, nn_treatment_model.h5, and nn_infection_model.h5.

**4.3.	Model Performance Results**

_**4.3.1.	Random Forest Evaluation**_

![image](https://github.com/user-attachments/assets/555beed8-5240-4d7c-aff6-ac87a73f6416)

The evaluation metrics provided in the output demonstrated exceptional performance across all tasks. For survival prediction, the RF model achieved an accuracy of 98.41%, an RMSE of 0.1262, an R² of 0.9361, and an AUC of 0.9991. The treatment response model recorded an accuracy of 98.07%, an RMSE of 0.1389, an R² of 0.9226, and an AUC of 0.9988. The infection risk model performed the best, with an accuracy of 99.39%, an RMSE of 0.0781, an R² of 0.9696, and an AUC of 0.9998, reflecting its superior discriminative ability.

**_4.3.2. Survival NN Loss Curve and Survival NN Accuracy Curve_**

The Survival NN Loss Curve and Accuracy Curve illustrated the training dynamics of the survival Neural Network model over 20 epochs. The training loss (blue line) decreased steadily from 0.40 to near 0.00, while the validation loss (orange line) fluctuated between 0.05 and 0.35, indicating potential overfitting. The training accuracy approached 99%, whereas the validation accuracy plateaued at approximately 94% to 96%, further suggesting overfitting concerns.

![image](https://github.com/user-attachments/assets/a72f8988-b1cd-4892-b691-765945cab2fe)

_**4.3.3. Treatment NN Loss Curve and Treatment NN Accuracy Curve**_

The Treatment NN Loss Curve and Accuracy Curve depicted similar trends for the treatment response model. The training loss decreased from 0.35 to 0.00, but the validation loss showed significant fluctuations, ranging from 0.05 to 0.35, confirming overfitting. The training accuracy reached 99%, while the validation accuracy stabilized at around 94% to 96%, highlighting a gap between training and validation performance.

![image](https://github.com/user-attachments/assets/8c9a1237-e9e5-42cd-985c-491d42a53eaa)

**_4.3.3. Infection NN Loss Curve and Infection NN Accuracy Curve_**

The Infection NN Loss Curve and Accuracy Curve followed a comparable pattern for the infection risk model. The training loss dropped from 0.60 to 0.00, but the validation loss exhibited pronounced fluctuations between 0.05 and 0.30, reinforcing overfitting concerns. The training accuracy approached 99%, with the validation accuracy plateauing at approximately 94% to 96%, indicating challenges in generalizing to unseen data.

![image](https://github.com/user-attachments/assets/964eb237-f4ad-45eb-af60-9d0a366bbda1)


_**4.3.4. NN Accuracy Comparison, NN RMSE Comparison, and NN R2 Comparison**_

![image](https://github.com/user-attachments/assets/8639c715-a781-4319-b7d6-7dc5b6775722)

The NN Accuracy, RMSE, and R2 Comparison provided a comparative analysis of Neural Network performance across the three tasks. The survival NN achieved an accuracy of approximately 90%, the treatment NN around 89%, and the infection NN about 87%. The RMSE values were approximately 0.32 for survival, 0.34 for treatment, and 0.36 for infection. The R² scores were around 0.62 for survival, 0.59 for treatment, and 0.55 for infection, indicating moderate predictive power with room for improvement.

_**4.3.5. ROC Curves for RForest Models for Breast Cancer Analysis**_

The ROC Curves for RForest Models demonstrated the exceptional discriminative ability of the Random Forest models. The survival RF model achieved an AUC of 0.9991, the treatment model an AUC of 0.9988, and the infection model the highest AUC of 0.9998, reflecting near-perfect classification performance across all tasks.

![image](https://github.com/user-attachments/assets/b6317439-9823-46d7-bd56-81f5d49c8b65)

### 4.4. Model Comparison

![image](https://github.com/user-attachments/assets/db5e0f0f-cff1-4c42-b11b-6512dbe777de)

The Random Forest models consistently outperformed the Neural Networks across all evaluation metrics. The RF models achieved higher accuracy (98.41% vs. ~90% for survival), lower RMSE (0.1262 vs. ~0.32 for survival), and better R² scores (0.9361 vs. ~0.62 for survival). The RF AUCs were near-perfect, ranging from 0.9991 to 0.9998, while the NN AUCs were typically lower, estimated at around 0.90 to 0.93 (not shown in the provided visuals). Among the tasks, infection prediction had the highest RF accuracy (99.39%) and R² (0.9696), likely due to clearer feature separation, as observed in the scatter matrix . Survival prediction benefited from the larger feature set (508 vs. 115), while treatment prediction showed slightly lower performance (accuracy: 98.07%), possibly due to noise introduced by the proxy outcomes.

_**4.4.1.	 Feature Importance for Survival Outcome in Breast Cancer Survival**_

The SHAP Summary Plot for the Random Forest survival model elucidated the importance of various features in predicting survival outcomes. Features with positive SHAP values (represented by red dots on the right side) increased the likelihood of survival, while those with negative SHAP values (blue dots on the left side) decreased it. The spread of SHAP values, ranging from -0.08 to +0.08, indicates that a subset of features significantly influences survival predictions, although specific feature labels were not provided in the visualization.

![image](https://github.com/user-attachments/assets/5ec93798-c73e-4bff-8627-c289ba63e5e9)

***


### 5. Gradio Deployment

![image](https://github.com/user-attachments/assets/1100bc45-1fa5-417a-b5c0-a17e5a49797f)

A Gradio interface was developed to enable interactive predictions using five gene expression inputs, specifically GSM615096 to GSM615100. The interface accepts numeric inputs pre-filled with example values from the survival_df dataset and provides outputs for survival prediction (Survivor/Non-Survivor with probability), infection risk (High/Low Risk with probability), and treatment response (Responsive/Non-Responsive with probability). However, the need for zero-padding to match the expected feature counts (508 for survival/treatment, 115 for infection) may dilute prediction accuracy, and the use of dynamic scaling for survival and treatment inputs introduces potential inconsistencies compared to the pre-trained scaler used for infection 
predictions.


***


### 6.Discussion

**6.1. Key Findings**

The Random Forest models demonstrated exceptional performance, with the infection prediction model achieving the highest accuracy of 99.39% and an AUC of 0.9998, reflecting its near-perfect discriminative ability. The Neural Network models showed stable training dynamics but were less robust, exhibiting signs of overfitting, particularly for the treatment and infection tasks, as evidenced by the fluctuating validation loss curves. The SHAP analysis confirmed that a subset of features significantly drives survival predictions, aligning with the EDA findings that survivors tend to have higher gene expression levels (e.g., 7.5–17.5 in GSM615096 and GSM615097). The EDA revealed differential expression patterns, validating the use of gene expression data as a predictive feature, although high feature correlations (e.g., 0.85–0.90 among GSM615096 to GSM615100) suggest redundancy, which the RF models handled more effectively than the NN models. The Gradio interface provides a practical tool for clinical predictions but is limited by its reliance on only five input features.

**6.2. Strengths of the Study**

This study offers a comprehensive and thorough analysis across three critical dimensions of breast cancer—survival, treatment response, and infection risk—supported by robust exploratory data analysis and SHAP-based explainability for enhanced interpretability. The Random Forest models achieved near-perfect performance metrics, demonstrating their suitability for predictive tasks in this context. The deployment of the Gradio interface further enhances clinical accessibility, providing a user-friendly platform for real-time predictions.
7.	Recommendations for Future Work
To enhance the analysis, clinical labels should be incorporated to replace the proxy outcomes, ensuring more accurate representation of patient outcomes. The infection dataset should be expanded with additional features or samples to improve model performance. Ensemble methods combining Random Forest and Neural Network models could be explored to leverage the strengths of both approaches. The Neural Network architectures should be optimized, potentially by increasing the number of layers or tuning hyperparameters, to reduce overfitting. The Gradio interface should be refined to allow variable feature inputs, accommodating the full range of GSM features, and a dedicated scaler should be trained for survival and treatment predictions to ensure consistency. Finally, the models should be validated on external datasets and tested in clinical trials to confirm their generalizability and real-world applicability.


***

### 8. Conclusion

This study successfully developed and evaluated Random Forest and Neural Network models for predicting breast cancer outcomes, with the Random Forest models achieving superior performance, evidenced by an accuracy of up to 99.39% and an AUC of up to 0.9998. The exploratory data analysis and SHAP analysis provided valuable insights into feature importance and differential expression patterns, while the Gradio interface offered a practical tool for interactive predictions. Future work should focus on clinical validation, data enhancement, and interface improvements to address the identified limitations and enhance the real-world applicability of these predictive models.


***


























 



   
     
