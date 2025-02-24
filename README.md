# Credit-Card-Fraud-Detection
# 💳 Credit Card Fraud Detection  

## 📌 Project Overview  
Credit card fraud is a significant global problem, leading to financial losses and security concerns. This project applies **machine learning models** to detect fraudulent transactions based on real-world credit card transaction data. By analyzing patterns in transaction features, we aim to build a model that accurately identifies fraud cases.  

## 📁 Repository Contents  
- **`Credit-Card-Fraud-Detection.ipynb`** - Jupyter Notebook with data preprocessing, model training, and evaluation.  
- **`card_transdata.csv`** - Processed dataset used for training and testing. It's bigger than 25MB, so we can just use data from the link below. 
- **`Credit-Card-Fraud-Detection.pptx`** - Presentation summarizing methodology, findings, and conclusions.  

## 📊 Dataset Information  
- **Source:** Kaggle Dataset by [Dhanush Narayanan R](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud)  
- The dataset contains transaction details labeled as fraudulent or non-fraudulent.  
- A **random sample of 5,000 records** was extracted to enhance computational efficiency while maintaining data distribution.  

## 🔍 Data Preprocessing  
1. **Feature Selection**: Identifying relevant transaction attributes.  
2. **Data Splitting**: 70% of the data is used for training, 30% for testing.  
3. **Feature Scaling**: Standardization with `StandardScaler` to normalize numerical features.  

## 🏆 Machine Learning Models Used  
We experimented with multiple classification models:  
- **Logistic Regression**  
- **K-Nearest Neighbors (KNN)**  
- **Decision Tree**  
- **Random Forest**  
- **Linear Discriminant Analysis (LDA)**  
- **Naïve Bayes**  
- **Support Vector Machine (SVM)** (Linear, Gaussian, Polynomial)  

## 📈 Model Evaluation Metrics  
- **True Positive Rate (TPR)**  
- **True Negative Rate (TNR)**  
- **Accuracy Score**  

### **🚀 Key Findings**
- **Random Forest** performed the best, achieving the highest **TPR, TNR, and accuracy**.  
- **LDA (Linear Discriminant Analysis)** had the lowest detection performance.  
- Feature scaling and hyperparameter tuning improved classification results.  

## 🔧 How to Run the Project  
1. **Install required Python libraries**  
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
2.Open Jupyter Notebook and run CS677_Final_Project.ipynb.
3.The dataset card_transdata.csv should be in the same directory as the notebook.

## 🔗 References
Dataset: Kaggle - Credit Card Fraud
Libraries Used: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
## 👤 Author
Zhenghao An

## ⭐ Contributions & Feedback
If you find this project useful, star ⭐ this repository and provide feedback!
## Disclaimer
This project was developed as part of my coursework at Boston University (BU MET Program). The contents of this repository represent my own work and do not include any proprietary course materials, data, or solutions provided by BU. If you are a current student, please adhere to BU’s academic integrity policies.
