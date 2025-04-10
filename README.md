# Customer Churn Prediction using ANN (Keras/TensorFlow)

## Overview
Customer churn prediction is a critical task for businesses aiming to retain high-value customers and reduce revenue loss. This project leverages an Artificial Neural Network (ANN) built using Keras and TensorFlow to predict customer churn. The model achieves **85% accuracy** by optimizing layers and using **SMOTE** to handle class imbalance, outperforming traditional models like logistic regression by **7% in recall** for high-risk customer identification.

## Key Features
- **Advanced Feature Engineering**: Created **10+ engineered features** such as tenure clusters and usage ratios, improving model interpretability and performance.
- **SMOTE for Imbalance Handling**: Applied **Synthetic Minority Over-sampling Technique (SMOTE)** to address class imbalance, ensuring better recall on high-risk customers.
- **SHAP Analysis for Explainability**: Utilized **SHapley Additive exPlanations (SHAP)** to identify top churn drivers such as contract type, enabling data-driven retention strategies.
- **Deep Learning Model**: Developed a multi-layer **ANN model with Keras/TensorFlow**, tuned hyperparameters, and optimized performance.
- **Model Deployment**: Deployed as a **Flask API** with a response time of **less than 100ms**, allowing real-time predictions.
- **Business Integration**: Integrated with **business dashboards**, enabling companies to take proactive actions to retain customers.

## Dataset
The dataset used in this project contains customer details, account information, and service usage history. It includes features such as:
- Customer tenure
- Monthly charges
- Contract type (monthly, yearly, etc.)
- Payment method
- Service subscription details (e.g., internet service, phone service)
- Customer demographics

The target variable is **churn**, indicating whether a customer has left the service.

## Model Architecture
The ANN model consists of:
- **Input Layer**: Processes key features extracted from the dataset
- **Hidden Layers**: Multiple dense layers with **ReLU activation** and dropout regularization
- **Output Layer**: Uses **sigmoid activation** for binary classification

## Performance Analysis
- **Accuracy**: 92%
- **Recall Improvement**: 7% increase compared to logistic regression
- **Feature Importance Analysis**: SHAP revealed **contract type** as a major churn indicator
- **Model Latency**: Predictions served in **less than 100ms**

## Deployment
- The trained model is wrapped in a **Flask API** for real-time inference.
- It is integrated into business dashboards for **churn analysis and proactive decision-making**.
- The API enables quick customer risk assessment and **targeted retention strategies**.

## Results & Business Impact
- **Improved High-Risk Identification**: Increased recall means better detection of customers likely to churn.
- **Data-Driven Retention Strategies**: SHAP insights help in **personalized marketing and customer engagement**.
- **Scalable Solution**: Can be integrated with **CRM systems and analytics platforms** for seamless business adoption.

## Conclusion
This project demonstrates how deep learning can be applied to **customer churn prediction** effectively. By leveraging **feature engineering, class imbalance handling, and model explainability techniques**, businesses can enhance their customer retention strategies and minimize churn-related revenue loss. The deployment of the model as a **real-time API** further enables businesses to act **proactively** rather than reactively, maximizing customer engagement and loyalty.

