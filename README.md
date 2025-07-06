# 📈 Intelligent Investment Advisor

**A Stacked Machine Learning-Based Investment Sector Recommendation System with Explainable Insights**

This project proposes an explainable AI framework that leverages Machine Learning (ML), Deep Learning (DL), and Explainable AI (XAI) to recommend optimal investment sectors based on investor profiles. The system provides personalized investment recommendations and builds user trust using transparent explanations powered by LIME.

> 📌 Dataset: [Investment Survey Dataset – Kaggle](https://www.kaggle.com/datasets/tmmhratul/investment-survey-dataset)

---

## 🔍 Project Highlights

- ✅ Predicts suitable **investment sectors** like Stocks, Real Estate, Mutual Funds, etc., based on financial goals and risk profile.
- 🤖 Combines ML models: Random Forest, SVM, KNN, Logistic Regression, Naive Bayes, Decision Tree, and Gradient Boosting.
- 🧠 Stacked and Voting Ensemble Classifiers for increased performance (up to **94.12%** accuracy).
- 💬 Transparent decision-making using **LIME** and **Feature Importance**.
- 🧪 Custom dataset gathered through structured surveys (2,000+ responses).
- 🌐 Optional **Flask-based frontend** for user interaction.

---

## 📁 Repository Structure

nvestment_Sector_Recommendation/
│
├── data/ # Dataset & preprocessing scripts
├── models/ # Training scripts & saved model files
├── explainer/ # LIME-based interpretation scripts
├── app/ # Flask frontend (optional)
├── requirements.txt # Python dependencies
├── README.md # Project documentation


---

## ⚙️ Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/TMRatul49/Investment_Sector_Recommendation.git
cd Investment_Sector_Recommendation


### 2. Create Virtual Environment & Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate        # For Windows: venv\Scripts\activate
pip install -r requirements.txt


### 3. Run the Web Interface

```bash
cd app
flask run

Then open http://localhost:5000 to interact with the recommendation system via a web interface.

## 📊 Performance Summary

| Model               | Accuracy (%) | Highlights                                 |
|--------------------|--------------|--------------------------------------------|
| Logistic Regression| 88.24        | Consistent predictions                     |
| SVM (Tuned)        | 94.24        | Best accuracy across multiple sectors      |
| Random Forest      | 82.35        | Strong in Marketing & Mutual Funds         |
| Deep Learning      | 92.86        | Excellent generalization                   |
| Voting Classifier  | 92.86        | Robust majority voting                     |
| Stacking Classifier| 94.12        | Best overall performance across all sectors|


## 📄 Dataset

- **Source**: Custom-structured survey  
- **Hosted on**: [Kaggle Dataset](https://www.kaggle.com/datasets/tmmhratul/investment-survey-dataset)  
- **Size**: 2,000+ entries  

### 🔑 Features

- Age, Gender, Annual Income, Working Status  
- Monthly Investment Amount, Duration  
- Investment Goal, Motivation, Resource Used  
- Mode of Investment (Categorical + Numeric)


## 🧠 ML Models Used

- Logistic Regression  
- Naive Bayes  
- Support Vector Machine  
- Random Forest  
- K-Nearest Neighbors  
- Decision Tree  
- Gradient Boosting  
- Deep Learning Neural Network  
- **Ensemble Methods**: Voting & Stacking Classifiers  

---

## 🧪 Explainable AI

We use **LIME** and **Feature Importance Analysis** to:

- 🧩 Provide local instance-level explanations  
- 📊 Visualize top influential features per prediction  
- 🔍 Ensure transparency and trustworthiness  

---

## 📌 Key Contributions

- 📊 Accurate predictions with stacked ensemble learning  
- 🔍 Interpretable insights using LIME  
- 🤝 User-friendly system for both novice and expert investors  
- 🧪 Ethically sourced dataset reflecting real investor behavior  
- 💼 Decision support for financial institutions, policy makers, and individuals  

---

## 🛠️ Future Work

- 🌐 Integrate SHAP for global explainability  
- 📡 Add real-time financial data streams  
- 📱 Deploy as full-stack or mobile app  
- ⏱️ Optimize runtime for low-resource devices  


## 📄 License

This project is licensed under the **MIT License**.  
See the [LICENSE](LICENSE) file for details.

