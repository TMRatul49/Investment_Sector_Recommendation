import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

app = Flask(__name__)

# Load and preprocess the dataset
df = pd.read_csv("investment_survey.csv")
categorical_columns = ['Gender', 'Motivation_cause', 'Resources_used', 'Goal_for_investment']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

mode_of_investment_encoder = LabelEncoder()
df['Mode_of_investment_numeric'] = mode_of_investment_encoder.fit_transform(df['Mode_of_investment'])
X = df.drop(columns=["Mode_of_investment", "Mode_of_investment_numeric"])
y = df["Mode_of_investment_numeric"]

# Remove classes with only one sample
class_counts = y.value_counts()
classes_with_one_sample = class_counts[class_counts < 2].index.tolist()
if classes_with_one_sample:
    mask = ~y.isin(classes_with_one_sample)
    X = X[mask]
    y = y[mask]

# Normalize features
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

# Balance classes
smote = SMOTE(random_state=42, k_neighbors=3)
X_balanced, y_balanced = smote.fit_resample(X_normalized, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.1, random_state=42, stratify=y_balanced)

# Train stacking classifier
base_learners = [
    ('lr', LogisticRegression()),
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier()),
    ('nb', GaussianNB()),
    ('rf', RandomForestClassifier(random_state=42)),
    ('svm', SVC(probability=True)),
    ('gb', GradientBoostingClassifier())
]

stacking_clf_rf = StackingClassifier(estimators=base_learners, final_estimator=RandomForestClassifier(random_state=42))
stacking_clf_rf.fit(X_train, y_train)

# Lime explainer
explainer = LimeTabularExplainer(X_train, mode='classification', training_labels=y_train, feature_names=X.columns)

def transform_user_input(user_input):
    input_df = pd.DataFrame([user_input], columns=['Gender', 'Age', 'Working_professional', 'Annual_income', 
                                                   'Investment_per_month', 'Goal_for_investment', 
                                                   'Duration_to_save(in_Years)'])
    input_encoded = pd.get_dummies(input_df, columns=['Gender', 'Goal_for_investment'], drop_first=True)
    missing_cols = set(X.columns) - set(input_encoded.columns)
    for col in missing_cols:
        input_encoded[col] = 0
    input_encoded = input_encoded[X.columns]
    return input_encoded

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def predict():
    try:
        # Get input from form
        gender = int(request.form['gender'])
        age = int(request.form['age'])
        working_professional = int(request.form['working_professional'])
        annual_income = float(request.form['annual_income'])
        investment_per_month = float(request.form['investment_per_month'])
        goal_for_investment = int(request.form['goal_for_investment'])
        duration = int(request.form['duration'])

        # Prepare user input
        user_input = [gender, age, working_professional, annual_income, investment_per_month, goal_for_investment, duration]
        user_input_encoded = transform_user_input(user_input)
        user_input_scaled = scaler.transform(user_input_encoded)

        # Prediction
        prediction = stacking_clf_rf.predict(user_input_scaled)
        predicted_mode = mode_of_investment_encoder.inverse_transform(prediction)[0]

        # Explain prediction
        explanation = explainer.explain_instance(user_input_scaled[0], stacking_clf_rf.predict_proba, num_features=10)
        explanation_html = explanation.as_html()

        return render_template('result.html', prediction=predicted_mode, explanation=explanation_html)
    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
