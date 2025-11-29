
# Logistic Regression Model for Placement Prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
import joblib

# 1. Load Dataset

df = pd.read_csv("placementdata.csv")
print("Dataset loaded successfully! Shape:", df.shape)
print(df.head())

# 2. Data Cleaning & Preparation

if "StudentID" in df.columns:
    df = df.drop(columns=["StudentID"])

if "PlacementStatus" in df.columns:
    df.rename(columns={"PlacementStatus": "Placed"}, inplace=True)

df["Placed"] = df["Placed"].map({"Placed": 1, "NotPlaced": 0})

cat_cols = ["ExtracurricularActivities", "PlacementTraining"]
num_cols = [c for c in df.columns if c not in cat_cols + ["Placed"]]

print("\nCategorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# 3. Basic Visualization
plt.figure(figsize=(5, 3))
sns.countplot(x="Placed", data=df)
plt.title("Placement Distribution")
plt.show()

sns.heatmap(df[num_cols + ['Placed']].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# 4. Split Data

X = df.drop(columns=["Placed"])
y = df["Placed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n Train size: {X_train.shape}, Test size: {X_test.shape}")


# 5. Preprocessing & Model Pipeline

numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])


# 6. Train the Model

model.fit(X_train, y_train)
print("\n Model trained successfully!")

# 7. Evaluation

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Not Placed", "Placed"],
            yticklabels=["Not Placed", "Placed"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# 8. Save Model

joblib.dump(model, "placement_model.joblib")
print("\n Model saved as placement_model.joblib")

# 9. Predict New Data from CSV

try:
    students = pd.read_csv("students.csv")
    print("\n New data loaded from students.csv")

    if "Name" in students.columns:
        X_new = students.drop("Name", axis=1)
    else:
        X_new = students

    loaded_model = joblib.load("placement_model.joblib")
    pred = loaded_model.predict(X_new)
    prob = loaded_model.predict_proba(X_new)[:, 1]

    students["Predicted_Placement"] = pred
    students["Placement_Probability (%)"] = (prob * 100).round(2)

    print("\n Predictions for new students:")
    print(students.head())

    students.to_csv("students_predictions.csv", index=False)
    print(" Predictions saved as students_predictions.csv")

except FileNotFoundError:
    print("\n No students.csv file found — skipping CSV prediction step.")

# 10. Interactive Input

print("\n Interactive Placement Prediction Demo")
print("Enter details for a new student:\n")

cgpa = float(input("Enter CGPA: "))
internships = int(input("Enter number of Internships: "))
projects = int(input("Enter number of Projects: "))
workshops = int(input("Enter number of Workshops/Certifications: "))
aptitude = float(input("Enter Aptitude Test Score: "))
softskills = float(input("Enter Soft Skills Rating (1–5): "))
extra = input("Participated in Extracurricular Activities? (Yes/No): ").capitalize()
training = input("Attended Placement Training? (Yes/No): ").capitalize()
ssc = float(input("Enter SSC/10th Marks (%): "))
hsc = float(input("Enter HSC/12th Marks (%): "))

# Create DataFrame for new input
new_student = pd.DataFrame({
    "CGPA": [cgpa],
    "Internships": [internships],
    "Projects": [projects],
    "Workshops/Certifications": [workshops],
    "AptitudeTestScore": [aptitude],
    "SoftSkillsRating": [softskills],
    "ExtracurricularActivities": [extra],
    "PlacementTraining": [training],
    "SSC_Marks": [ssc],
    "HSC_Marks": [hsc]
})

# Load trained model and predict
loaded_model = joblib.load("placement_model.joblib")
pred = loaded_model.predict(new_student)
prob = loaded_model.predict_proba(new_student)[:, 1][0]

print(" Prediction Result ")
if pred[0] == 1:
    print(" Student is LIKELY TO BE PLACED!")
else:
    print(" Student is NOT LIKELY TO BE PLACED.")

print(f" Probability of Placement: {prob * 100:.2f}%")
