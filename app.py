import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os

# Load the dataset after downloading
df = pd.read_csv('./data/raw/AB_NYC_2019.csv')

# Continue with preprocessing and other tasks
print(df.head())

# Ensure the data directory exists
data_dir = "./data/raw/"
os.makedirs(data_dir, exist_ok=True)

# Load the dataset
url = "https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv"
csv_path = os.path.join(data_dir, "AB_NYC_2019.csv")
df = pd.read_csv(csv_path)

# Data exploration
print(df.head())
print(df.info())
print(df.describe())

# Dropping irrelevant columns
df = df.drop(columns=['id', 'host_name', 'last_review'])  # Example of irrelevant columns

# Handling missing values
df = df.dropna()

# Convert categorical variables to numeric using One-Hot Encoding or Label Encoding
df = pd.get_dummies(df, drop_first=True)

# Features: Drop the target column
X = df.drop('price', axis=1)  # Assuming 'price' is the target variable

# Target
y = df['price']

# Checking correlation matrix (optional, to understand relationships)
corr_matrix = df.corr()
print(corr_matrix['price'].sort_values(ascending=False))

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

df = pd.read_csv('data/raw/AB_NYC_2019.csv')
print(df.info())
print(df.isna().sum())
