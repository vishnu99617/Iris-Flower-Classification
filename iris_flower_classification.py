# iris_flower_classification.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Step 2: Data Preprocessing
def preprocess_data(df):
    # Handle missing values (if any)
    df = df.dropna()
    
    # Split data into features (X) and target (y)
    X = df.drop(columns=['Id', 'Species'])
    y = df['Species']
    
    # Encode the target variable (Species)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Normalize numerical features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Step 3: Train the model
def train_model(X_train, y_train):
    # Initialize Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model on the training data
    model.fit(X_train, y_train)
    print("Model trained successfully")
    return model

# Step 4: Evaluate the model
def evaluate_model(model, X_test, y_test):
    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Classification report for detailed metrics
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Setosa', 'Versicolor', 'Virginica'], yticklabels=['Setosa', 'Versicolor', 'Virginica'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

# Main function to tie all steps together
def main():
    # Load the dataset
    df = load_data("Iris.csv")
    
    if df is not None:
        # Preprocess the data
        X, y = preprocess_data(df)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = train_model(X_train, y_train)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
