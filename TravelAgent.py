import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import time
import warnings

# Suppress warnings to keep the command line interface clean
warnings.filterwarnings("ignore") 

def load_data():
    """Loads destination training data from the CSV file."""
    # Resolve absolute path to the data folder so it runs from anywhere
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "data", "destinations.csv")
    
    try:
        df = pd.DataFrame(pd.read_csv(csv_path))
        return df
    except FileNotFoundError:
        print(f"❌ Error: Could not find training data at {csv_path}")
        print("Please ensure the 'data/destinations.csv' file exists.")
        exit(1)

def get_valid_input(prompt, valid_options):
    while True:
        print(f"\n{prompt}")
        for i, option in enumerate(valid_options, 1):
            print(f"  {i}. {option.capitalize()}")
        try:
            choice = int(input("\nEnter the number of your choice: "))
            if 1 <= choice <= len(valid_options):
                return valid_options[choice - 1]
            else:
                print("⚠️  Invalid choice. Please pick a number from the list.")
        except ValueError:
            print("⚠️  Please enter a valid number.")

def train_model(df):
    """Builds and trains the ML Pipeline."""
    X = df[["Climate", "Budget", "Activity"]]
    y = df["Destination"]

    # 1. Build the Machine Learning Pipeline
    # OneHotEncoder transforms our text categories into numerical vectors
    # KNeighborsClassifier clusters and finds the 'nearest' historical trips matching our inputs
    pipeline = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown='ignore')),
        ("classifier", KNeighborsClassifier(n_neighbors=3, weights='distance'))
    ])

    # 2. Fit (Train) the model on the data
    pipeline.fit(X, y)
    return pipeline

def main():
    print("="*60)
    print("ML-Powered AI Travel Agent")
    print("="*60)
    print("This AI uses a K-Nearest Neighbors (KNN) Machine Learning")
    print("model trained on historical data to classify destinations!\n")
    time.sleep(1)

    # 1. Load Data
    df = load_data()

    # 2. Train Model
    pipeline = train_model(df)

    climate = get_valid_input("What type of climate do you prefer?",
                              ["tropical", "monsoon", "cold", "temperate", "desert"])
    budget = get_valid_input("What is your budget for this trip?",
                             ["low", "medium", "high"])
    activity = get_valid_input("What is your primary goal for this trip?",
                               ["relaxation", "adventure", "cultural", "nightlife"])

    print("\n🤖 ML Model is extracting features and making a prediction...")
    time.sleep(1.5)

    # 3. Preparing user data for prediction matching our training pipeline format
    user_data = pd.DataFrame([[climate, budget, activity]], columns=["Climate", "Budget", "Activity"])
    
    # 4. Extract probabilities from the nearest neighbors to show top choices
    probabilities = pipeline.predict_proba(user_data)[0]
    classes = pipeline.classes_
    
    # Map classes to their probability similarity scores
    prob_scores = [(classes[i], probabilities[i]) for i in range(len(classes)) if probabilities[i] > 0]
    prob_scores.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*50)
    if prob_scores:
        top_dest = prob_scores[0][0]
        print(f"🎉 The ML Model highly recommends: ** {top_dest} **!")
        
        # Show statistical runner ups
        if len(prob_scores) > 1:
            print("\nOther statistically significant matches in model:")
            for dest, score in prob_scores[1:4]: # Show up to 3 alternatives
                print(f"  - {dest} (Similarity Score: {score:.2f})")
    else:
        # Fallback raw prediction
        prediction = pipeline.predict(user_data)[0]
        print(f"🎉 The ML Model's base prediction is: ** {prediction} **!")

    print("="*50 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGoodbye! Have a safe ML journey! 🛫\n")
