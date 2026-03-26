# Project Report: AI Travel Agent Recommender

## 1. The Problem
**Problem Statement:** Planning a vacation is often an overwhelming experience due to the "paradox of choice". Consumers are bombarded with countless destinations, making it difficult to find a location that perfectly aligns with their specific weather preferences, financial constraints, and desired activities. 
**Why it matters:** People spend hours—sometimes days—researching potential trips, which leads directly to decision fatigue. A simple text-based tool that can instantly narrow down choices based on historical success patterns saves significant time and provides highly personalized starting points for travelers.

## 2. The Solution & Approach
**Solution:** I built a Machine Learning Command-Line Interface (CLI) application that acts as an AI Travel Agent. It asks the user three primary questions (Climate, Budget, Activity) and uses a predictive model to output the optimal destination.

**Approach:** I used **K-Nearest Neighbors (KNN)** classification to solve this problem. Rather than using rigid, hard-coded `if/else` logic, the system relies on a dataset of historical trip profiles located in a central CSV.
1. **Data Ingestion:** The app reads base data dynamically using `pandas`.
2. **Preprocessing:** I implemented a Scikit-Learn `Pipeline` with a `OneHotEncoder` to transform categorical text inputs (like "tropical" or "medium") into a numerical feature space.
3. **Classification:** The `KNeighborsClassifier` calculates the "distance" between the user's input and the training dataset to find the most statistically similar destinations.

## 3. Key Decisions
- **KNN over Decision Trees:** I initially considered a standard Decision Tree classifier. However, since the goal is to find the *closest match* and provide runner-up alternatives weighted by similarity scores, KNN (using `predict_proba` logic) was the most elegant architectural choice for a recommendation system.
- **Scikit-Learn Pipeline:** Structuring the preprocessing and model inference into a unified `Pipeline` ensures that the user's raw categorical inputs are transformed exactly the same way as the training data. This prevents data leakage and handles unknown categories seamlessly.
- **Modular Dataset:** Decoupling the destination matrix from the source code and putting it into `data/destinations.csv` allows for immediate scalability. Thousands of rows can be added without bloating the engine logic.

## 4. Challenges Faced
- **Handling Categorical Data:** ML models natively require numerical inputs, but our features were purely text-based ("tropical", "low budget"). Learning to properly implement `OneHotEncoder`—especially within an abstraction like `Pipeline` rather than manually replacing strings—was a significant but rewarding hurdle.
- **Handling Proxies:** If a user inputs a combination that doesn't perfectly match the precise training dataset, the system inherently needed a way to gracefully degrade. KNN's distance weighting resolved this by finding the "closest" neighbor even if it wasn't a 100% exact feature permutation.

## 5. What I Learned
- I deepened my understanding of how fundamental Machine Learning methodologies can be applied directly to everyday decision-making problems.
- I learned how to build a robust, end-to-end `scikit-learn` Pipeline from scratch.
- I grasped the critical importance of modular project structure architecture (separating raw data, dependencies, environment ignore files, and application logic) to make code readable and maintainable over time.
