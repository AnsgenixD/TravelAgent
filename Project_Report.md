1. The Problem

The Issue: Planning a vacation can easily become overwhelming. There are just too many options out there, making it hard for people to pick a place that actually fits their budget, their ideal weather, and the activities they want to do.

Why it matters: People often spend hours or even days researching trips, which just leads to decision fatigue. I wanted to build a straightforward, text-based tool that instantly narrows down the choices based on what has worked for similar travelers in the past. This saves time and gives users a solid, personalized starting point.
2. The Solution & Approach

Solution: I created a command-line application that works like an AI travel agent. It asks the user for their preferred climate, budget, and activity level, and then uses a predictive machine learning model to suggest the best destination.

Approach: I chose the K-Nearest Neighbors (KNN) classification algorithm for this. Instead of writing a bunch of hard-coded if/else statements, the app pulls from a CSV dataset of past trip profiles.

    Data Loading: The app reads the dataset dynamically using pandas.

    Preprocessing: I set up a Scikit-Learn Pipeline and used OneHotEncoder to turn text inputs like "tropical" or "medium budget" into numbers the model can actually understand.

    Classification: The KNeighborsClassifier looks at the "distance" between the user's input and the training data to find the closest matching destinations.

3. Key Decisions

    Choosing KNN over Decision Trees: I originally thought about using a Decision Tree. However, since my goal was to find the closest match and also offer runner-up options based on similarity, KNN made more sense. Leveraging its probability features was a better fit for a recommendation tool.

    Using a Scikit-Learn Pipeline: Putting the preprocessing and the model into a single Pipeline guarantees that the user's text inputs are handled exactly like the training data. This avoids data leakage and takes care of unexpected inputs smoothly.

    Keeping the dataset separate: I kept the destination data out of the main code and stored it in a destinations.csv file. This makes the project much easier to scale. We can add thousands of new locations later without cluttering the core logic.

4. Challenges Faced

    Dealing with categorical data: Machine learning models need numbers, but my features were text like "tropical" or "low budget". Figuring out how to use OneHotEncoder correctly inside a Pipeline instead of just doing manual string replacements was a tough but rewarding learning curve.

    Handling unmatched inputs: I had to account for users typing in combinations that didn't perfectly match the training data. KNN naturally solved this problem. Its distance weighting finds the closest neighbor even if the user's exact combination of preferences isn't completely identical to a row in the dataset.

5. What I Learned

    I got a much better grasp on how basic machine learning concepts can be applied to everyday decision-making.

    I learned how to set up a fully functional scikit-learn Pipeline from scratch.

    I realized why project structure is so important. Separating the raw data, dependencies, and main app logic makes the codebase much easier to read and maintain over time.
