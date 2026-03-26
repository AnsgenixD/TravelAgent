---------------------

The Problem:

---------------------

Careful planning is essential to guarantee a seamless and enjoyable vacation. This tool operationalizes those priorities—filtering destinations by budget thresholds, seasonal climate data, and activity tags—so users receive targeted suggestions (for example, filtering for 'beach' activities and temperatures 75–85°F narrows options quickly).


Why it's important:

Spending days off or time unproductive just imagining where to vacation for the holidays is tiresome. I built a straightforward, text-based tool that instantly narrows down the choices based on what has worked for similar travelers in the past- and what you want to see.


---------------------

The Solution:

---------------------

 I created a command-line application that works like an AI travel agent. The tool collects preferred climate, budget, and activity level, level and then uses applies a predictive machine learning machine-learning model to suggest the best recommend an optimal destination.


Approach: 

I chose K-Nearest Neighbors (KNN) because I prioritized top-3 recommendation relevance and simple online updates, which KNN supports by ranking nearest neighbors in feature space. Instead of writing a bunch of hard-coded if/else statements, the app pulls from a CSV dataset of past trip profiles. (i.e data/destinations.csv)

Data Loading: The app reads the dataset dynamically using pandas.

Preprocessing: Using a Scikit-Learn Pipeline and OneHotEncoder to turn text inputs like "tropical" or "medium budget" into numbers and weights the model can actually understand.

 Classification: The KNeighborsClassifier looks at the "distance" between the user's input and the training data to find the closest matching destinations.




---------------------

Key Decisions:

---------------------


Given the goal of identifying the closest match and providing similar runner-up recommendations, k-nearest neighbors was selected over decision-tree methods. Leveraging its probability features was a better fit for a recommendation tool.


  Using a Scikit-Learn Pipeline: Putting the preprocessing and the model into a single Pipeline guarantees that the user's text inputs are handled exactly like the training data. This avoids data leakage and takes care of unexpected inputs smoothly.


  Keeping the dataset separate: I kept the destination data out of the main code and stored it in a destinations.csv file. This makes the project much easier to scale. We can add thousands of new locations later without cluttering the core logic.



---------------------

Challenges Faced:

---------------------


Dealing with categorical data: Machine learning models need numbers, but my features were text like "tropical" or "low budget". Figuring out how to use OneHotEncoder correctly inside a Pipeline instead of just doing manual string replacements was a tough but rewarding learning curve.


  KNN handles unmatched inputs by returning the nearest training examples by distance—for instance, input 'beach+budget+hiking' returned destinations A (distance 0.12), B (0.18), and C (0.25), providing useful runner-ups even when the exact combination was unseen. Its distance weighting finds the closest neighbor even if the user's exact combination of preferences isn't completely identical to a row in the dataset.



---------------------

What I learned:

---------------------


I got a much better grasp on how basic machine learning concepts can be applied to everyday decision-making.


  I learned how to set up a fully functional scikit-learn Pipeline from scratch.


  I realized why project structure is so important. Separating the raw data, dependencies, and main app logic makes the codebase much easier to read and maintain over time.
