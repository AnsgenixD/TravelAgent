# TravelAgent

A Machine Learning powered Command-Line Interface (CLI) application that recommends optimal travel destinations based on user preferences.

## Overview
This project solves the "paradox of choice" in vacation planning. By taking simple user inputs (Climate, Budget, and Activity preferences), it utilizes a **K-Nearest Neighbors (KNN)** classification model built with Scikit-Learn to compare inputs against a historical trip dataset and predict the statistically closest destination.

## Technologies Used
- **Python 3.13**
- **Pandas**: For data ingestion and frame manipulation.
- **Scikit-Learn**: For the ML pipeline (`OneHotEncoder`, `KNeighborsClassifier`).

## Project Structure
- `TravelAgent.py`: The main application script and ML pipeline.
- `data/destinations.csv`: The training dataset containing historical travel categories and their outcomes.
- `requirements.txt`: Python package dependencies.
- `Project_Report.md`: A detailed report on the problem, solution, architecture, and learnings.

## Setup Instructions

1. **Clone the repository** (or download the source code):
   ```bash
   git clone https://github.com/AnsgenixD/TravelAgent
   cd TravelAgent
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Simply run the main python script from your terminal:
```bash
python TravelAgent.py
```

Follow the interactive prompts to enter your:
1. **Climate** (e.g., tropical, cold, temperate)
2. **Budget** (e.g., low, medium, high)
3. **Activity** (e.g., relaxation, adventure, cultural)

The AI Model will analyze your inputs against the `data/destinations.csv` dataset and return the best destination, alongside similarity scores for statistically relevant alternatives!
