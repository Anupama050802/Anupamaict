from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# --- LOAD MODELS/FILES HERE ---
# e.g., model = joblib.load('model.joblib')
with open("gradient.pkl",'rb') as f:
  model= pickle.load(f)


@app.route('/')
def home():
    """Renders the main page (index.html)."""
    # Pass an empty string so the template doesn't error on first load
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    """Receives form data, makes a prediction, and returns it."""
    
    try:
        # --- 1. Get data from the form ---
        # Example:
        # feature1 = request.form['feature1']
        # feature2 = float(request.form['feature2'])
        
        # --- 2. Process the data and make a prediction ---
        # Example:
        # input_data = [feature1, feature2]
        # prediction = model.predict([input_data])
        # output = prediction[0]

        song_english = int(request.form["Song_In_English"])
        group_solo = request.form["Group_Solo"]
        danceability = float(request.form["danceability"])
        energy = float(request.form["energy"])

        if group_solo == "Solo":
            group_solo_val = 1
        else:
            group_solo_val = 0

        input_data = [song_english, group_solo_val,danceability,energy]
        prediction = model.predict([input_data])
        output = prediction[0]
        
        # Placeholder text - replace with your model's output
        output = "Prediction Result"
        prediction_text = f"The result is: {output}"

    except Exception as e:
        prediction_text = f"An error occurred: {e}"

    # --- 3. Render the page again with the prediction ---
    return render_template('index.html', prediction_text=prediction_text)

# BONUS TASK ROUTE
@app.route("/voting_history", methods=["POST"])
def voting_history():

    # Load dataset
    voting = pd.read_csv("Voting Final.csv")

    selected_country = request.form["country"]

    # Filter votes received by selected country
    filtered = voting[voting["Country"] == selected_country]

    # Calculate total points from each giver
    result = (
        filtered.groupby("Giver")["Points"]
        .sum()
        .sort_values(ascending=False)
    )

    best_friend = result.index[0]

    return render_template(
        "index.html",
        voting_result=f"{best_friend} historically gives the most points to {selected_country}"
    )

if __name__ == "__main__":
    app.run(debug=True)