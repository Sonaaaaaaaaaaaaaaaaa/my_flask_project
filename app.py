from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Placeholder preprocessing objects
scaler = StandardScaler()
le_day = LabelEncoder()

# Dummy dataset to train models
sample_data = pd.DataFrame({
    'day_of_week': [0, 1, 2, 3, 4, 5, 6],
    'num_customers': [30, 50, 80, 60, 70, 40, 55],
    'seating_capacity': [50, 50, 50, 50, 50, 50, 50],
    'weather_condition': [0, 1, 2, 1, 0, 2, 1],
    'local_event': [0, 1, 1, 0, 0, 1, 0],
    'traffic_condition': [1, 2, 3, 4, 1, 2, 3],
    'table_turnover_rate': [2.0, 2.5, 2.0, 2.5, 3.0, 2.0, 2.5]
})

# Train and fit the models
X = sample_data[['day_of_week', 'num_customers', 'seating_capacity', 'weather_condition',
                 'local_event', 'traffic_condition', 'table_turnover_rate']]
y = [0, 1, 2, 3, 4, 1, 0]  # Dummy target variable, represent class labels for wait times

# Initialize models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
dt_model = DecisionTreeClassifier(random_state=42)
nb_model = GaussianNB()
lr_model = LogisticRegression(max_iter=200, random_state=42)

# Fit the models
rf_model.fit(X, y)
dt_model.fit(X, y)
nb_model.fit(X, y)
lr_model.fit(X, y)

# Fit the scaler to the data
scaler.fit(X)

# Nearby restaurant pairs
nearby_restaurant_pairs = {
    "100, Anna Nagar, Chennai": "67, Kilpauk, Chennai",
    "67, Kilpauk, Chennai": "100, Anna Nagar, Chennai",
    "23, Perungudi, Chennai": "104, Velachery, Chennai",
    "104, Velachery, Chennai": "23, Perungudi, Chennai",
    "8, Mount Road, Chennai": "17, Nungambakkam, Chennai",
    "17, Nungambakkam, Chennai": "8, Mount Road, Chennai",
    "92, Adyar, Chennai": "37, Besant Nagar, Chennai",
    "37, Besant Nagar, Chennai": "92, Adyar, Chennai",
    "12, Mylapore, Chennai": "55, T Nagar, Chennai",
    "55, T Nagar, Chennai": "12, Mylapore, Chennai",
}

# Dummy data for dropdowns
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
restaurant_names = list(nearby_restaurant_pairs.keys())

# Function to predict wait time
def predict_wait_time_with_reason(name, day, time, model, model_name):
    day_mapping = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                   "Friday": 4, "Saturday": 5, "Sunday": 6}

    day_encoded = day_mapping.get(day, -1)
    if day_encoded == -1:
        return "Invalid day provided."

    # Simulated input data (for now, use random data or defaults)
    input_data = pd.DataFrame([{
        'day_of_week': day_encoded,
        'num_customers': np.random.randint(20, 100),
        'seating_capacity': 50,
        'weather_condition': np.random.choice([0, 1, 2]),
        'local_event': np.random.choice([0, 1]),
        'traffic_condition': np.random.randint(1, 5),
        'table_turnover_rate': 2.0
    }])

    # Scale input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_class = model.predict(input_scaled)[0]
    wait_time_bins = ['0-5 mins', '5-15 mins', '15-30 mins', '30-60 mins', '60+ mins']
    return wait_time_bins[predicted_class]

# Function to calculate average wait time
def calculate_final_wait_time(predictions):
    wait_time_midpoints = {
        '0-5 mins': 2.5,
        '5-15 mins': 10,
        '15-30 mins': 22.5,
        '30-60 mins': 45,
        '60+ mins': 75
    }
    midpoints = [wait_time_midpoints[pred] for pred in predictions]
    return sum(midpoints) / len(midpoints)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form data
        name = request.form.get("name", "User")
        day = request.form.get("day", "Monday")
        time = request.form.get("time", "12:00 PM")
        restaurant = request.form.get("restaurant", "Unknown")

        # Print to check values
        print(f"Received data - Name: {name}, Day: {day}, Time: {time}, Restaurant: {restaurant}")

        # Make predictions using different models
        rf_wait_time, rf_reason = predict_wait_time_with_reason(name, day, time, rf_model)
        dt_wait_time, dt_reason = predict_wait_time_with_reason(name, day, time, dt_model)
        nb_wait_time, nb_reason = predict_wait_time_with_reason(name, day, time, nb_model)
        lr_wait_time, lr_reason = predict_wait_time_with_reason(name, day, time, lr_model)

        # Collect predictions and reasons
        predictions = [rf_wait_time, dt_wait_time, nb_wait_time, lr_wait_time]
        reasons = [rf_reason, dt_reason, nb_reason, lr_reason]

        # Calculate the final wait time (averaging the midpoints)
        final_wait_time = calculate_final_wait_time(predictions)

        # Print to verify
        print(f"Predicted wait time: {final_wait_time}")
        print(f"Day: {day}, Restaurant: {restaurant}")

        # Determine nearby restaurant suggestion if needed
        if final_wait_time > 20:
            nearby_restaurant = nearby_restaurant_pairs.get(restaurant, "No nearby restaurant found.")
            suggestion = f"Consider visiting {nearby_restaurant}."
        else:
            suggestion = "No need to check for nearby restaurants."

        # Render the result page with the predictions and reasons
        return render_template(
            "result.html",
            name=name,
            day=day,
            time=time,
            restaurant=restaurant,
            final_wait_time=f"{final_wait_time:.1f}",
            suggestion=suggestion,
            predictions=zip(predictions, reasons),  # Send predictions and reasons to the template
        )

    return render_template("index.html", days_of_week=days_of_week, restaurant_names=restaurant_names)

def predict_wait_time_with_reason(name, day, time, model):
    # Mapping the day to numerical value (0 to 6)
    day_mapping = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
        "Friday": 4, "Saturday": 5, "Sunday": 6
    }

    day_encoded = day_mapping.get(day, -1)
    if day_encoded == -1:
        return "Invalid day provided.", "No valid day input"

    # Generate simulated or default input data for prediction
    input_data = pd.DataFrame([{
        'day_of_week': day_encoded,
        'num_customers': np.random.randint(20, 100),  # You can adjust this to match your requirements
        'seating_capacity': 50,
        'weather_condition': np.random.choice([0, 1, 2]),
        'local_event': np.random.choice([0, 1]),
        'traffic_condition': np.random.randint(1, 5),
        'table_turnover_rate': 2.0
    }])

    # Scale the data (ensure it's done properly as per your scaler)
    input_scaled = scaler.transform(input_data)

    # Make prediction
    predicted_class = model.predict(input_scaled)[0]

    # Define the wait time bins
    wait_time_bins = ['0-5 mins', '5-15 mins', '15-30 mins', '30-60 mins', '60+ mins']
    wait_time = wait_time_bins[predicted_class]

    # Provide a reason based on the prediction
    reasons = {
        0: "It's rainy weather.",
        1: "Normal conditions.",
        2: "There is heavy traffic in the area.",
        3: "It's peak hours due to local events.",
        4: "Moderate demand, possibly due to peak hours."
    }

    reason = reasons.get(predicted_class, "No specific reason.")

    return wait_time, reason



# Function to calculate average wait time
def calculate_final_wait_time(predictions):
    wait_time_midpoints = {
        '0-5 mins': 2.5,
        '5-15 mins': 10,
        '15-30 mins': 22.5,
        '30-60 mins': 45,
        '60+ mins': 75
    }
    midpoints = [wait_time_midpoints[pred] for pred in predictions]
    return sum(midpoints) / len(midpoints)

if __name__ == "__main__":
    app.run(debug=True)