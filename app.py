from flask import Flask, request, render_template
from flask_cors import cross_origin
import pandas as pd
import pickle
import re

app = Flask(__name__)

# The pickled object is now a full sklearn Pipeline (preprocessing + model),
# not a bare model. So we don't need any manual one-hot encoding here anymore -
# the pipeline handles Airline/Source/Destination encoding internally, the exact
# same way it did during training. This removes the risk of train/serve skew.
pipeline = pickle.load(open("flight_price_pipeline.pkl", "rb"))


@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":

        # ---- Departure ----
        dep_dt = pd.to_datetime(request.form["Dep_Time"], format="%Y-%m-%dT%H:%M")
        Journey_day = dep_dt.day
        Journey_month = dep_dt.month
        Journey_weekday = dep_dt.weekday()
        Dep_hour = dep_dt.hour
        Dep_min = dep_dt.minute

        # ---- Arrival ----
        arr_dt = pd.to_datetime(request.form["Arrival_Time"], format="%Y-%m-%dT%H:%M")
        Arrival_hour = arr_dt.hour
        Arrival_min = arr_dt.minute

        # ---- Duration: computed correctly, accounting for overnight flights ----
        # (This fixes the old bug where dep/arrival hours were subtracted directly,
        # which broke for any flight crossing midnight.)
        duration_minutes = int((arr_dt - dep_dt).total_seconds() / 60)
        if duration_minutes <= 0:
            duration_minutes += 24 * 60  # arrival is next day
        Duration_hours = duration_minutes // 60
        Duration_mins = duration_minutes % 60

        # ---- Stops: the home.html form already sends a numeric string (0-4),
        #      which matches the ordinal encoding used during training directly.
        #      (Earlier version wrongly assumed the form sent text labels like
        #      "non-stop" - fixed after testing surfaced the KeyError.) ----
        Total_Stops = int(request.form["stops"])

        # ---- Build a single-row DataFrame with the SAME column names/order
        #      the pipeline was trained on. The ColumnTransformer inside the
        #      pipeline takes care of one-hot encoding Airline/Source/Destination
        #      exactly as it did at training time - no manual if/elif needed. ----
        input_df = pd.DataFrame([{
            "Airline": request.form["airline"],
            "Source": request.form["Source"],
            "Destination": request.form["Destination"],
            "Total_Stops": Total_Stops,
            "Journey_day": Journey_day,
            "Journey_month": Journey_month,
            "Journey_weekday": Journey_weekday,
            "Dep_hour": Dep_hour,
            "Dep_min": Dep_min,
            "Arrival_hour": Arrival_hour,
            "Arrival_min": Arrival_min,
            "Duration_hours": Duration_hours,
            "Duration_mins": Duration_mins,
            "Duration_total_mins": duration_minutes,
        }])

        prediction = pipeline.predict(input_df)
        output = round(float(prediction[0]), 2)

        return render_template(
            "home.html",
            prediction_text="Your Flight price is Rs. {}".format(output)
        )

    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)