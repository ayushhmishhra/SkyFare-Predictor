# SETUP — READ THIS FIRST

This folder has everything EXCEPT 3 files that already exist in your current
repo and were not re-generated (I don't have their content, so nothing to fix
here — just copy them over as-is):

    templates/home.html
    static/css/styles.css
    images/preview.png

## How to assemble your final repo folder

1. Copy this ENTIRE folder's contents into your local SkyFare-Predictor repo,
   overwriting: app.py, flight_price.ipynb, README.md, requirements.txt
   (Your existing templates/, static/, images/ folders stay untouched.)

2. Delete the old flight_rf.pkl — it's replaced by flight_price_pipeline.pkl

3. In `templates/home.html`, check the <form> - it currently posts fields
   named e.g. "Dep_Time", "Arrival_Time", "stops", "airline", "Source",
   "Destination". The new app.py expects those exact same field names, so
   IF your form fields already match (they should, since app.py's logic for
   reading request.form[...] is unchanged) - no HTML edit needed at all.

4. Then follow the run + push steps below.
