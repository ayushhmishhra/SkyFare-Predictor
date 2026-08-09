# ✈️ SkyFare Predictor

An end-to-end Machine Learning web application that predicts flight ticket prices, built on a cleaned data pipeline, a compared/validated model selection, and an sklearn `Pipeline` deployed via Flask.

---

## 🌐 Live Demo

🚀 https://skyfare-predictor.onrender.com
*(redeploy after pushing these changes — see Deployment section)*

---

## 🛠️ Tech Stack

**ML:** Python, Pandas, NumPy, Scikit-learn
**Backend:** Flask, Gunicorn
**Frontend:** HTML, CSS
**Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

```
SkyFare-Predictor/
│
├── Flight Fare/
│   ├── Data_Train.xlsx
│   ├── Test_set.xlsx
│   └── Sample_submission.xlsx
│
├── results/
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── eda_plots.png
│
├── images/
│   └── preview.png
│
├── static/css/styles.css
├── templates/home.html
│
├── app.py
├── train.py
├── compare_xgboost.py
├── flight_price.ipynb
├── flight_price_pipeline.pkl
├── requirements.txt
├── Procfile
├── .python-version
└── LICENSE
```

---

## 🤖 Machine Learning Workflow

1. **Data Cleaning** — removed 220 duplicate rows, dropped 1 row with missing `Route`/`Total_Stops`, fixed inconsistent category labels (`'No Info'` vs `'No info'`), removed 1 corrupted row (`Duration = '5m'` on a 2-stop flight)
2. **Feature Engineering** — parsed journey date, departure/arrival time, and **duration directly from the airline's reported duration string** (not derived by subtracting timestamps, which breaks on overnight flights)
3. **Preprocessing** — `ColumnTransformer` (one-hot encoding for Airline/Source/Destination) wrapped in an sklearn `Pipeline` — no manual if/elif encoding
4. **Model Comparison** — Linear Regression vs Random Forest vs Gradient Boosting
5. **Cross-Validation** — 5-fold CV on the winning model to confirm stability
6. **Feature Importance** — inspected which features actually drive price
7. **Deployment** — pipeline saved as a single `.pkl`, served via Flask

---

## 📈 Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **Random Forest** | **1833.96** | **1137.39** | **0.837** |
| Gradient Boosting | 2186.34 | 1532.13 | 0.769 |
| Linear Regression | 2888.30 | 1998.04 | 0.597 |

5-fold CV (Random Forest): mean RMSE ≈ **1992.85**, std ≈ **106.17**

Top features driving price: `Duration_total_mins` (42%), `Journey_day` (12%), `Airline_Jet Airways Business` (7%)

Full details in `results/` and in the write-up below.

---

## 🔧 What Changed From the Original Version

- **Fixed a duration-calculation bug**: the original app computed duration as `abs(Arrival_hour - Dep_hour)`, which breaks for flights crossing midnight. Now computed correctly from full datetimes, matching how duration is parsed during training.
- **Replaced ~250 lines of manual if/elif one-hot encoding** in `app.py` with an sklearn `Pipeline`, removing the risk of train/serve encoding mismatch.
- **Added real model comparison** (previously only Random Forest was tried, with no baseline).
- **Added 5-fold cross-validation** to confirm results aren't a lucky split.
- **Capped Random Forest depth** (`max_depth=18`, `min_samples_leaf=2`) — this actually *improved* test RMSE (1917 → 1834) versus an unlimited-depth forest, while also shrinking the model file from 98MB to 40MB.

---

## 🚀 Installation

```bash
git clone https://github.com/ayushhmishhra/SkyFare-Predictor.git
cd SkyFare-Predictor
pip install -r requirements.txt
```

**Retrain the model (optional — a trained pipeline is already included):**
```bash
python train.py
```

**Run the app:**
```bash
python app.py
```
Open `http://127.0.0.1:5000`

---

## 👨‍💻 Author

**Ayush Mishra** — [GitHub](https://github.com/ayushhmishhra)

## 📄 License

MIT License
