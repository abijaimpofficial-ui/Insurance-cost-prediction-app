# Insurance Cost Prediction - Ridge & Lasso Regression

Machine learning pipeline, interactive Streamlit web dashboard, live web application, and standalone native Android APK for predicting medical insurance costs using actuarial regression modeling.

---

### 🌐 Live Web Application
* **Live Web App on GitHub**: [https://abijaimpofficial-ui.github.io/Insurance-cost-prediction-app/](https://abijaimpofficial-ui.github.io/Insurance-cost-prediction-app/)
* **Streamlit Community Cloud**: Deploy `app.py` directly from GitHub using [Streamlit Community Cloud](https://share.streamlit.io/).

---

### 📱 Android Application & APK
* **Download APK directly**: [`InsuranceCostPredictor.apk`](InsuranceCostPredictor.apk)
* **Architecture**: Standalone native Android application (`com.insurance.costprediction`) with Material Design 3 mobile UI and an offline two-way JavaScript/Java bridge (`PredictorBridge.java`).
* **Features**: 100% offline actuarial prediction engine matching the scikit-learn Lasso model to the exact decimal, with real-time risk tier categorization and cost breakdown.
* **Compatibility**: Android 5.0 (API 21) up to Android 14+ (API 34).

---

## 💻 Web App & ML Pipeline Structure
- `index.html` - Live responsive web application hosted on GitHub Pages.
- `insurance_model.py` - Complete ML pipeline (EDA, preprocessing, StandardScaler, feature engineering, GridSearchCV, cross-validation, metrics, plots, and model export).
- `app.py` - Interactive Streamlit web interface.
- `generate_dataset.py` - Benchmark dataset generator (creates `data/insurance.csv` if missing).
- `data/insurance.csv` - Medical insurance charges dataset.
- `outputs/` - Generated visualization plots, correlation heatmaps, `metrics.csv`, and `coefficients.csv`.
- `InsuranceCostPredictor.apk` - Production signed, installable Android APK.
- `android_app/` - Full Android project source code, assets, and manifests.
- `Report_Insurance_Cost_Prediction.docx` - Comprehensive project report.

---

## 🚀 Quick Start (Local Setup)

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows (Linux/macOS: source .venv/bin/activate)

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model and generate outputs
python insurance_model.py

# 4. Launch the Streamlit dashboard
streamlit run app.py
```

---

## 📊 Model Performance Benchmarks
- **Lasso Regression (Selected)**: $R^2 \approx 0.897$ (Test), $0.913$ (Train), $\text{CV } R^2 \approx 0.909$, $\text{MAE} \approx \$2,988.71$
- **Ridge Regression**: $R^2 \approx 0.897$ (Test), $0.913$ (Train), $\text{CV } R^2 \approx 0.909$, $\text{MAE} \approx \$2,990.02$
- **Linear Regression**: $R^2 \approx 0.897$ (Test), $0.913$ (Train), $\text{CV } R^2 \approx 0.909$, $\text{MAE} \approx \$2,983.92$
