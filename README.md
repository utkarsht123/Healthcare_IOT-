# Healthcare IoT Analytics

A PyTorch pipeline that classifies patient health status ("Healthy" vs "Unhealthy") from
time-series vital-sign readings collected by IoT sensors — temperature, systolic/diastolic
blood pressure, heart rate, and device battery level. It combines a CNN + Bidirectional
LSTM + self-attention model with an interactive Plotly Dash dashboard for per-patient
monitoring. The script started as a Google Colab notebook (`iot_analytics.py` is the
exported `.py` version).

## What it does

- **Preprocesses sensor data**: normalizes features with `StandardScaler` and builds
  sliding-window sequences (default 30 timesteps) from a patient vitals CSV.
- **Balances classes**: oversamples underrepresented health-status classes by adding
  Gaussian noise to existing samples (`_augment_rare_classes`).
- **Trains a hybrid deep learning model** (`BiLSTMAttentionModel`): parallel 1D-CNN and
  bidirectional LSTM branches, the LSTM output passed through a multi-head self-attention
  layer, concatenated and fed through fully connected layers to a classification head.
  Trained with Adam + cross-entropy loss and manual early stopping (patience of 20 epochs,
  up to 500 max epochs).
- **Evaluates the model**: accuracy, precision, recall, F1, confusion matrix, and
  training curve / attention-weight visualizations (matplotlib/seaborn).
- **Assesses individual patients**: `assess_custom_patient()` runs a single patient's
  30-step vitals sequence through the trained model and reports a risk classification;
  `main()` demonstrates this on three hand-authored example patients (healthy, moderate
  risk, high risk) and can also prompt for custom vitals on the command line.
- **Interactive dashboard** (`create_dashboard`): a Dash app with a patient selector,
  a live vitals graph, the model's prediction, a feature-importance bar chart (based on
  per-feature standard deviation), and a z-score-based anomaly plot that flags outlier
  readings in each vital sign.

## Tech Stack

- **ML/DL**: PyTorch (CNN, BiLSTM, `nn.MultiheadAttention`), scikit-learn
  (`StandardScaler`, `MinMaxScaler`, `KNNImputer`, train/test split, metrics)
- **Data**: NumPy, pandas
- **Visualization**: Matplotlib, Seaborn, Plotly, Dash

## Running it

This repository contains a single script (`iot_analytics.py`); there's no `requirements.txt`
included yet, so install the libraries it imports:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch plotly dash
```

The script expects a CSV named `healthcare_iot_target_dataset.csv` in the working
directory (not included in this repo) with at least these columns: `Patient_ID`,
`Timestamp`, `Sensor_ID`, `Sensor_Type`, `Target_Blood_Pressure`, `Target_Heart_Rate`,
`Target_Health_Status` (used as the classification label — "Healthy"/"Unhealthy"), plus
the vital-sign feature columns (e.g. temperature, blood pressure, heart rate, battery
level).

```bash
python iot_analytics.py
```

Running it end-to-end loads the dataset, trains the model, prints evaluation metrics,
walks through the three example patient assessments, and then optionally prompts for
custom patient vitals in the terminal.

## Notes for anyone extending this

- The pipeline currently lives in one script; splitting data loading, the model, training,
  and the dashboard into separate modules would make it easier to test and reuse.
- The sample dataset isn't checked into the repo, so a compatible CSV is required to run
  the training/evaluation flow end-to-end.
