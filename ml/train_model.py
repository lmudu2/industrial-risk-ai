import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

print("=" * 80)
print("🚀 TRAINING AI MODEL FOR PREDICTIVE MAINTENANCE (SYNTHETIC DATA)")
print("=" * 80)

# ─────────────────────────────────────────────
# STEP 1: LOAD SYNTHETIC DATA
# ─────────────────────────────────────────────

print("\n📂 Step 1: Loading Synthetic Datasets from CSV...")

base_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_dir, '..', 'data', 'datasets')

sensors_df = pd.read_csv(os.path.join(datasets_dir, 'sensors.csv'))
maint_df = pd.read_csv(os.path.join(datasets_dir, 'maintenance_records.csv'))
assets_df = pd.read_csv(os.path.join(datasets_dir, 'assets.csv'))
industries_df = pd.read_csv(os.path.join(datasets_dir, 'industries.csv'))
costs_df = pd.read_csv(os.path.join(datasets_dir, 'cost_records.csv'))

# Convert timestamps and round to nearest hour to align sensors recorded within the same execution loop
sensors_df['timestamp'] = pd.to_datetime(sensors_df['timestamp']).dt.round('H')
maint_df['start_date'] = pd.to_datetime(maint_df['start_date']).dt.round('H')

# Convert asset dates
assets_df['installation_date'] = pd.to_datetime(assets_df['installation_date'])
assets_df['warranty_expiry'] = pd.to_datetime(assets_df['warranty_expiry'])

print(f"✅ Loaded {len(sensors_df)} sensor readings, {len(maint_df)} maintenance records, and {len(assets_df)} assets")

# ─────────────────────────────────────────────
# STEP 2: PIVOT SENSOR DATA
# ─────────────────────────────────────────────

print("\n🔄 Step 2: Pivoting sensor data for Machine Learning...")

# Pivot so that each timestamp has the 5 sensor types as columns
df = sensors_df.pivot_table(index=['asset_id', 'timestamp'], columns='sensor_type', values='value').reset_index()

# Define the expected 5 core sensors and their default healthy values
core_sensors = ['temperature', 'vibration', 'pressure', 'rpm', 'current']
defaults = {'temperature': 40, 'vibration': 0.5, 'pressure': 100, 'rpm': 1500, 'current': 50}

# Add missing columns if any
for col in core_sensors:
    if col not in df.columns:
        df[col] = np.nan

# Fill missing values: first try asset median, then global default
for col in core_sensors:
    # Asset median
    df[col] = df.groupby('asset_id')[col].transform(lambda x: x.fillna(x.median()))
    # Global default if asset has no history for this sensor
    df[col] = df[col].fillna(defaults[col])

print(f"✅ Pivoted data into {len(df)} discrete time-series snapshots with {len(core_sensors)} sensing features")

# ─────────────────────────────────────────────
# STEP 2.5: FEATURE ENGINEERING (ASSET METADATA & CONTEXT)
# ─────────────────────────────────────────────
print("\n🧬 Step 2.5: Engineering Features from Interlinked Datasets...")

# --- 1. Environmental Context (Industries) ---
assets_df = assets_df.merge(industries_df, left_on='industry_id', right_on='id', suffixes=('', '_ind'))
# Map industries to a continuous risk scaler based on expected harshness
industry_risk_map = {
    'Oil & Gas': 1.0, 'Mining': 0.9, 'Maritime': 0.8, 'Railways': 0.7,
    'Manufacturing': 0.6, 'Energy': 0.5, 'Water Treatment': 0.4, 'Healthcare': 0.2
}
assets_df['industry_risk_factor'] = assets_df['name_ind'].map(industry_risk_map).fillna(0.5)

# --- 2. Maintenance "Medical" History ---
# We calculate history BEFORE the sensor timestamp
maint_df = maint_df.sort_values(['asset_id', 'start_date'])

def get_maintenance_history(row):
    past_maint = maint_df[(maint_df['asset_id'] == row['asset_id']) & (maint_df['start_date'] < row['timestamp'])]
    failures = past_maint[past_maint['maintenance_type'] == 'corrective']
    
    historic_failure_count = len(failures)
    
    if len(past_maint) > 0:
        days_since_last = (row['timestamp'] - past_maint['start_date'].max()).total_seconds() / (24 * 3600)
    else:
        days_since_last = (row['timestamp'] - row['installation_date']).total_seconds() / (24 * 3600)
        
    return pd.Series([historic_failure_count, max(0, days_since_last)])

# Merge base asset info into time-series
df = df.merge(assets_df[['id', 'installation_date', 'warranty_expiry', 'industry_risk_factor']], left_on='asset_id', right_on='id', how='left')

# Calculate Age & Warranty Context
df['asset_age_days'] = (df['timestamp'] - df['installation_date']).dt.total_seconds() / (24 * 3600)
df['warranty_remaining_days'] = (df['warranty_expiry'] - df['timestamp']).dt.total_seconds() / (24 * 3600)

# To make this scalable we'll do an exact merge_asof or vectorized approach for history
# Vectorized Maintenance History Calculation
df = df.sort_values('timestamp')
maint_df['maint_date'] = maint_df['start_date']
maint_df = maint_df.sort_values('maint_date')

# Calculate Days Since Last Maintenance
merged_maint = pd.merge_asof(
    df[['asset_id', 'timestamp']],
    maint_df[['asset_id', 'maint_date']],
    left_on='timestamp',
    right_on='maint_date',
    by='asset_id',
    direction='backward'
)
df['days_since_last_maintenance'] = (df['timestamp'] - merged_maint['maint_date']).dt.total_seconds() / (24 * 3600)
df['days_since_last_maintenance'] = df['days_since_last_maintenance'].fillna(df['asset_age_days']) # fallback to age

# Calculate Historic Failures (Cumulative count before timestamp)
failures_df = maint_df[maint_df['maintenance_type'] == 'corrective'].copy()
failures_df['failure_count'] = 1
failures_df = failures_df.sort_values(['asset_id', 'start_date'])
failures_df['cumulative_failures'] = failures_df.groupby('asset_id')['failure_count'].cumsum()
failures_df = failures_df.sort_values('start_date')

merged_failures = pd.merge_asof(
    df[['asset_id', 'timestamp']],
    failures_df[['asset_id', 'start_date', 'cumulative_failures']],
    left_on='timestamp',
    right_on='start_date',
    by='asset_id',
    direction='backward'
)
df['historic_failure_count'] = merged_failures['cumulative_failures'].fillna(0)

# --- 3. Financial Weighting (Costs) ---
# Sum up all costs prior to the sensor reading
costs_df['cost_date'] = pd.to_datetime(costs_df['created_at'])
costs_df = costs_df.sort_values('cost_date')
costs_df['cumulative_cost'] = costs_df.groupby('work_order_id')['total_cost'].cumsum() # Simplified

# Actually we need cumulative cost per asset. We have to join Wo.
wo_df = pd.read_csv(os.path.join(datasets_dir, 'work_orders.csv'))
costs_asset = costs_df.merge(wo_df[['id', 'asset_id']], left_on='work_order_id', right_on='id')
costs_asset = costs_asset.sort_values('cost_date')
costs_asset['cum_asset_cost'] = costs_asset.groupby('asset_id')['total_cost'].cumsum()

merged_costs = pd.merge_asof(
    df[['asset_id', 'timestamp']],
    costs_asset[['asset_id', 'cost_date', 'cum_asset_cost']],
    left_on='timestamp',
    right_on='cost_date',
    by='asset_id',
    direction='backward'
)
df['cumulative_repair_cost'] = merged_costs['cum_asset_cost'].fillna(0)


# Replace any missing metadata with median/0
df['asset_age_days'] = df['asset_age_days'].fillna(df['asset_age_days'].median()).clip(lower=0)
df['warranty_remaining_days'] = df['warranty_remaining_days'].fillna(0)

# The new extended feature set
ml_features = core_sensors + [
    'asset_age_days', 
    'warranty_remaining_days', 
    'industry_risk_factor',
    'days_since_last_maintenance',
    'historic_failure_count',
    'cumulative_repair_cost'
]

print(f"✅ Engineered {len(ml_features)} total features (Telemetry + Contextual History)")

# ─────────────────────────────────────────────
# STEP 3: CALCULATE RUL (REMAINING USEFUL LIFE)
# ─────────────────────────────────────────────

print("\n🔧 Step 3: Calculating Remaining Useful Life (RUL) to next failure...")

# We calculate time until the next CORRECTIVE maintenance event
failures_df = maint_df[maint_df['maintenance_type'] == 'corrective'][['asset_id', 'start_date']].copy()

# Sort both dataframes to use merge_asof (matches each reading to the NEXT failure)
df = df.sort_values('timestamp')
failures_df = failures_df.sort_values('start_date')

df = pd.merge_asof(
    df,
    failures_df,
    left_on='timestamp',
    right_on='start_date',
    by='asset_id',
    direction='forward'
)

# Calculate RUL in days
df['RUL_days'] = (df['start_date'] - df['timestamp']).dt.total_seconds() / (24 * 3600)

# Drop any readings that occurred AFTER the last failure (since we cannot know their true RUL yet)
# merge_asof leaves start_date as NaT for these rows
df = df.dropna(subset=['start_date', 'RUL_days'])

print(f"✅ Calculated RUL for {len(df)} valid readings")

# ─────────────────────────────────────────────
# STEP 4: CREATE RISK LABELS
# ─────────────────────────────────────────────

print("\n🎯 Step 4: Creating Risk Categories...")

def categorize_risk(rul_days):
    """
    Convert RUL (days) into actionable risk categories based on synthetic generation
    """
    if rul_days > 15:
        return 0  # Healthy
    elif rul_days > 7:
        return 1  # Warning
    elif rul_days > 3:
        return 2  # High Risk
    else:
        return 3  # Critical

df['risk_label'] = df['RUL_days'].apply(categorize_risk)

risk_labels = ['Healthy', 'Warning', 'High Risk', 'Critical']
print("✅ Risk category distribution:")
for i, label in enumerate(risk_labels):
    count = (df['risk_label'] == i).sum()
    pct = (count / len(df)) * 100 if len(df) > 0 else 0
    print(f"   {label:12s}: {count:6d} ({pct:5.1f}%)")

# ─────────────────────────────────────────────
# STEP 4.5: EXPORT MASTER ML DATASET (FOR INSPECTION)
# ─────────────────────────────────────────────

print("\n💾 Step 4.5: Exporting Master ML Dataset to CSV...")
export_path = os.path.join(datasets_dir, 'ml_training_data_snapshot.csv')
df.to_csv(export_path, index=False)
print(f"✅ Master dataset (11 features + labels) saved to: {export_path}")

# ─────────────────────────────────────────────
# STEP 5: PREPARE FEATURES FOR MACHINE LEARNING
# ─────────────────────────────────────────────

print("\n🔢 Step 5: Preparing Features...")

X = df[ml_features].values  # Features (7 dimensions now)
y = df['risk_label'].values  # Labels

print(f"✅ Feature matrix: {X.shape[0]} samples × {X.shape[1]} engineered features")

# ─────────────────────────────────────────────
# STEP 6: SPLIT DATA
# ─────────────────────────────────────────────

print("\n✂️  Step 6: Splitting Data (80% train, 20% validation)...")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"✅ Training set:   {len(X_train):6d} samples")
print(f"✅ Validation set: {len(X_val):6d} samples")

# ─────────────────────────────────────────────
# STEP 7: STANDARDIZE FEATURES
# ─────────────────────────────────────────────

print("\n📏 Step 7: Standardizing Features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("✅ Data scaled successfully")

# ─────────────────────────────────────────────
# STEP 8: TRAIN THE MODEL!
# ─────────────────────────────────────────────

print("\n🧠 Step 8: Training Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

model.fit(X_train_scaled, y_train)

# ─────────────────────────────────────────────
# STEP 9: EVALUATE MODEL
# ─────────────────────────────────────────────

print("\n📊 Step 9: Evaluating Model Performance...")

y_pred = model.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred)

print(f"\n{'='*60}")
print(f"🎯 VALIDATION ACCURACY: {accuracy:.2%}")
print(f"{'='*60}")

print("\n📋 Detailed Classification Report:")
print(classification_report(y_val, y_pred, target_names=risk_labels, digits=2))

# ─────────────────────────────────────────────
# STEP 10: SAVE THE MODEL
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────────────

print("\n💾 Step 10: Saving Models...")

models_dir = os.path.join(base_dir, 'models')
os.makedirs(models_dir, exist_ok=True)

joblib.dump(model, os.path.join(models_dir, 'failure_predictor.pkl'))
joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
joblib.dump(ml_features, os.path.join(models_dir, 'feature_columns.pkl'))  # We save ml_features instead of core_sensors

print(f"✅ Model saved to {models_dir}/failure_predictor.pkl")
print(f"✅ Scaler saved to {models_dir}/scaler.pkl")
print(f"✅ Feature columns saved to {models_dir}/feature_columns.pkl")

print("\n" + "=" * 80)
print("🎉 TRAINING COMPLETE!")
print("=" * 80)