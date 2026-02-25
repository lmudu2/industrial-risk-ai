import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database import SessionLocal
from models import CostRecord, WorkOrder, Asset, Industry
from sqlalchemy.orm import joinedload
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

def train_cost_model():
    print("=" * 80)
    print("💰 TRAINING MAINTENANCE COST FORECASTING MODEL (v2 – Log-Scale)")
    print("=" * 80)

    # ─────────────────────────────────────────────
    # STEP 1: EXTRACT DATA
    # ─────────────────────────────────────────────
    
    print("\n📊 Step 1: Extracting Data from Database...")
    
    db = SessionLocal()
    
    records = db.query(CostRecord).options(
        joinedload(CostRecord.work_order).joinedload(WorkOrder.asset).joinedload(Asset.industry)
    ).all()
    
    data = []
    
    for r in records:
        if not r.work_order or not r.work_order.asset:
            continue
            
        wo = r.work_order
        asset = wo.asset
        
        # Calculate asset age at time of work order
        age_days = (wo.created_at - asset.purchase_date).days
        
        data.append({
            'total_cost': r.total_cost,
            'priority': wo.priority.value if hasattr(wo.priority, 'value') else str(wo.priority),
            'asset_type': asset.asset_type,
            'industry': asset.industry.name if asset.industry else 'Unknown',
            'asset_age_days': age_days,
            'description_len': len(wo.description) if wo.description else 0
        })
    
    db.close()
    
    df = pd.DataFrame(data)
    print(f"✅ Extracted {len(df)} records")
    
    # Show cost distribution per priority to verify model can learn the difference
    print("\n📈 Cost distribution by priority:")
    for p in ['low', 'medium', 'high', 'critical']:
        subset = df[df['priority'] == p]['total_cost']
        if len(subset) > 0:
            print(f"   {p:10s}: median=${subset.median():>12,.0f}  mean=${subset.mean():>12,.0f}  n={len(subset)}")
    
    # ─────────────────────────────────────────────
    # STEP 2: LOG-TRANSFORM TARGET
    # ─────────────────────────────────────────────
    
    print("\n🔢 Step 2: Applying Log-Transform to target...")
    
    # Log-transform makes the model treat $100→$200 the same as $100K→$200K
    # This is critical for cost data spanning 5 orders of magnitude
    df['log_cost'] = np.log1p(df['total_cost'])
    
    X = df[['priority', 'asset_type', 'industry', 'asset_age_days', 'description_len']]
    y = df['log_cost']
    
    categorical_features = ['priority', 'asset_type', 'industry']
    numerical_features = ['asset_age_days', 'description_len']
    
    print(f"   Target range: ${df['total_cost'].min():,.0f} – ${df['total_cost'].max():,.0f}")
    print(f"   Log range: {y.min():.2f} – {y.max():.2f}")
    
    # ─────────────────────────────────────────────
    # STEP 3: BUILD PIPELINE (GradientBoosting)
    # ─────────────────────────────────────────────
    
    print("\n🛠️  Step 3: Building Pipeline (GradientBoosting)...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        ))
    ])
    
    # ─────────────────────────────────────────────
    # STEP 4: TRAIN
    # ─────────────────────────────────────────────
    
    print("\n🧠 Step 4: Training Model...")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    print("✅ Model trained successfully")
    
    # ─────────────────────────────────────────────
    # STEP 5: EVALUATE (in log AND real-dollar space)
    # ─────────────────────────────────────────────
    
    print("\n📊 Step 5: Evaluation...")
    
    y_pred_log = model.predict(X_test)
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)
    
    r2_log  = r2_score(y_test, y_pred_log)
    mae_log = mean_absolute_error(y_test, y_pred_log)
    r2_real = r2_score(y_test_real, y_pred_real)
    mae_real = mean_absolute_error(y_test_real, y_pred_real)
    
    print(f"   [Log-space]  R²: {r2_log:.3f}  |  MAE: {mae_log:.3f}")
    print(f"   [Real $]     R²: {r2_real:.3f}  |  MAE: ${mae_real:,.0f}")
    
    # Verify the model actually differentiates priorities
    print("\n🔍 Priority differentiation test:")
    test_asset = X_test['asset_type'].mode().iloc[0] if not X_test.empty else 'Conveyor Belt'
    test_industry = X_test['industry'].mode().iloc[0] if not X_test.empty else 'Manufacturing'
    for p in ['low', 'medium', 'high', 'critical']:
        test_input = pd.DataFrame([{
            'priority': p, 'asset_type': test_asset,
            'industry': test_industry, 'asset_age_days': 1000,
            'description_len': 50
        }])
        pred_log = model.predict(test_input)[0]
        pred_real = np.expm1(pred_log)
        print(f"   {p:10s} → ${pred_real:>12,.0f}")
    
    # ─────────────────────────────────────────────
    # STEP 6: SAVE
    # ─────────────────────────────────────────────
    
    print("\n💾 Step 6: Saving Model...")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/cost_predictor.pkl')
    
    print("✅ Model saved to ml/models/cost_predictor.pkl")
    print("=" * 80)

if __name__ == "__main__":
    train_cost_model()
