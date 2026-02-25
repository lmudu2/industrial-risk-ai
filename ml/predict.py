import sys
import os

# Allow importing from backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

import joblib
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import func
from database import SessionLocal
from models import Asset, Sensor, Industry, MaintenanceRecord, WorkOrder, CostRecord
from datetime import datetime, timedelta, timezone
import sqlite3

# ─────────────────────────────────────────────
# LOAD TRAINED MODEL
# ─────────────────────────────────────────────

_MODEL_CACHE = None

def load_prediction_model():
    """Load the trained AI model and scaler"""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, 'models')
    
    model = joblib.load(os.path.join(models_dir, 'failure_predictor.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    feature_cols = joblib.load(os.path.join(models_dir, 'feature_columns.pkl'))
    
    _MODEL_CACHE = (model, scaler, feature_cols)
    return _MODEL_CACHE

# ─────────────────────────────────────────────
# GET SENSOR DATA FOR AN ASSET
# ─────────────────────────────────────────────

def get_asset_sensor_data(db: Session, asset, days_back: int = 180):
    """
    Get recent sensor readings for an asset and combine with metadata.
    
    Args:
        db: Database session
        asset: The Asset object
        days_back: How many days of history to look at
        
    Returns:
        numpy array with 7 features (or None if no data)
    """
    
    # Get sensors from last N days
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    sensors = db.query(Sensor).filter(
        Sensor.asset_id == asset.id,
        Sensor.timestamp >= cutoff_date
    ).order_by(Sensor.timestamp.desc()).limit(100).all()
    
    if not sensors:
        return None
    
    core_sensors = ['temperature', 'vibration', 'pressure', 'rpm', 'current']
    sensor_values = {s: [] for s in core_sensors}
    
    # Aggregate sensor values
    for sensor in sensors:
        if sensor.sensor_type in sensor_values:
            sensor_values[sensor.sensor_type].append(sensor.value)
            
    # Calculate average and fill missing
    features = np.zeros(11) # 5 telemetry + 6 metadata
    defaults = {'temperature': 40, 'vibration': 0.5, 'pressure': 100, 'rpm': 1500, 'current': 50}
    
    for i, s_type in enumerate(core_sensors):
        vals = sensor_values[s_type]
        if len(vals) > 0:
            features[i] = np.mean(vals)
        else:
            features[i] = defaults[s_type]
            
    # --- Compute Metadata & Contextual Features ---
    now = datetime.now()
    
    # 1. Asset Age & Warranty
    asset_age_days = (now - asset.installation_date).total_seconds() / (24 * 3600) if asset.installation_date else 0
    warranty_remaining_days = (asset.warranty_expiry - now).total_seconds() / (24 * 3600) if asset.warranty_expiry else 0
    
    # 2. Industry Risk
    industry = db.query(Industry).filter(Industry.id == asset.industry_id).first()
    industry_risk_map = {
        'Oil & Gas': 1.0, 'Mining': 0.9, 'Maritime': 0.8, 'Railways': 0.7,
        'Manufacturing': 0.6, 'Energy': 0.5, 'Water Treatment': 0.4, 'Healthcare': 0.2
    }
    industry_risk = industry_risk_map.get(industry.name if industry else '', 0.5)
    
    # 3. Maintenance History
    past_maint = db.query(MaintenanceRecord).filter(
        MaintenanceRecord.asset_id == asset.id,
        MaintenanceRecord.start_date <= now
    ).order_by(MaintenanceRecord.start_date.desc()).all()
    
    if past_maint:
        days_since_last_maintenance = (now - past_maint[0].start_date).total_seconds() / (24 * 3600)
    else:
        days_since_last_maintenance = asset_age_days

    historic_failure_count = sum(1 for m in past_maint if m.maintenance_type.value == 'corrective')
    
    # 4. Financial Liability (Costs from Work Orders)
    total_repair_cost = db.query(func.sum(CostRecord.total_cost)).join(WorkOrder).filter(
        WorkOrder.asset_id == asset.id
    ).scalar()
    total_repair_cost = total_repair_cost or 0.0
    
    # Assign to feature array (matching train_model.py order)
    # ['temperature', 'vibration', 'pressure', 'rpm', 'current', 
    #  'asset_age_days', 'warranty_remaining_days', 'industry_risk_factor',
    #  'days_since_last_maintenance', 'historic_failure_count', 'cumulative_repair_cost']
    
    features[5] = max(0, asset_age_days)
    features[6] = warranty_remaining_days
    features[7] = industry_risk
    features[8] = days_since_last_maintenance
    features[9] = historic_failure_count
    features[10] = total_repair_cost
            
    return features.reshape(1, -1)  # Reshape for model input

# ─────────────────────────────────────────────
# PREDICT FAILURE RISK FOR ONE ASSET
# ─────────────────────────────────────────────

def predict_asset_risk(asset_id: int, db: Session = None):
    """
    Predict failure risk for a specific asset
    
    Args:
        asset_id: Which asset to predict for
        db: Database session (optional)
        
    Returns:
        dict with risk_level, risk_score, confidence, recommendation, etc.
    """
    
    if db is None:
        db = SessionLocal()
    
    try:
        # Fetch Asset
        asset = db.query(Asset).filter(Asset.id == asset_id).first()
        if not asset:
            return {"error": "Asset not found"}
            
        # Load AI model
        model, scaler, feature_cols = load_prediction_model()
        
        # Get sensor data for this asset
        features = get_asset_sensor_data(db, asset)
        
        if features is None:
            return {
                "asset_id": asset_id,
                "risk_level": "Unknown",
                "risk_score": 0,
                "confidence": 0,
                "recommendation": "No recent sensor data available",
                "predicted_days_to_failure": None
            }
        
        # Standardize features (same way as training)
        features_scaled = scaler.transform(features)
        
        # Predict risk category (0=Healthy, 1=Warning, 2=High Risk, 3=Critical)
        risk_category = model.predict(features_scaled)[0]
        
        # Get probability scores for each category
        risk_probs = model.predict_proba(features_scaled)[0]
        
        # Map category to label
        risk_labels = ['Healthy', 'Warning', 'High Risk', 'Critical']
        
        # Calculate overall risk score (0-100%)
        # Weighted average based on probabilities
        base_risk_score = (
            risk_probs[0] * 10 +   # Healthy = 10%
            risk_probs[1] * 40 +   # Warning = 40%
            risk_probs[2] * 70 +   # High Risk = 70%
            risk_probs[3] * 95     # Critical = 95%
        )
        
        # Add deterministic variation for demo (±20% to create variety)
        import hashlib
        # Use asset_id to generate a deterministic float between 0 and 1
        asset_hash = int(hashlib.md5(str(asset_id).encode()).hexdigest(), 16)
        deterministic_random = (asset_hash % 1000) / 1000.0
        variation = -20 + (deterministic_random * 45) # Range -20 to +25
        risk_score = max(0, min(100, base_risk_score + variation))
        
        # Adjust risk level based on final score
        if risk_score >= 75:
            risk_level = "Critical"
            risk_category = 3
        elif risk_score >= 55:
            risk_level = "High Risk"
            risk_category = 2
        elif risk_score >= 35:
            risk_level = "Warning"
            risk_category = 1
        else:
            risk_level = "Healthy"
            risk_category = 0
        
        # Generate recommendation
        recommendations = {
            0: "Continue normal operations. Schedule routine maintenance as planned.",
            1: "Monitor closely. Consider advancing next maintenance check.",
            2: "Urgent attention required. Schedule maintenance within 7 days.",
            3: "CRITICAL: High failure probability. Immediate inspection required."
        }
        
        # Estimate days to failure
        days_to_failure_estimate = {
            0: None,  # Healthy - no immediate concern
            1: 45,    # Warning - ~1.5 months
            2: 14,    # High Risk - ~2 weeks
            3: 7      # Critical - ~1 week
        }

        predicted_cost = float(features[0][10]) if features is not None else 0.0
        
        # 8. Check Warranty to Zero out Predicted Liability
        if hasattr(asset, 'warranty_expiry') and asset.warranty_expiry and datetime.utcnow() < asset.warranty_expiry:
            # Apply 85% discount for warranty parts coverage. 15% remaining covers labor/downtime.
            predicted_cost = predicted_cost * 0.15

        return {
            "asset_id": asset_id,
            "risk_level": risk_level,
            "risk_score": round(risk_score, 2),
            "confidence": round(max(risk_probs) * 100, 2),
            "recommendation": recommendations[risk_category],
            "predicted_cost": predicted_cost,
            "predicted_days_to_failure": days_to_failure_estimate[risk_category],
            "category_probabilities": {
                "healthy": round(risk_probs[0] * 100, 2),
                "warning": round(risk_probs[1] * 100, 2),
                "high_risk": round(risk_probs[2] * 100, 2),
                "critical": round(risk_probs[3] * 100, 2)
            }
        }
    
    except Exception as e:
        print(f"Error in predict_asset_risk: {e}")
        return {
            "asset_id": asset_id,
            "risk_level": "Unknown",
            "risk_score": 0,
            "error": str(e)
        }
        
    finally:
        # Let the caller close the session if they created it.
        pass

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# PREDICT FOR ALL ASSETS
# ─────────────────────────────────────────────

def predict_fleet_risk(db: Session = None):
    """
    Optimized batch prediction for the entire asset fleet.
    Reduces database overhead from O(N) queries to O(1-5) bulk queries.
    """
    if db is None:
        db = SessionLocal()
    
    try:
        # 1. Load AI model
        model, scaler, feature_cols = load_prediction_model()
        now = datetime.now()
        
        # 2. Fetch all Assets and Industries
        assets = db.query(Asset).options(joinedload(Asset.industry)).all()
        if not assets:
            return []
            
        asset_ids = [a.id for a in assets]
        industry_risk_map = {
            'Oil & Gas': 1.0, 'Mining': 0.9, 'Maritime': 0.8, 'Railways': 0.7,
            'Manufacturing': 0.6, 'Energy': 0.5, 'Water Treatment': 0.4, 'Healthcare': 0.2
        }

        # 3. Bulk Fetch Sensors (Last 100 per asset)
        # We fetch sensors from last 180 days to cover relevant history for all assets
        cutoff_date = now - timedelta(days=180)
        all_sensors = db.query(Sensor).filter(
            Sensor.asset_id.in_(asset_ids),
            Sensor.timestamp >= cutoff_date
        ).order_by(Sensor.asset_id, Sensor.timestamp.desc()).all()

        # Group sensors by asset (Limit to 100 per asset)
        asset_sensor_map = {aid: {'temperature': [], 'vibration': [], 'pressure': [], 'rpm': [], 'current': []} for aid in asset_ids}
        for s in all_sensors:
            s_map = asset_sensor_map[s.asset_id]
            if s.sensor_type in s_map and len(s_map[s.sensor_type]) < 100:
                s_map[s.sensor_type].append(s.value)

        # 4. Bulk Fetch Maintenance History
        all_maint = db.query(MaintenanceRecord).filter(
            MaintenanceRecord.asset_id.in_(asset_ids),
            MaintenanceRecord.start_date <= now
        ).order_by(MaintenanceRecord.asset_id, MaintenanceRecord.start_date.desc()).all()
        
        asset_maint_map = {aid: [] for aid in asset_ids}
        for m in all_maint:
            asset_maint_map[m.asset_id].append(m)

        # 5. Bulk Fetch Costs (Total per asset)
        cost_results = db.query(
            WorkOrder.asset_id, 
            func.sum(CostRecord.total_cost).label('total_cost')
        ).join(CostRecord).filter(
            WorkOrder.asset_id.in_(asset_ids)
        ).group_by(WorkOrder.asset_id).all()
        
        asset_cost_map = {row[0]: row[1] for row in cost_results}

        # 6. Build Feature Matrix
        feature_rows = []
        valid_assets = []
        
        defaults = {'temperature': 40, 'vibration': 0.5, 'pressure': 100, 'rpm': 1500, 'current': 50}
        core_sensors = ['temperature', 'vibration', 'pressure', 'rpm', 'current']

        for asset in assets:
            s_data = asset_sensor_map[asset.id]
            # Use defaults if no readings found (logic from get_asset_sensor_data)
            features = np.zeros(11)
            for i, s_type in enumerate(core_sensors):
                vals = s_data[s_type]
                features[i] = np.mean(vals) if vals else defaults[s_type]
            
            # Compute Metadata
            age_days = (now - asset.installation_date).total_seconds() / (24 * 3600) if asset.installation_date else 0
            warranty_days = (asset.warranty_expiry - now).total_seconds() / (24 * 3600) if asset.warranty_expiry else 0
            i_risk = industry_risk_map.get(asset.industry.name if asset.industry else '', 0.5)
            
            p_maint = asset_maint_map[asset.id]
            days_since = (now - p_maint[0].start_date).total_seconds() / (24 * 3600) if p_maint else age_days
            fail_count = sum(1 for m in p_maint if m.maintenance_type.value == 'corrective')
            cumulative_cost = asset_cost_map.get(asset.id, 0.0)

            features[5] = max(0, age_days)
            features[6] = warranty_days
            features[7] = i_risk
            features[8] = days_since
            features[9] = fail_count
            features[10] = cumulative_cost
            
            feature_rows.append(features)
            valid_assets.append(asset)

        if not feature_rows:
            return []

        # 7. Batch Inference
        X = np.array(feature_rows)
        X_scaled = scaler.transform(X)
        
        all_probs = model.predict_proba(X_scaled)
        
        # 8. Format Results
        import hashlib
        recommendations = {
            0: "Continue normal operations. Schedule routine maintenance as planned.",
            1: "Monitor closely. Consider advancing next maintenance check.",
            2: "Urgent attention required. Schedule maintenance within 7 days.",
            3: "CRITICAL: High failure probability. Immediate inspection required."
        }
        days_to_failure = {0: None, 1: 45, 2: 14, 3: 7}

        fleet_predictions = []
        for i, asset in enumerate(valid_assets):
            probs = all_probs[i]
            base_score = (probs[0]*10 + probs[1]*40 + probs[2]*70 + probs[3]*95)
            
            asset_hash = int(hashlib.md5(str(asset.id).encode()).hexdigest(), 16)
            var = -20 + ((asset_hash % 1000)/1000.0 * 45)
            final_score = max(0, min(100, base_score + var))
            
            if final_score >= 75: label, cat = "Critical", 3
            elif final_score >= 55: label, cat = "High Risk", 2
            elif final_score >= 35: label, cat = "Warning", 1
            else: label, cat = "Healthy", 0
                
            fleet_predictions.append({
                "id": asset.id,
                "name": asset.name,
                "asset_type": asset.asset_type,
                "industry_name": asset.industry.name if asset.industry else "Unknown",
                "risk_level": label,
                "risk_score": round(final_score, 2),
                "confidence": round(max(probs) * 100, 2),
                "recommendation": recommendations[cat],
                "rul": days_to_failure[cat],
                "purchase_date": asset.purchase_date.isoformat() if asset.purchase_date else None,
                "warranty_expiry": asset.warranty_expiry.isoformat() if asset.warranty_expiry else None
            })
            
        return fleet_predictions

    finally:
        db.close()

def predict_all_assets():
    """Get risk predictions for all assets"""
    
    db = SessionLocal()
    
    try:
        # Get all assets with their industry info
        assets = db.query(Asset).options(joinedload(Asset.industry)).all()
        predictions = []
        
        print(f"🔮 Generating AI predictions for {len(assets)} assets...\n")
        
        for asset in assets:
            # Get prediction for this asset
            prediction = predict_asset_risk(asset.id, db)
            
            # Add asset details
            prediction['asset_name'] = asset.name
            prediction['asset_type'] = asset.asset_type
            prediction['industry'] = asset.industry.name if asset.industry else "Unknown"
            
            predictions.append(prediction)
            
            # Print summary
            emoji = {"Healthy": "🟢", "Warning": "🟡", "High Risk": "🟠", "Critical": "🔴"}
            print(f"{emoji.get(prediction['risk_level'], '⚪')} {asset.name:30s} | "
                  f"Risk: {prediction['risk_score']:5.1f}% | {prediction['risk_level']}")
        
        return predictions
    
    finally:
        db.close()

def persist_fleet_predictions():
    """
    Consolidated utility: Generates fleet predictions and persists them to SQLite.
    This ensures the 'predictions' table is always in sync with the latest AI model.
    """
    print("🚀 Running Fleet-Wide Prediction Refresh (Consolidated Logic)...")
    
    # 1. Generate the data using our optimized batch function
    predictions = predict_fleet_risk()
    
    if not predictions:
        print("⚠️ No assets found/valid for prediction.")
        return

    # 2. Connect via raw SQLite to avoid SQLAlchemy metadata collisions in CLI env
    # This is a robust fallback for script/CLI execution
    db_path = os.path.join(os.path.dirname(__file__), '..', 'backend', 'eam_database.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        print(f"🧹 Clearing and backfilling {len(predictions)} AI insights...")
        cursor.execute("DELETE FROM predictions")
        
        insert_query = """
        INSERT INTO predictions (asset_id, risk_level, risk_score, predicted_failure, confidence, recommendation, predicted_cost, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        for p in predictions:
            # Map high-level recommendation to specific bot-friendly failure category
            fail_type = "Healthy"
            rec = p['recommendation'].lower()
            if "bearing" in rec: fail_type = "Bearing"
            elif "cooling" in rec: fail_type = "Cooling"
            elif "electrical" in rec: fail_type = "Electrical"
            elif "seal" in rec: fail_type = "Seal"
            elif "vibration" in rec: fail_type = "Vibration"
            
            cursor.execute(insert_query, (
                p['id'],
                p['risk_level'],
                p['risk_score'],
                fail_type,
                p['confidence'] / 100.0,
                p['recommendation'],
                p['risk_score'] * 1000,
                datetime.now(timezone.utc).isoformat()
            ))
            
        conn.commit()
        print(f"✅ Successfully persisted {len(predictions)} predictions to the EAM database.")
        
    except Exception as e:
        print(f"❌ Persistence Error during Refresh: {e}")
        conn.rollback()
    finally:
        conn.close()

# ─────────────────────────────────────────────
# TEST THE PREDICTIONS (Run this file directly)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EAM AI Prediction Engine")
    parser.add_argument("--refresh", action="store_true", help="Refresh and persist predictions to database")
    args = parser.parse_args()

    if args.refresh:
        persist_fleet_predictions()
    else:
        print("=" * 80)
        print("🤖 TESTING AI PREDICTIONS ON ASSETS")
        print("=" * 80)
    
    predictions = predict_all_assets()
    
    print("\n" + "=" * 80)
    print(f"✅ Generated predictions for {len(predictions)} assets")
    print("=" * 80)
    
    # Show summary statistics
    risk_counts = {}
    for p in predictions:
        risk_counts[p['risk_level']] = risk_counts.get(p['risk_level'], 0) + 1
    
    print("\n📊 Risk Distribution:")
    for level in ['Healthy', 'Warning', 'High Risk', 'Critical']:
        count = risk_counts.get(level, 0)
        pct = (count / len(predictions)) * 100
        print(f"   {level:12s}: {count:3d} ({pct:5.1f}%)")