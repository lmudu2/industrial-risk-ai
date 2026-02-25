import sys
import os
import time
import random
from datetime import datetime

# Allow importing from backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database import SessionLocal
from models import Asset, Sensor

# Configuration for live simulation
SIMULATION_TICK_RATE = 5          # generate new reading every N seconds
DEGRADATION_CHANCE = 0.05         # 5% chance an asset starts degrading
MAX_DEGRADING_ASSETS = 3          # Maximum number of simultaneously degrading assets

print("=" * 80)
print("📡 STARTING LIVE IoT TELEMETRY SIMULATOR")
print("=" * 80)
print(f"Update interval: {SIMULATION_TICK_RATE} seconds")
print(f"Database: eam_database.db")
print("Press Ctrl+C to stop the simulator.")
print("-" * 80)

# Track which assets are intentionally degrading for the demo
degrading_assets = {}

def get_latest_sensor_reading(db, asset_id, sensor_type):
    """Fetch the most recent reading for a given sensor and asset"""
    return db.query(Sensor).filter(
        Sensor.asset_id == asset_id,
        Sensor.sensor_type == sensor_type
    ).order_by(Sensor.timestamp.desc()).first()

def simulate_reading(db, asset, last_sensors, degrading_assets):
    """Generate a single new reading for all sensors on an asset"""
    
    # 5 Core generic sensors
    sensor_types = ['temperature', 'vibration', 'pressure', 'rpm', 'current']
    
    # Default boundaries if a sensor has no history
    defaults = {
        'temperature': {'val': 40, 'min': 20, 'max': 80},
        'vibration':   {'val': 0.5, 'min': 0, 'max': 2.0},
        'pressure':    {'val': 100, 'min': 80, 'max': 150},
        'rpm':         {'val': 1500, 'min': 1000, 'max': 3000},
        'current':     {'val': 50,  'min': 10, 'max': 100}
    }
    
    for s_type in sensor_types:
        last_sensor = last_sensors.get(s_type)
        
        if last_sensor:
            base_value = last_sensor.value
            min_thresh = last_sensor.min_threshold
            max_thresh = last_sensor.max_threshold
        else:
            base_value = defaults[s_type]['val']
            min_thresh = defaults[s_type]['min']
            max_thresh = defaults[s_type]['max']
        
        # 1. Natural IoT Jitter (Sensors always bounce around randomly by ~1%)
        noise = random.uniform(0.99, 1.01)
        new_value = base_value * noise
        
        # 2. Inject Intentional Degradation (for the Demo)
        if asset.id in degrading_assets:
            degrad_type = degrading_assets[asset.id]
            
            # Make the new value violently trend towards failure
            if degrad_type == "vibration_spike" and s_type == "vibration":
                new_value += random.uniform(0.05, 0.1)  # Increase vibration aggressively
            elif degrad_type == "overheat" and s_type == "temperature":
                new_value += random.uniform(1.0, 3.0)   # Increase temp quickly
            elif degrad_type == "pressure_drop" and s_type == "pressure":
                new_value -= random.uniform(2.0, 5.0)   # Drop pressure
        
        # Keep healthy sensors somewhat mean-reverting so they don't randomly drift into failure over time
        elif not asset.id in degrading_assets:
            # If a base reading wanders too high/low from normal, nudge it back
            midpoint = (min_thresh + max_thresh) / 2
            if new_value > (midpoint * 1.2):
                new_value *= 0.98 # nudge down
            elif new_value < (midpoint * 0.8):
                new_value *= 1.02 # nudge up
                
        # Insert the new row as the "Present" moment
        new_sensor = Sensor(
            asset_id=asset.id,
            sensor_type=s_type,
            unit=last_sensor.unit if last_sensor else "unit",
            value=round(new_value, 2),
            min_threshold=min_thresh,
            max_threshold=max_thresh,
            timestamp=datetime.now()
        )
        db.add(new_sensor)

def run_simulation():
    db = SessionLocal()
    try:
        assets = db.query(Asset).all()
        if not assets:
            print("❌ No assets found in the database. Run generate_data.py first.")
            return

        tick = 0
        while True:
            start_time = time.time()
            tick += 1
            
            # Randomly pick an asset to start failing (if we haven't hit the max)
            if len(degrading_assets) < MAX_DEGRADING_ASSETS and random.random() < DEGRADATION_CHANCE:
                unlucky_asset = random.choice(assets)
                if unlucky_asset.id not in degrading_assets:
                    failure_mode = random.choice(["vibration_spike", "overheat", "pressure_drop"])
                    degrading_assets[unlucky_asset.id] = failure_mode
                    print(f"\n⚠️  ALERT: {unlucky_asset.name} (ID: {unlucky_asset.id}) has begun degrading ({failure_mode})!")

            # Generate reading for every asset
            for asset in assets:
                # Get the last readings for this asset efficiently
                latest = db.query(Sensor).filter(Sensor.asset_id == asset.id).order_by(Sensor.timestamp.desc()).limit(5).all()
                last_sensors = {s.sensor_type: s for s in latest}
                
                simulate_reading(db, asset, last_sensors, degrading_assets)
                
            # Commit the whole batch simultaneously
            db.commit()
            
            # Console output
            calc_time = time.time() - start_time
            print(f"[Tick {tick}] Pushed {len(assets)*5} live readings. ({len(degrading_assets)} assets currently degrading) - Calc time: {calc_time:.2f}s", end="\r")
            
            # Sleep until the next tick
            sleep_time = max(0, SIMULATION_TICK_RATE - calc_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n🛑 Simulator stopped by user.")
    finally:
        db.close()

if __name__ == "__main__":
    run_simulation()
