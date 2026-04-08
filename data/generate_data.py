import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from sqlalchemy.orm import Session
from database import SessionLocal, engine, Base
from models import (
    Industry, Asset, Sensor, 
    MaintenanceRecord, WorkOrder, CostRecord,
    AssetStatus, MaintenanceType, WorkOrderStatus, Priority
)
from faker import Faker
from datetime import datetime, timedelta
import random
import numpy as np

fake = Faker()

# ═════════════════════════════════════════════════════════════
# CONFIGURATION - ML-SCALE DATA
# ═════════════════════════════════════════════════════════════

industries_data = [
    {"name": "Manufacturing", "description": "Automotive, Steel, Chemical, Food & Beverage"},
    {"name": "Oil & Gas", "description": "Drilling rigs, offshore platforms, pipeline pumping"},
    {"name": "Renewable Energy", "description": "Wind turbines, solar trackers"},
    {"name": "Power Generation", "description": "Steam/gas turbines, generators, transformers"},
    {"name": "Water Treatment", "description": "Centrifugal pumps, filtration, valve actuators"},
    {"name": "Aerospace", "description": "Jet engines, landing gear, auxiliary power units"},
    {"name": "Railways", "description": "Locomotive engines, wheelsets, signaling equipment"},
    {"name": "Maritime", "description": "Marine diesel engines, propulsion, cargo cranes"},
    {"name": "Mining", "description": "Excavators, haul trucks, crushers, ventilation fans"},
    {"name": "Healthcare", "description": "MRI/CT scanners, centrifuges, cleanroom HVAC"},
]

asset_types = {
    "Manufacturing": ["Robotic Arm", "Conveyor Belt", "CNC Machine", "Paint Booth Fan", "Extruder", "Injection Molder"],
    "Oil & Gas": ["Drilling Rig", "Pipeline Pump", "Offshore Platform", "Compressor", "Separator"],
    "Renewable Energy": ["Wind Turbine", "Solar Tracker", "Gearbox", "Inverter"],
    "Power Generation": ["Steam Turbine", "Gas Turbine", "Generator", "Transformer", "Boiler"],
    "Water Treatment": ["Centrifugal Pump", "Filtration System", "Valve Actuator", "Chlorination Unit"],
    "Aerospace": ["Jet Engine", "Landing Gear", "Auxiliary Power Unit", "Hydraulic System"],
    "Railways": ["Locomotive Engine", "Wheelset", "Signaling Equipment", "Braking System"],
    "Maritime": ["Marine Diesel Engine", "Propulsion System", "Cargo Crane", "Navigation System"],
    "Mining": ["Excavator", "Haul Truck", "Crusher", "Ventilation Fan", "Drill Machine"],
    "Healthcare": ["MRI Scanner", "CT Scanner", "Centrifuge", "Tablet Press", "Cleanroom HVAC"],
}

industry_frequencies = {
    "Aerospace": 1,          # Every 1 hour
    "Power Generation": 2,   # Every 2 hours
    "Maritime": 4,           # Every 4 hours
    "Oil & Gas": 4,          # Every 4 hours
    "Manufacturing": 8,      # Every 8 hours
    "Mining": 8,             # Every 8 hours
    "Renewable Energy": 12,  # Every 12 hours
    "Water Treatment": 12,   # Every 12 hours
    "Railways": 12,          # Every 12 hours
    "Healthcare": 24,        # Every 24 hours
}

# ═════════════════════════════════════════════════════════════
# REAL-WORLD PURCHASE COST RANGES (USD) — sourced from industry data
# ═════════════════════════════════════════════════════════════

purchase_cost_ranges = {
    # Manufacturing
    "Robotic Arm":         (25_000,    350_000),
    "Conveyor Belt":       (5_000,     50_000),
    "CNC Machine":         (50_000,    500_000),
    "Paint Booth Fan":     (3_000,     15_000),
    "Extruder":            (20_000,    200_000),
    "Injection Molder":    (50_000,    500_000),
    # Oil & Gas
    "Drilling Rig":        (20_000_000, 100_000_000),
    "Pipeline Pump":       (150_000,   1_500_000),
    "Offshore Platform":   (50_000_000, 200_000_000),
    "Compressor":          (100_000,   1_200_000),
    "Separator":           (80_000,    500_000),
    # Renewable Energy
    "Wind Turbine":        (1_000_000, 4_000_000),
    "Solar Tracker":       (50_000,    150_000),
    "Gearbox":             (200_000,   800_000),
    "Inverter":            (5_000,     50_000),
    # Power Generation
    "Steam Turbine":       (3_000_000, 15_000_000),
    "Gas Turbine":         (5_000_000, 20_000_000),
    "Generator":           (40_000,    3_000_000),
    "Transformer":         (50_000,    1_000_000),
    "Boiler":              (30_000,    500_000),
    # Water Treatment
    "Centrifugal Pump":    (5_000,     150_000),
    "Filtration System":   (50_000,    500_000),
    "Valve Actuator":      (5_000,     50_000),
    "Chlorination Unit":   (3_000,     25_000),
    # Aerospace
    "Jet Engine":          (8_000_000, 25_000_000),
    "Landing Gear":        (1_000_000, 4_000_000),
    "Auxiliary Power Unit": (400_000,  1_500_000),
    "Hydraulic System":    (150_000,   600_000),
    # Railways
    "Locomotive Engine":   (2_000_000, 6_000_000),
    "Wheelset":            (50_000,    200_000),
    "Signaling Equipment": (100_000,   500_000),
    "Braking System":      (50_000,    200_000),
    # Maritime
    "Marine Diesel Engine": (2_000_000, 10_000_000),
    "Propulsion System":   (1_000_000, 5_000_000),
    "Cargo Crane":         (500_000,   2_500_000),
    "Navigation System":   (10_000,    100_000),
    # Mining
    "Excavator":           (300_000,   3_000_000),
    "Haul Truck":          (2_000_000, 7_000_000),
    "Crusher":             (100_000,   2_000_000),
    "Ventilation Fan":     (5_000,     30_000),
    "Drill Machine":       (50_000,    735_000),
    # Healthcare
    "MRI Scanner":         (1_000_000, 3_000_000),
    "CT Scanner":          (150_000,   2_000_000),
    "Centrifuge":          (5_000,     60_000),
    "Tablet Press":        (20_000,    200_000),
    "Cleanroom HVAC":      (25_000,    100_000),
}

# ═════════════════════════════════════════════════════════════
# REAL-WORLD REPAIR COST RANGES (USD) by priority level
# Format: {asset_type: {priority: (min_cost, max_cost)}}
# ═════════════════════════════════════════════════════════════

repair_cost_ranges = {
    # ── Aerospace ────────────────────────────────────────────
    "Jet Engine": {
        "low":      (50_000,   150_000),    # routine inspection, minor part swap
        "medium":   (200_000,  800_000),    # hot section refurbishment
        "high":     (500_000,  1_500_000),  # major overhaul
        "critical": (1_000_000, 3_000_000), # full engine overhaul / replacement
    },
    "Landing Gear": {
        "low":      (20_000,   80_000),
        "medium":   (80_000,   200_000),
        "high":     (200_000,  500_000),
        "critical": (400_000,  750_000),
    },
    "Auxiliary Power Unit": {
        "low":      (10_000,   40_000),
        "medium":   (40_000,   120_000),
        "high":     (100_000,  300_000),
        "critical": (200_000,  500_000),
    },
    "Hydraulic System": {
        "low":      (5_000,    20_000),
        "medium":   (20_000,   80_000),
        "high":     (60_000,   200_000),
        "critical": (150_000,  400_000),
    },
    # ── Manufacturing ────────────────────────────────────────
    "Robotic Arm": {
        "low":      (500,      3_000),
        "medium":   (3_000,    15_000),
        "high":     (10_000,   40_000),
        "critical": (25_000,   80_000),
    },
    "CNC Machine": {
        "low":      (500,      2_000),
        "medium":   (2_000,    8_000),
        "high":     (5_000,    20_000),
        "critical": (10_000,   40_000),
    },
    "Conveyor Belt": {
        "low":      (200,      1_000),
        "medium":   (1_000,    5_000),
        "high":     (3_000,    10_000),
        "critical": (6_000,    20_000),
    },
    "Paint Booth Fan": {
        "low":      (200,      800),
        "medium":   (500,      2_000),
        "high":     (1_500,    5_000),
        "critical": (3_000,    10_000),
    },
    "Extruder": {
        "low":      (500,      3_000),
        "medium":   (2_000,    10_000),
        "high":     (8_000,    30_000),
        "critical": (20_000,   60_000),
    },
    "Injection Molder": {
        "low":      (500,      3_000),
        "medium":   (2_000,    10_000),
        "high":     (8_000,    30_000),
        "critical": (20_000,   75_000),
    },
    # ── Oil & Gas ────────────────────────────────────────────
    "Drilling Rig": {
        "low":      (50_000,   200_000),
        "medium":   (200_000,  500_000),
        "high":     (500_000,  2_000_000),
        "critical": (1_000_000, 5_000_000),
    },
    "Pipeline Pump": {
        "low":      (5_000,    20_000),
        "medium":   (15_000,   60_000),
        "high":     (40_000,   150_000),
        "critical": (100_000,  400_000),
    },
    "Offshore Platform": {
        "low":      (100_000,  500_000),
        "medium":   (500_000,  2_000_000),
        "high":     (2_000_000, 10_000_000),
        "critical": (5_000_000, 20_000_000),
    },
    "Compressor": {
        "low":      (5_000,    20_000),
        "medium":   (15_000,   60_000),
        "high":     (40_000,   150_000),
        "critical": (100_000,  300_000),
    },
    "Separator": {
        "low":      (3_000,    15_000),
        "medium":   (10_000,   40_000),
        "high":     (30_000,   100_000),
        "critical": (60_000,   200_000),
    },
    # ── Renewable Energy ─────────────────────────────────────
    "Wind Turbine": {
        "low":      (5_000,    15_000),
        "medium":   (15_000,   50_000),
        "high":     (30_000,   100_000),
        "critical": (80_000,   250_000),    # gearbox replacement
    },
    "Solar Tracker": {
        "low":      (200,      1_000),
        "medium":   (800,      3_000),
        "high":     (2_000,    8_000),
        "critical": (5_000,    15_000),
    },
    "Gearbox": {
        "low":      (5_000,    20_000),
        "medium":   (15_000,   50_000),
        "high":     (30_000,   100_000),
        "critical": (80_000,   200_000),
    },
    "Inverter": {
        "low":      (200,      500),
        "medium":   (500,      2_000),
        "high":     (1_500,    5_000),
        "critical": (3_000,    10_000),
    },
    # ── Power Generation ─────────────────────────────────────
    "Steam Turbine": {
        "low":      (50_000,   200_000),
        "medium":   (150_000,  500_000),
        "high":     (400_000,  1_500_000),
        "critical": (1_000_000, 5_000_000),
    },
    "Gas Turbine": {
        "low":      (100_000,  300_000),
        "medium":   (200_000,  800_000),
        "high":     (500_000,  2_000_000),
        "critical": (1_500_000, 8_000_000),
    },
    "Generator": {
        "low":      (2_000,    10_000),
        "medium":   (8_000,    30_000),
        "high":     (20_000,   80_000),
        "critical": (50_000,   200_000),
    },
    "Transformer": {
        "low":      (1_000,    5_000),
        "medium":   (3_000,    15_000),
        "high":     (10_000,   50_000),
        "critical": (30_000,   100_000),
    },
    "Boiler": {
        "low":      (2_000,    10_000),
        "medium":   (8_000,    30_000),
        "high":     (20_000,   80_000),
        "critical": (50_000,   150_000),
    },
    # ── Water Treatment ──────────────────────────────────────
    "Centrifugal Pump": {
        "low":      (300,      1_500),
        "medium":   (1_000,    5_000),
        "high":     (3_000,    15_000),
        "critical": (8_000,    30_000),
    },
    "Filtration System": {
        "low":      (1_000,    5_000),
        "medium":   (3_000,    15_000),
        "high":     (10_000,   50_000),
        "critical": (30_000,   100_000),
    },
    "Valve Actuator": {
        "low":      (150,      500),
        "medium":   (400,      2_000),
        "high":     (1_500,    5_000),
        "critical": (3_000,    10_000),
    },
    "Chlorination Unit": {
        "low":      (200,      800),
        "medium":   (500,      2_500),
        "high":     (1_500,    5_000),
        "critical": (3_000,    8_000),
    },
    # ── Railways ─────────────────────────────────────────────
    "Locomotive Engine": {
        "low":      (20_000,   80_000),
        "medium":   (50_000,   200_000),
        "high":     (150_000,  500_000),
        "critical": (300_000,  1_000_000),
    },
    "Wheelset": {
        "low":      (2_000,    8_000),
        "medium":   (5_000,    20_000),
        "high":     (15_000,   50_000),
        "critical": (30_000,   80_000),
    },
    "Signaling Equipment": {
        "low":      (1_000,    5_000),
        "medium":   (3_000,    15_000),
        "high":     (10_000,   40_000),
        "critical": (25_000,   100_000),
    },
    "Braking System": {
        "low":      (1_000,    5_000),
        "medium":   (3_000,    15_000),
        "high":     (10_000,   40_000),
        "critical": (25_000,   80_000),
    },
    # ── Maritime ─────────────────────────────────────────────
    "Marine Diesel Engine": {
        "low":      (20_000,   80_000),
        "medium":   (60_000,   200_000),
        "high":     (150_000,  500_000),
        "critical": (300_000,  1_500_000),
    },
    "Propulsion System": {
        "low":      (15_000,   50_000),
        "medium":   (40_000,   150_000),
        "high":     (100_000,  400_000),
        "critical": (250_000,  800_000),
    },
    "Cargo Crane": {
        "low":      (5_000,    20_000),
        "medium":   (15_000,   60_000),
        "high":     (40_000,   150_000),
        "critical": (100_000,  400_000),
    },
    "Navigation System": {
        "low":      (200,      1_000),
        "medium":   (800,      3_000),
        "high":     (2_000,    8_000),
        "critical": (5_000,    15_000),
    },
    # ── Mining ───────────────────────────────────────────────
    "Excavator": {
        "low":      (10_000,   40_000),
        "medium":   (30_000,   100_000),
        "high":     (80_000,   300_000),
        "critical": (200_000,  600_000),
    },
    "Haul Truck": {
        "low":      (15_000,   50_000),
        "medium":   (40_000,   150_000),
        "high":     (100_000,  400_000),
        "critical": (250_000,  800_000),
    },
    "Crusher": {
        "low":      (5_000,    20_000),
        "medium":   (15_000,   60_000),
        "high":     (40_000,   150_000),
        "critical": (100_000,  400_000),
    },
    "Ventilation Fan": {
        "low":      (500,      2_000),
        "medium":   (1_500,    5_000),
        "high":     (3_000,    10_000),
        "critical": (8_000,    25_000),
    },
    "Drill Machine": {
        "low":      (2_000,    8_000),
        "medium":   (5_000,    20_000),
        "high":     (15_000,   50_000),
        "critical": (30_000,   100_000),
    },
    # ── Healthcare ───────────────────────────────────────────
    "MRI Scanner": {
        "low":      (5_000,    20_000),
        "medium":   (20_000,   60_000),
        "high":     (50_000,   120_000),
        "critical": (100_000,  200_000),
    },
    "CT Scanner": {
        "low":      (3_000,    15_000),
        "medium":   (10_000,   40_000),
        "high":     (30_000,   80_000),
        "critical": (60_000,   150_000),
    },
    "Centrifuge": {
        "low":      (200,      800),
        "medium":   (500,      3_000),
        "high":     (2_000,    8_000),
        "critical": (5_000,    20_000),
    },
    "Tablet Press": {
        "low":      (500,      2_000),
        "medium":   (1_500,    8_000),
        "high":     (5_000,    20_000),
        "critical": (15_000,   50_000),
    },
    "Cleanroom HVAC": {
        "low":      (500,      2_000),
        "medium":   (1_500,    8_000),
        "high":     (5_000,    20_000),
        "critical": (15_000,   40_000),
    },
}

sensor_configs = [
    {"type": "temperature", "unit": "°C",   "min": 20,  "max": 80,   "critical": 100},
    {"type": "vibration",   "unit": "mm/s", "min": 0.1, "max": 1.0,  "critical": 2.0},
    {"type": "pressure",    "unit": "PSI",  "min": 50,  "max": 200,  "critical": 250},
    {"type": "rpm",         "unit": "RPM",  "min": 800, "max": 3000, "critical": 3500},
    {"type": "current",     "unit": "Amps", "min": 10,  "max": 100,  "critical": 120},
]

# ═════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═════════════════════════════════════════════════════════════

def get_sensor_readings_for_period(db, asset_id, start_date, end_date):
    """Get average sensor readings for a time period"""
    
    sensors = db.query(Sensor).filter(
        Sensor.asset_id == asset_id,
        Sensor.timestamp >= start_date,
        Sensor.timestamp <= end_date
    ).all()
    
    if not sensors:
        return None
    
    # Group by sensor type
    by_type = {}
    for s in sensors:
        if s.sensor_type not in by_type:
            by_type[s.sensor_type] = []
        by_type[s.sensor_type].append(s.value)
    
    # Calculate averages
    averages = {
        'temperature': np.mean(by_type.get('temperature', [70])),
        'vibration': np.mean(by_type.get('vibration', [0.5])),
        'pressure': np.mean(by_type.get('pressure', [150])),
        'rpm': np.mean(by_type.get('rpm', [2000])),
        'current': np.mean(by_type.get('current', [50])),
    }
    
    return averages

def determine_component_failure(sensor_averages):
    """
    CORRELATION LOGIC: Determine which component failed based on sensor patterns
    
    This creates realistic correlations the ML model can learn!
    """
    
    temp = sensor_averages['temperature']
    vib = sensor_averages['vibration']
    curr = sensor_averages['current']
    pressure = sensor_averages['pressure']
    rpm = sensor_averages['rpm']
    
    # High vibration → Bearing failure
    if vib > 1.1:  # Lowered from 1.2
        return {
            'component': 'Bearing',
            'description': 'Bearing replacement due to excessive wear and vibration',
            'type': MaintenanceType.corrective
        }
    
    # High temperature + normal vibration → Cooling issue
    elif temp > 82 and vib < 1.0:  # Lowered from 85
        return {
            'component': 'Cooling',
            'description': 'Cooling system repair and fluid replacement',
            'type': MaintenanceType.corrective
        }
    
    # High current → Electrical/Motor issue
    elif curr > 85:  # Lowered from 90
        return {
            'component': 'Electrical',
            'description': 'Electrical fault diagnosis and motor repair',
            'type': MaintenanceType.corrective
        }
    
    # High pressure → Seal/Valve issue
    elif pressure > 210:  # Lowered from 220
        return {
            'component': 'Seal',
            'description': 'Seal replacement and valve adjustment',
            'type': MaintenanceType.corrective
        }
    
    # Multiple issues → General overhaul
    elif vib > 0.8 and temp > 78:  # Lowered from 0.9/80
        return {
            'component': 'Multiple',
            'description': 'Full overhaul and component replacement',
            'type': MaintenanceType.corrective
        }


    
    # Normal readings → Preventive maintenance
    else:
        return {
            'component': 'Preventive',
            'description': random.choice([
                'Routine inspection and lubrication',
                'Sensor calibration and testing',
                'Filter cleaning and fluid top-up',
                'Scheduled preventive maintenance check'
            ]),
            'type': MaintenanceType.preventive
        }

# ═════════════════════════════════════════════════════════════
# MAIN DATA GENERATION
# ═════════════════════════════════════════════════════════════

def generate_data(num_assets_per_industry: int = 20):
    
    db: Session = SessionLocal()
    Base.metadata.create_all(bind=engine)

    print("=" * 80)
    print("🚀 GENERATING ML-SCALE DATA WITH REALISTIC CORRELATIONS")
    print("=" * 80)
    print("\n📊 Target Scale:")
    print("   - 6 months of variable-frequency sensor history (hourly up to daily)")
    print("   - 20-40 maintenance events per asset")
    print("   - ~200,000 total sensor readings")
    print("   - ~3,000 maintenance records")
    print("   - Realistic sensor→component correlations")
    print("\n⏱️  Estimated time: 3-5 minutes\n")

    # ─────────────────────────────────────────────
    # STEP 1: CREATE INDUSTRIES
    # ─────────────────────────────────────────────
    
    print("📊 Step 1: Creating Industries...")
    industry_objects = []
    
    for ind in industries_data:
        # Check if industry already exists
        industry = db.query(Industry).filter(Industry.name == ind["name"]).first()
        if not industry:
            industry = Industry(name=ind["name"], description=ind["description"])
            db.add(industry)
            db.flush() # Ensure ID is populated
        
        industry_objects.append(industry)
    
    db.commit()
    print(f"✅ Resolved {len(industry_objects)} industries")

    # ─────────────────────────────────────────────
    # STEP 2: CREATE ASSETS
    # ─────────────────────────────────────────────
    
    print("\n🏭 Step 2: Creating Assets...")
    asset_objects = []
    
    for industry in industry_objects:
        types = asset_types[industry.name]
        
        for i in range(num_assets_per_industry):
            asset_type = random.choice(types)
            purchase_date = fake.date_time_between(start_date="-10y", end_date="-3y")
            # Real-world purchase cost from researched ranges
            cost_range = purchase_cost_ranges.get(asset_type, (50_000, 300_000))
            purchase_cost = round(random.uniform(cost_range[0], cost_range[1]), 2)

            asset = Asset(
                name=f"{asset_type} #{random.randint(100, 999)}",
                asset_type=asset_type,
                location=f"Site {random.choice(['A','B','C','D'])} - Zone {random.randint(1,5)}",
                manufacturer=fake.company(),
                model_number=fake.bothify(text="MDL-????-###"),
                serial_number=fake.unique.bothify(text="SN-########"),
                purchase_date=purchase_date,
                installation_date=purchase_date + timedelta(days=random.randint(10, 60)),
                warranty_expiry=purchase_date + timedelta(days=random.randint(365, 1825)),
                status=random.choices(
                    list(AssetStatus),
                    weights=[70, 15, 10, 5]
                )[0],
                purchase_cost=purchase_cost,
                current_value=round(purchase_cost * random.uniform(0.3, 0.9), 2),
                industry_id=industry.id
            )
            db.add(asset)
            asset_objects.append(asset)

    db.commit()
    print(f"✅ Created {len(asset_objects)} assets")

    # Mapping industry_id to its generation frequency
    industry_id_to_freq = {ind.id: industry_frequencies[ind.name] for ind in industry_objects}

    # ─────────────────────────────────────────────
    # STEP 3: CREATE SENSOR DATA (6 MONTHS)
    # WITH REALISTIC DEGRADATION CYCLES
    # ─────────────────────────────────────────────
    
    print("\n🌡️  Step 3: Generating 6 Months of Sensor Data...")
    print("   (This takes ~1-2 minutes...)")
    
    sensor_count = 0
    DAYS_OF_HISTORY = 180  # 6 months
    HOURS_OF_HISTORY = DAYS_OF_HISTORY * 24
    
    for asset_idx, asset in enumerate(asset_objects):
        # Each asset gets 2-4 sensors
        num_sensors = random.randint(2, 4)
        selected_sensors = random.sample(sensor_configs, num_sensors)

        freq_hours = industry_id_to_freq.get(asset.industry_id, 12)

        for s in selected_sensors:
            # Generate readings with multiple degradation cycles
            # (simulates multiple maintenance cycles over 6 months)
            
            current_hour = 0
            while current_hour < HOURS_OF_HISTORY:
                # Each cycle: healthy → degrading → maintenance → reset (1-2 months)
                cycle_length_hours = random.randint(30*24, 60*24)
                
                for hour_in_cycle in range(0, min(cycle_length_hours, HOURS_OF_HISTORY - current_hour), freq_hours):
                    timestamp = datetime.now() - timedelta(hours=(HOURS_OF_HISTORY - current_hour))
                    
                    # Degradation within cycle
                    degradation_factor = 1 + (hour_in_cycle / cycle_length_hours) * 0.4
                    
                    # Add some noise
                    noise = random.uniform(0.95, 1.05)
                    
                    # Occasional anomalies (adjusted for frequency so high-freq doesn't get flooded)
                    base_anomaly_prob = 0.25
                    adjusted_anomaly_prob = base_anomaly_prob * (freq_hours / 24)
                    is_anomaly = random.random() < adjusted_anomaly_prob
                    
                    if is_anomaly:
                        value = random.uniform(s["max"], s["critical"]) * degradation_factor
                    else:
                        base_value = random.uniform(s["min"], s["max"])
                        value = base_value * degradation_factor * noise
                    
                    value = round(value, 2)

                    sensor = Sensor(
                        asset_id=asset.id,
                        sensor_type=s["type"],
                        unit=s["unit"],
                        value=value,
                        min_threshold=s["min"],
                        max_threshold=s["max"],
                        timestamp=timestamp
                    )
                    db.add(sensor)
                    sensor_count += 1
                    current_hour += freq_hours
                    
                    # Commit in batches
                    if sensor_count % 5000 == 0:
                        db.commit()
                        print(f"   Progress: {sensor_count:,} sensor readings created...")
                
                # After cycle ends (maintenance happens), reset degradation
                current_hour += freq_hours

        # Progress update per asset
        if (asset_idx + 1) % 10 == 0:
            db.commit()
            print(f"   Completed {asset_idx + 1}/{len(asset_objects)} assets...")

    db.commit()
    print(f"✅ Created {sensor_count:,} sensor readings with degradation cycles")

    # ─────────────────────────────────────────────
    # STEP 4: CREATE MAINTENANCE RECORDS
    # BASED ON SENSOR PATTERNS (CORRELATION!)
    # ─────────────────────────────────────────────
    
    print("\n🔧 Step 4: Creating Maintenance Records with Correlations...")
    maintenance_count = 0
    
    for asset in asset_objects:
        # Each asset has 20-40 maintenance events over 6 months
        num_records = random.randint(20, 40)
        
        # Spread maintenance events across 6 months
        maintenance_dates = []
        start_date = datetime.now() - timedelta(days=180)
        
        for i in range(num_records):
            days_offset = (180 / num_records) * i + random.randint(-5, 5)
            maintenance_date = start_date + timedelta(days=days_offset)
            maintenance_dates.append(maintenance_date)
        
        for maint_date in maintenance_dates:
            # Look at sensor readings 7 days BEFORE maintenance
            lookback_start = maint_date - timedelta(days=7)
            sensor_avg = get_sensor_readings_for_period(db, asset.id, lookback_start, maint_date)
            
            if sensor_avg:
                # Determine component failure based on sensor patterns
                failure_info = determine_component_failure(sensor_avg)
            else:
                # Fallback if no sensor data
                failure_info = {
                    'component': 'Preventive',
                    'description': 'Scheduled preventive maintenance',
                    'type': MaintenanceType.preventive
                }
            
            duration = random.randint(1, 48)
            
            record = MaintenanceRecord(
                asset_id=asset.id,
                maintenance_type=failure_info['type'],
                description=failure_info['description'],
                performed_by=fake.name(),
                start_date=maint_date,
                end_date=maint_date + timedelta(hours=duration),
                downtime_hours=round(duration * random.uniform(0.5, 1.0), 1),
                outcome=random.choices(
                    ["Resolved", "Partially Fixed", "Escalated", "No Issue Found"],
                    weights=[70, 15, 10, 5]
                )[0]
            )
            db.add(record)
            maintenance_count += 1
        
        if maintenance_count % 500 == 0:
            db.commit()
            print(f"   Progress: {maintenance_count:,} maintenance records created...")

    db.commit()
    print(f"✅ Created {maintenance_count:,} maintenance records with sensor correlations")

    # ─────────────────────────────────────────────
    # STEP 5: CREATE WORK ORDERS
    # ─────────────────────────────────────────────
    
    print("\n📋 Step 5: Creating Work Orders...")
    work_order_objects = []
    
    for asset in asset_objects:
        # 15-30 work orders per asset
        num_orders = random.randint(15, 30)
        
        for _ in range(num_orders):
            created = fake.date_time_between(start_date="-180d", end_date="now")
            status = random.choices(
                list(WorkOrderStatus),
                weights=[20, 30, 45, 5]
            )[0]


            # Define weighted titles for more realistic data distribution
            title_options = [
                "Replace worn bearing",
                "Investigate abnormal vibration",
                "Scheduled maintenance service",
                "Emergency repair",
                "Sensor replacement",
                "Cooling system repair",
                "Electrical system check",
                "AI alert follow-up inspection"
            ]
            title_weights = [15, 10, 30, 5, 10, 10, 10, 10]
            
            wo_title = random.choices(title_options, weights=title_weights)[0]
            
            wo_descriptions = {
                "Replace worn bearing": "Bearing displaying signs of pitting and wear. Requires standard replacement.",
                "Investigate abnormal vibration": "Vibration levels exceeding 1.2mm/s. Check alignment and mounting bolts.",
                "Scheduled maintenance service": "Quarterly preventive maintenance checklist execution.",
                "Emergency repair": "Unplanned stoppage reported. Immediate technician response required.",
                "Sensor replacement": "Sensor readings erratic. Replace unit and recalibrate.",
                "Cooling system repair": "Temperature alerts received. Check coolant levels and fan operation.",
                "Electrical system check": "Current draw fluctuating. Inspect motor windings and connections.",
                "AI alert follow-up inspection": "Predictive model flagged potential anomaly. Verify sensor readings manually."
            }

            wo = WorkOrder(
                asset_id=asset.id,
                title=wo_title,
                description=wo_descriptions.get(wo_title, fake.sentence()),
                status=status,

                priority=random.choices(
                    list(Priority),
                    weights=[20, 40, 30, 10]
                )[0],
                assigned_to=fake.name(),
                created_at=created,
                due_date=created + timedelta(days=random.randint(1, 30)),
                completed_at=created + timedelta(days=random.randint(1, 15)) if status == WorkOrderStatus.completed else None
            )
            db.add(wo)
            work_order_objects.append(wo)

    db.commit()
    print(f"✅ Created {len(work_order_objects):,} work orders")

    # ─────────────────────────────────────────────
    # STEP 6: CREATE COST RECORDS
    # ─────────────────────────────────────────────
    
    print("\n💰 Step 6: Creating Cost Records...")
    cost_count = 0
    
    # Build asset lookup for asset_type
    asset_by_id = {a.id: a for a in asset_objects}
    
    # Define a pool of technicians with varying skills, rates, and efficiencies
    technician_pool = [
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Junior", "rate": random.uniform(35, 60), "efficiency": random.uniform(1.2, 1.8)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Junior", "rate": random.uniform(35, 60), "efficiency": random.uniform(1.2, 1.8)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Junior", "rate": random.uniform(35, 60), "efficiency": random.uniform(1.2, 1.8)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Intermediate", "rate": random.uniform(70, 110), "efficiency": random.uniform(0.9, 1.1)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Intermediate", "rate": random.uniform(70, 110), "efficiency": random.uniform(0.9, 1.1)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Intermediate", "rate": random.uniform(70, 110), "efficiency": random.uniform(0.9, 1.1)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Intermediate", "rate": random.uniform(70, 110), "efficiency": random.uniform(0.9, 1.1)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Senior", "rate": random.uniform(130, 180), "efficiency": random.uniform(0.7, 0.85)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Senior", "rate": random.uniform(130, 180), "efficiency": random.uniform(0.7, 0.85)},
        {"name": f"{fake.first_name()} {fake.last_name()}", "skill": "Master", "rate": random.uniform(200, 300), "efficiency": random.uniform(0.5, 0.65)}
    ]
    
    for wo in work_order_objects:
        # Get asset type for real-world cost lookup
        asset = asset_by_id.get(wo.asset_id)
        asset_type = asset.asset_type if asset else "CNC Machine"  # fallback
        priority_str = wo.priority.value if hasattr(wo.priority, 'value') else str(wo.priority)
        
        # Look up real-world repair cost range for this asset type + priority
        type_costs = repair_cost_ranges.get(asset_type, {
            "low": (1_000, 5_000), "medium": (5_000, 20_000),
            "high": (15_000, 60_000), "critical": (40_000, 120_000)
        })
        cost_range = type_costs.get(priority_str, (1_000, 10_000))
        total_cost = round(random.uniform(cost_range[0], cost_range[1]), 2)
        
        # Split total into labor / parts / other (with slight randomization)
        labor_share = random.uniform(0.45, 0.60)
        parts_share = random.uniform(0.25, 0.38)
        other_share = 1.0 - labor_share - parts_share
        
        base_labor_cost = round(total_cost * labor_share, 2)
        parts_cost = round(total_cost * parts_share, 2)
        other_cost = round(total_cost * max(other_share, 0.02), 2)
        
        # --- NEW LOGIC: Skill-Based Labor Cost ---
        # 1. Pick a technician
        tech = random.choice(technician_pool)
        
        # 2. Determine base hours at a "standard" $100/hr rate
        base_hours = max(1.0, base_labor_cost / 100.0)
        
        # 3. Apply the technician's efficiency modifier (Master is faster, Junior is slower)
        actual_hours = round(base_hours * tech["efficiency"], 1)
        
        # 4. Calculate actual labor cost based on their specific rate
        labor_cost = round(actual_hours * tech["rate"], 2)
        
        # 5. Recalculate true total cost (parts and other costs remain the same)
        total_cost = round(labor_cost + parts_cost + other_cost, 2)
        
        cost = CostRecord(
            work_order_id=wo.id,
            labor_cost=labor_cost,
            parts_cost=parts_cost,
            other_cost=other_cost,
            total_cost=total_cost,
            labor_hours=actual_hours,
            technician_name=tech["name"],
            technician_skill_level=tech["skill"],
            technician_hourly_rate=round(tech["rate"], 2),
            created_at=wo.completed_at if wo.completed_at else wo.created_at
        )
        db.add(cost)
        cost_count += 1

    db.commit()
    print(f"✅ Created {cost_count:,} cost records")

    db.close()
    
    # ─────────────────────────────────────────────
    # FINAL SUMMARY
    # ─────────────────────────────────────────────
    
    print("\n" + "=" * 80)
    print("🎉 ML-SCALE DATA GENERATION COMPLETE!")
    print("=" * 80)
    print(f"   Industries:          {len(industry_objects)}")
    print(f"   Assets:              {len(asset_objects)}")
    print(f"   Sensor Readings:     {sensor_count:,}")
    print(f"   Maintenance Records: {maintenance_count:,}")
    print(f"   Work Orders:         {len(work_order_objects):,}")
    print(f"   Cost Records:        {cost_count:,}")
    print("\n🔗 KEY FEATURE: Realistic sensor→component correlations built in!")
    print("   High vibration → Bearing failures")
    print("   High temperature → Cooling issues")
    print("   High current → Electrical problems")
    print("=" * 80)

if __name__ == "__main__":
    generate_data()