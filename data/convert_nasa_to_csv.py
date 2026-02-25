import pandas as pd
import os

print("=" * 50)
print("🚀 EXPORTING NASA CMAPSS DATA TO CSV WITH HEADERS")
print("=" * 50)

columns = ['engine_id', 'cycle', 'altitude', 'mach_number', 'throttle_resolver'] + \
          [
              'fan_inlet_temp',       # Sensor 1 (T2)
              'lpc_outlet_temp',      # Sensor 2 (T24)
              'hpc_outlet_temp',      # Sensor 3 (T30)
              'lpt_outlet_temp',      # Sensor 4 (T50)
              'fan_inlet_press',      # Sensor 5 (P2)
              'bypass_duct_press',    # Sensor 6 (P15)
              'hpc_outlet_press',     # Sensor 7 (P30)
              'phys_fan_speed',       # Sensor 8 (Nf)
              'phys_core_speed',      # Sensor 9 (Nc)
              'engine_press_ratio',   # Sensor 10 (epr)
              'hpc_outlet_static_press', # Sensor 11 (Ps30)
              'fuel_flow_ratio',      # Sensor 12 (phi)
              'corr_fan_speed',       # Sensor 13 (NRf)
              'corr_core_speed',      # Sensor 14 (NRc)
              'bypass_ratio',         # Sensor 15 (BPR)
              'burner_fuel_air_ratio',# Sensor 16 (farB)
              'bleed_enthalpy',       # Sensor 17 (htBleed)
              'demanded_fan_speed',   # Sensor 18 (Nf_dmd)
              'demanded_corr_fan_speed', # Sensor 19 (PCNFR_dmd)
              'hpt_coolant_bleed',    # Sensor 20 (W31)
              'lpt_coolant_bleed'     # Sensor 21 (W32)
          ]

base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, 'datasets', 'train_FD001.txt')
output_path = os.path.join(base_dir, 'datasets', 'nasa_sensor_data_with_headers.csv')

try:
    df = pd.read_csv(train_path, sep=r'\s+', header=None, names=columns)
    df.to_csv(output_path, index=False)
    print(f"✅ Exported {len(df)} records to {output_path}")
except FileNotFoundError:
    print(f"❌ Could not find {train_path}. Please check if the file exists.")
except Exception as e:
    print(f"❌ An error occurred: {e}")
