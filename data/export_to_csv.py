import sys
import os
import pandas as pd

# Allow importing from backend folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from database import engine

def export_to_csv():
    """
    Exports all tables from the SQLite database to CSV files.
    """
    print("=" * 50)
    print("🚀 STARTING DATA EXPORT TO CSV")
    print("=" * 50)

    # List of tables to export
    tables = [
        "industries",
        "assets",
        "sensors",
        "maintenance_records",
        "work_orders",
        "cost_records"
    ]
    
    # Output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'datasets')
    os.makedirs(output_dir, exist_ok=True)
    
    for table in tables:
        try:
            print(f"   Exporting {table}...", end=" ")
            df = pd.read_sql_table(table, engine)
            output_path = os.path.join(output_dir, f"{table}.csv")
            df.to_csv(output_path, index=False)
            print(f"✅ Saved to {output_path} ({len(df)} rows)")
        except ValueError as e:
            print(f"❌ Error: {e}")
        except Exception as e:
            print(f"❌ Unexpected Error exporting {table}: {e}")

    print("=" * 50)
    print("🎉 EXPORT COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    export_to_csv()
