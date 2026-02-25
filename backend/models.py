from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
import enum

# ─────────────────────────────────────────────
# ENUMS - Fixed choices for certain fields
# ─────────────────────────────────────────────

# Why Enums? Some fields should only have specific values
# Example: Status can only be operational, failed, etc. (not random text)

class AssetStatus(str, enum.Enum):
    """Asset can only have these 4 statuses"""
    operational = "operational"
    under_maintenance = "under_maintenance"
    failed = "failed"
    decommissioned = "decommissioned"

class MaintenanceType(str, enum.Enum):
    """3 types of maintenance"""
    preventive = "preventive"      # Scheduled before failure
    corrective = "corrective"      # After something breaks
    predictive = "predictive"      # AI recommended

class WorkOrderStatus(str, enum.Enum):
    """Work order lifecycle"""
    open = "open"
    in_progress = "in_progress"
    completed = "completed"
    cancelled = "cancelled"

class Priority(str, enum.Enum):
    """How urgent is this?"""
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

# ─────────────────────────────────────────────
# TABLE 1 - INDUSTRIES
# ─────────────────────────────────────────────

class Industry(Base):
    """
    Stores industry categories (Oil & Gas, Healthcare, etc.)
    
    Why separate table? So we can easily query "Show all Healthcare assets"
    """
    __tablename__ = "industries"

    # Primary key = unique ID for each industry
    id = Column(Integer, primary_key=True, index=True)
    
    # Industry name (unique = no duplicates allowed)
    name = Column(String, unique=True, nullable=False)
    
    # Optional description
    description = Column(Text, nullable=True)

    # Relationship: One industry has many assets
    # This creates a virtual link - we can do: industry.assets
    assets = relationship("Asset", back_populates="industry")
# ─────────────────────────────────────────────
# TABLE 2 - ASSETS (The Heart of the System!)
# ─────────────────────────────────────────────

class Asset(Base):
    """
    Stores all information about each machine/equipment
    
    This is the CENTRAL table - everything connects to this
    """
    __tablename__ = "assets"

    # Basic identification
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)                     # e.g. "Turbine Engine #234"
    asset_type = Column(String, nullable=False)               # e.g. "Turbine", "Pump", "MRI Scanner"
    location = Column(String, nullable=False)                 # e.g. "Plant A - Floor 2"
    
    # Manufacturing details
    manufacturer = Column(String, nullable=True)              # e.g. "Siemens"
    model_number = Column(String, nullable=True)              # e.g. "SGT-800"
    serial_number = Column(String, unique=True, nullable=False)  # Unique identifier
    
    # Dates
    purchase_date = Column(DateTime, nullable=False)
    installation_date = Column(DateTime, nullable=True)
    warranty_expiry = Column(DateTime, nullable=True)
    
    # Status and value
    status = Column(Enum(AssetStatus), default=AssetStatus.operational)
    purchase_cost = Column(Float, nullable=False)             # Original cost
    current_value = Column(Float, nullable=True)              # After depreciation
    
    # Foreign key - links to Industries table
    industry_id = Column(Integer, ForeignKey("industries.id"))

    # Relationships - Links to other tables
    industry = relationship("Industry", back_populates="assets")
    sensors = relationship("Sensor", back_populates="asset")
    maintenance_records = relationship("MaintenanceRecord", back_populates="asset")
    work_orders = relationship("WorkOrder", back_populates="asset")

# ─────────────────────────────────────────────
# TABLE 3 - SENSORS (IoT Data)
# ─────────────────────────────────────────────

class Sensor(Base):
    """
    Stores sensor readings over time
    
    Each asset can have multiple sensors (temp, vibration, pressure)
    Each sensor has many readings over time
    """
    __tablename__ = "sensors"

    id = Column(Integer, primary_key=True, index=True)
    
    # Which asset does this sensor belong to?
    asset_id = Column(Integer, ForeignKey("assets.id"))
    
    # What type of sensor?
    sensor_type = Column(String, nullable=False)              # "temperature", "vibration", "pressure"
    unit = Column(String, nullable=False)                     # "°C", "mm/s", "PSI"
    
    # The actual reading
    value = Column(Float, nullable=False)                     # e.g. 75.2
    
    # Thresholds (what's normal?)
    min_threshold = Column(Float, nullable=True)              # Normal minimum
    max_threshold = Column(Float, nullable=True)              # Normal maximum (alert if exceeded)
    
    # When was this reading taken?
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship back to Asset
    asset = relationship("Asset", back_populates="sensors")

# ─────────────────────────────────────────────
# TABLE 4 - MAINTENANCE RECORDS (History Log)
# ─────────────────────────────────────────────

class MaintenanceRecord(Base):
    """
    Log of every maintenance activity performed
    
    Like a medical record - tracks everything done to the asset
    """
    __tablename__ = "maintenance_records"

    id = Column(Integer, primary_key=True, index=True)
    
    # Which asset was maintained?
    asset_id = Column(Integer, ForeignKey("assets.id"))
    
    # What type of maintenance?
    maintenance_type = Column(Enum(MaintenanceType), nullable=False)
    
    # Details
    description = Column(Text, nullable=True)                 # What was done
    performed_by = Column(String, nullable=False)             # Technician name
    
    # Timing
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=True)
    downtime_hours = Column(Float, nullable=True)             # How long was asset offline
    
    # Result
    outcome = Column(String, nullable=True)                   # "Resolved", "Partially Fixed", etc.

    # Relationship
    asset = relationship("Asset", back_populates="maintenance_records")

# ─────────────────────────────────────────────
# TABLE 5 - WORK ORDERS (Task Tickets)
# ─────────────────────────────────────────────

class WorkOrder(Base):
    """
    Formal maintenance task/ticket
    
    Like a JIRA ticket - tracks who's doing what, when, status
    """
    __tablename__ = "work_orders"

    id = Column(Integer, primary_key=True, index=True)
    
    # Which asset needs work?
    asset_id = Column(Integer, ForeignKey("assets.id"))
    
    # Task details
    title = Column(String, nullable=False)                    # e.g. "Replace bearing on Pump #2"
    description = Column(Text, nullable=True)
    
    # Status tracking
    status = Column(Enum(WorkOrderStatus), default=WorkOrderStatus.open)
    priority = Column(Enum(Priority), default=Priority.medium)
    
    # Assignment
    assigned_to = Column(String, nullable=True)               # Technician assigned
    
    # Dates
    created_at = Column(DateTime, default=datetime.utcnow)
    due_date = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    asset = relationship("Asset", back_populates="work_orders")
    costs = relationship("CostRecord", back_populates="work_order")


class Prediction(Base):
    """
    ML Model Predictions for Asset Health
    Stores the output of the component failure prediction model.
    """
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id"))
    
    # Predicted outcome
    risk_level = Column(String, nullable=True)         # "Healthy", "Warning", "High Risk", "Critical"
    risk_score = Column(Float, nullable=True)          # 0-100
    predicted_failure = Column(String, nullable=False) # "Healthy", "Bearing Failure", etc.
    confidence = Column(Float, nullable=False)         # 0.0 to 1.0
    recommendation = Column(Text, nullable=True)
    predicted_cost = Column(Float, nullable=True)
    
    # When was this prediction made?
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    asset = relationship("Asset")

# ─────────────────────────────────────────────
# TABLE 6 - COST RECORDS (Financial Tracking)
# ─────────────────────────────────────────────

class CostRecord(Base):
    """
    Tracks money spent on each work order
    
    Labor cost + Parts cost = Total cost
    """
    __tablename__ = "cost_records"

    id = Column(Integer, primary_key=True, index=True)
    
    # Which work order is this cost for?
    work_order_id = Column(Integer, ForeignKey("work_orders.id"))
    
    # Cost breakdown
    labor_cost = Column(Float, default=0.0)                   # Technician time
    parts_cost = Column(Float, default=0.0)                   # Spare parts used
    other_cost = Column(Float, default=0.0)                   # Misc expenses
    total_cost = Column(Float, default=0.0)                   # Sum of above
    
    # Labor tracking
    labor_hours = Column(Float, default=0.0)                  # Hours worked
    technician_name = Column(String, nullable=True)
    technician_skill_level = Column(String, nullable=True)    # e.g. "Beginner", "Intermediate", "Expert"
    technician_hourly_rate = Column(Float, default=0.0)       # e.g. 45.0, 85.0, 180.0
    
    # When was this cost recorded?
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    work_order = relationship("WorkOrder", back_populates="costs")