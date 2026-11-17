from pydantic import BaseModel, Field
from typing import Optional

class Facility(BaseModel):
    facility_id: str
    name: str
    city: str
    facility_type: str  # Hospital or Central Warehouse
    has_cold_chain: bool = True
    max_capacity: int = 10000

class MedicineBatch(BaseModel):
    batch_id: str
    facility_id: str
    medicine_name: str
    quantity: int = Field(ge=0)
    days_to_expiry: int = Field(ge=0)
    requires_cold_chain: bool = False
    unit_cost_inr: float = 120.0

class TransferAction(BaseModel):
    action_id: str
    medicine_name: str
    source_facility: str
    destination_facility: str
    transfer_qty: int
    eta_days: float
    transport_cost_inr: float
    min_shelf_life_on_arrival: int
    status: str = "PENDING_HUMAN_APPROVAL"
