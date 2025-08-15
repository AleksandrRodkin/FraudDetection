"""
Describe expected type of data
"""
from pydantic import BaseModel


class ApplicationData(BaseModel):
    class Config:
        from_attributes = True
        extra = "ignore"

    income: float
    name_email_similarity: float
    prev_address_months_count: int
    current_address_months_count: int
    customer_age: float
    days_since_request: float
    intended_balcon_amount: float
    payment_type: str
    zip_count_4w: int = None
    velocity_6h: float
    velocity_24h: float
    velocity_4w: float
    bank_branch_count_8w: int = None
    date_of_birth_distinct_emails_4w: int
    employment_status: str
    credit_risk_score: float
    email_is_free: bool
    housing_status: str
    phone_home_valid: bool
    phone_mobile_valid: bool
    bank_months_count: int
    has_other_cards: bool
    proposed_credit_limit: int
    foreign_request: bool
    source: str = None
    session_length_in_minutes: float
    device_os: str
    keep_alive_session: bool
    device_distinct_emails_8w: int
    device_fraud_count: int = None
    month: int = None


class Response(BaseModel):
    class Config:
        from_attributes = True

    fraud_indicator: int
    fraud_probability: float | None = None
