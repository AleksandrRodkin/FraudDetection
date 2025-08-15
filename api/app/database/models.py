"""
Description of DB tables

Note:
Many features, such as `name_email_similarity`, are derived from real-world data
(e.g., name and email), and are not stored in this form in actual production systems.
However, generating the original raw data (such as name, email, etc.) is not the goal of this project,
so the features are included in the schema as if they were explicitly stored.
"""

from sqlalchemy import ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from api.app.database.db import Base


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    income: Mapped[float]
    customer_age: Mapped[float]
    employment_status: Mapped[str]
    housing_status: Mapped[str]
    bank_months_count: Mapped[int]
    has_other_cards: Mapped[bool]
    email_is_free: Mapped[bool]
    phone_home_valid: Mapped[bool]
    phone_mobile_valid: Mapped[bool]
    name_email_similarity: Mapped[float]

    applications: Mapped[list["Application"]] = relationship(back_populates="customer", cascade="all, delete-orphan")
    address: Mapped["Address"] = relationship(back_populates="customer", uselist=False)


class Address(Base):
    __tablename__ = "addresses"

    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"), primary_key=True)
    prev_address_months_count: Mapped[int]
    current_address_months_count: Mapped[int]

    customer: Mapped["Customer"] = relationship(back_populates="address")


class Application(Base):
    __tablename__ = "applications"

    application_id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    customer_id: Mapped[int] = mapped_column(ForeignKey("customers.customer_id"))
    intended_balcon_amount: Mapped[float]
    payment_type: Mapped[str]
    proposed_credit_limit: Mapped[float]
    days_since_request: Mapped[int]
    foreign_request: Mapped[bool]
    credit_risk_score: Mapped[float]

    customer: Mapped["Customer"] = relationship(back_populates="applications")
    device: Mapped["Device"] = relationship(back_populates="application", uselist=False, cascade="all, delete-orphan")
    metrics: Mapped["ApplicationMetrics"] = relationship(back_populates="application", uselist=False,
                                                         cascade="all, delete-orphan")


class Device(Base):
    __tablename__ = "devices"

    application_id: Mapped[int] = mapped_column(ForeignKey("applications.application_id"), primary_key=True)
    source: Mapped[str]
    device_os: Mapped[str]
    session_length_in_minutes: Mapped[int]
    keep_alive_session: Mapped[bool]
    device_distinct_emails_8w: Mapped[int]
    device_fraud_count: Mapped[int]

    application: Mapped["Application"] = relationship(back_populates="device")


class ApplicationMetrics(Base):
    __tablename__ = "application_metrics"

    application_id: Mapped[int] = mapped_column(ForeignKey("applications.application_id"), primary_key=True)
    zip_count_4w: Mapped[int]
    velocity_6h: Mapped[float]
    velocity_24h: Mapped[float]
    velocity_4w: Mapped[float]
    bank_branch_count_8w: Mapped[int]
    date_of_birth_distinct_emails_4w: Mapped[int]

    application: Mapped["Application"] = relationship(back_populates="metrics")
