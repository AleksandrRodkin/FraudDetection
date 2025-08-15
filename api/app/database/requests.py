import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from tqdm import tqdm

from api.app.database.db import async_session
from api.app.database.models import Application, Customer, Address, Device, ApplicationMetrics
from api.logger_config import log


def orm_to_dict(obj):
    return {col.name: getattr(obj, col.name) for col in obj.__table__.columns}


async def get_application_data(application_ids: list[int]):
    async with async_session() as session:
        result = await session.execute(
            select(Application, Customer, Address, Device, ApplicationMetrics)
            .join(Customer, Customer.customer_id == Application.customer_id)
            .join(Address, Address.customer_id == Application.customer_id)
            .join(Device, Device.application_id == Application.application_id)
            .join(ApplicationMetrics, ApplicationMetrics.application_id == Application.application_id)
            .where(Application.application_id.in_(application_ids))
        )

        rows = result.all()

        if not rows:
            raise ValueError(f"No applications found for ids: {application_ids}")

        all_rows = []
        ids = []
        for application, customer, address, device, application_metrics in rows:
            response = {
                **orm_to_dict(application),
                **orm_to_dict(customer),
                **orm_to_dict(address),
                **orm_to_dict(device),
                **orm_to_dict(application_metrics),
            }
            all_rows.append(response)
            ids.append(application.application_id)

        return pd.DataFrame(all_rows, index=ids)


async def load_data_from_csv(df: pd.DataFrame, session: AsyncSession):
    log.info("Filling the database")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # 1. Customer
        customer = Customer(
            income=row["income"],
            customer_age=row["customer_age"],
            employment_status=row["employment_status"],
            housing_status=row["housing_status"],
            bank_months_count=row["bank_months_count"],
            has_other_cards=bool(row["has_other_cards"]),
            email_is_free=bool(row["email_is_free"]),
            phone_home_valid=bool(row["phone_home_valid"]),
            phone_mobile_valid=bool(row["phone_mobile_valid"]),
            name_email_similarity=row["name_email_similarity"]
        )
        session.add(customer)
        await session.flush()  # get customer_id

        # 2. Address
        address = Address(
            customer_id=customer.customer_id,
            prev_address_months_count=row["prev_address_months_count"],
            current_address_months_count=row["current_address_months_count"]
        )
        session.add(address)

        # 3. Application
        application = Application(
            customer_id=customer.customer_id,
            intended_balcon_amount=row["intended_balcon_amount"],
            payment_type=row["payment_type"],
            proposed_credit_limit=row["proposed_credit_limit"],
            days_since_request=row["days_since_request"],
            foreign_request=bool(row["foreign_request"]),
            credit_risk_score=row["credit_risk_score"]
        )
        session.add(application)
        await session.flush()  # application_id

        # 4. Device
        device = Device(
            application_id=application.application_id,
            source=row["source"],
            device_os=row["device_os"],
            session_length_in_minutes=row["session_length_in_minutes"],
            keep_alive_session=bool(row["keep_alive_session"]),
            device_distinct_emails_8w=row["device_distinct_emails_8w"],
            device_fraud_count=row["device_fraud_count"]
        )
        session.add(device)

        # 5. ApplicationMetrics
        metrics = ApplicationMetrics(
            application_id=application.application_id,
            zip_count_4w=row["zip_count_4w"],
            velocity_6h=row["velocity_6h"],
            velocity_24h=row["velocity_24h"],
            velocity_4w=row["velocity_4w"],
            bank_branch_count_8w=row["bank_branch_count_8w"],
            date_of_birth_distinct_emails_4w=row["date_of_birth_distinct_emails_4w"]
        )
        session.add(metrics)

    await session.commit()
