from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


def utcnow_naive() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, index=True)
    age = Column(Integer)
    sex = Column(String)
    diabetes_duration_years = Column(Integer)
    eye_side = Column(String)
    visit_date = Column(DateTime, default=utcnow_naive)
    notes = Column(String)
    source_image_path = Column(String)

    prediction = relationship("Prediction", back_populates="case", uselist=False)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"))
    predicted_class_index = Column(Integer)
    predicted_label = Column(String)
    expected_grade = Column(Float)
    confidence = Column(Float)
    class_probabilities = Column(String)
    ensemble_members = Column(Text)
    ensemble_member_weights = Column(Text)
    threshold_vector = Column(Text)
    ereg_graph = Column(Text)

    case = relationship("Case", back_populates="prediction")
    artifacts = relationship("Artifact", back_populates="prediction")


class Artifact(Base):
    __tablename__ = "artifacts"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    model_name = Column(String)
    gradcam_image_path = Column(String)

    prediction = relationship("Prediction", back_populates="artifacts")
