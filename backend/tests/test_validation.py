import pytest
from pydantic import ValidationError
from app.schemas import PropertyInput

def valid_payload():
    return dict(med_inc=8.33, house_age=41, ave_rooms=6.98, ave_bedrms=1.02,
                population=322, ave_occup=2.56, latitude=37.88, longitude=-122.23)

def test_valid_california_sample():
    assert PropertyInput(**valid_payload()).latitude == 37.88

def test_rejects_out_of_california_latitude():
    data = valid_payload()
    data["latitude"] = 45.2
    with pytest.raises(ValidationError):
        PropertyInput(**data)
