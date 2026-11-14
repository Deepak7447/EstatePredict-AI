from pydantic import BaseModel, ConfigDict, Field, field_validator


class PropertyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    med_inc: float = Field(gt=0, le=20, description="Annual median income in $10,000s")
    house_age: float = Field(ge=1, le=52, description="Median house age in years")
    ave_rooms: float = Field(gt=0, le=200, description="Average rooms per household")
    ave_bedrms: float = Field(gt=0, le=50, description="Average bedrooms per household")
    population: int = Field(gt=0, le=50000, description="Block-group population")
    ave_occup: float = Field(gt=0, le=1500, description="Average people per household")
    latitude: float = Field(ge=32.5, le=42.0)
    longitude: float = Field(ge=-124.4, le=-114.1)

    @field_validator("med_inc", "house_age", "ave_rooms", "ave_bedrms", "ave_occup", "latitude", "longitude")
    @classmethod
    def values_must_be_finite(cls, value: float) -> float:
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("Value must be a finite number")
        return value
