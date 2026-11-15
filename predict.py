import pickle

from fastapi import FastAPI
from typing import Dict, Any
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from typing import get_args
import uvicorn



class Records(BaseModel):

    day_of_week: Literal[
        'Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday'
    ]

    age_band_of_driver: Literal[
        '18-30', '31-50', 'Over 51', 'Unknown', 'Under 18'
    ]

    sex_of_driver: Literal['Male', 'Female', 'Unknown']

    educational_level: Literal[
        'Junior high school', 'Elementary school',
        'High school', 'Unknown', 'Above high school',
        'Writing & reading', 'Illiterate'
    ]

    vehicle_driver_relation: Literal[
        'Employee', 'Owner', 'Unknown', 'Other'
    ]

    driving_experience: Literal[
        '5-10yr', '2-5yr', 'Above 10yr', '1-2yr',
        'Below 1yr', 'Unknown', 'No Licence', 'unknown'
    ]

    type_of_vehicle: Literal[
        'Automobile', 'Lorry (41?100Q)', 'Other', 'Unknown',
        'Pick up upto 10Q', 'Public (12 seats)', 'Stationwagen',
        'Lorry (11?40Q)', 'Public (13?45 seats)',
        'Public (> 45 seats)', 'Long lorry', 'Taxi',
        'Motorcycle', 'Special vehicle', 'Ridden horse',
        'Turbo', 'Bicycle', 'Bajaj'
    ]

    owner_of_vehicle: Literal[
        'Owner', 'Governmental', 'Unknown', 'Organization', 'Other'
    ]

    area_accident_occured: Literal[
        'Other', 'Office areas', 'Residential areas',
        'Church areas', 'Industrial areas', 'School areas',
        'Recreational areas', 'Unknown', 'Outside rural areas',
        'Hospital areas', 'Market areas', 'Rural village areas'
    ]

    lanes_or_medians: Literal[
        'Two-way (divided with broken lines road marking)',
        'Undivided Two way', 'other',
        'Double carriageway (median)', 'One way',
        'Unknown',
        'Two-way (divided with solid lines road marking)'
    ]

    road_allignment: Literal[
        'Tangent road with flat terrain',
        'Tangent road with mild grade and flat terrain',
        'Steep grade downward with mountainous terrain',
        'Tangent road with mountainous terrain and',
        'Gentle horizontal curve', 'Unknown',
        'Escarpments', 'Sharp reverse curve',
        'Tangent road with rolling terrain',
        'Steep grade upward with mountainous terrain'
    ]

    types_of_junction: Literal[
        'Y Shape', 'No junction', 'Crossing', 'Unknown',
        'Other', 'O Shape', 'T Shape', 'X Shape'
    ]

    road_surface_type: Literal[
        'Asphalt roads', 'Earth roads', 'Gravel roads',
        'Other', 'Unknown', 'Asphalt roads with some distress'
    ]

    road_surface_conditions: Literal[
        'Dry', 'Wet or damp', 'Snow', 'Flood over 3cm. deep'
    ]

    light_conditions: Literal[
        'Daylight', 'Darkness - lights lit',
        'Darkness - no lighting', 'Darkness - lights unlit'
    ]

    weather_conditions: Literal[
        'Normal', 'Raining', 'Unknown', 'Other', 'Cloudy',
        'Windy', 'Snow', 'Raining and Windy', 'Fog or mist'
    ]

    type_of_collision: Literal[
        'Vehicle with vehicle collision',
        'Collision with roadside objects',
        'Collision with pedestrians', 'Rollover',
        'Unknown', 'Collision with animals',
        'Collision with roadside-parked vehicles',
        'Fall from vehicles', 'Other', 'With Train'
    ]

    number_of_vehicles_involved: int = Field(ge=1, le=7)

    vehicle_movement: Literal[
        'Going straight', 'Moving Backward', 'Other',
        'Reversing', 'Turnover', 'Unknown', 'Getting off',
        'Entering a junction', 'Overtaking', 'Stopping',
        'U-Turn', 'Waiting to go', 'Parked'
    ]

    pedestrian_movement: Literal[
        'Not a Pedestrian',
        'Crossing from nearside - masked by parked or stationary vehicle',
        'Unknown or other',
        "Crossing from driver's nearside",
        'Crossing from offside - masked by parked or stationary vehicle',
        'In carriageway, stationary - not crossing (standing or playing)',
        'Walking along in carriageway, back to traffic',
        'In carriageway, stationary - not crossing (standing or playing) - masked by parked or stationary vehicle',
        'Walking along in carriageway, facing traffic'
    ]

    hour: int = Field(ge=0, le=23)

    
    @field_validator('*', mode='before')
    @classmethod
    def normalize_categories(cls, value, info):

        if isinstance(value, str):
            value = value.strip()

            field = cls.model_fields[info.field_name]
            allowed = get_args(field.annotation)

            for option in allowed:
                if isinstance(option, str) and option.casefold() == value.casefold():
                    return option

        return value

class AccidentRequests(BaseModel):
    fatal_probability: float
    fatal: bool



app = FastAPI(title="accident-fatality-prediction")

with open('model.bin', 'rb') as f_in:
        pipeline = pickle.load(f_in)


def predict_single(record):
    result = pipeline.predict_proba(record)[0, 1]
    return float(result)


@app.post("/predict")
def predict(record: Records) -> AccidentRequests:
    probability = predict_single(record.model_dump())

    return AccidentRequests(
          fatal_probability= probability,
          fatal= probability >= 0.10
    )



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)


# print('probability of fatality: ', fatal)

# if fatal >= 0.10:
#         print('Fatal accident')
# else:
#         print('Non-fatal accident')



