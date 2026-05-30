"""Live inference handler for the House Price Stacking Regressor."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# Encoding maps must match training label encoding (alphabetical order)
_NEIGHBORHOOD_CLASSES  = ['Downtown', 'Historic', 'Rural', 'Suburbs', 'Waterfront']
_CONDITION_CLASSES     = ['Excellent', 'Fair', 'Good', 'Poor']
_GARAGE_CLASSES        = ['Attached', 'Detached', 'None']
_HEATING_CLASSES       = ['Electric', 'Gas', 'Oil', 'Solar']


def _encode(value: str, classes: list) -> int:
    return classes.index(value)


def make_prediction(
    model,
    year_built: int,
    remodel_year: int,
    lot_area: int,
    total_sqft: int,
    basement_sqft: int,
    bedrooms: int,
    bathrooms: float,
    garage_cars: int,
    fireplaces: int,
    has_pool: int,
    overall_qual: int,
    neighborhood: str,
    condition: str,
    garage_type: str,
    heating: str,
) -> float:
    """Run one prediction and return the price in dollars."""

    house_age   = 2023 - year_built
    remodel_age = 2023 - remodel_year
    remodeled   = int(remodel_year != year_built)
    price_per_sqft_approx = lot_area / (total_sqft + 1)
    total_rooms = bedrooms + bathrooms

    row = pd.DataFrame([{
        "Lot_Area":                lot_area,
        "Total_SqFt":              total_sqft,
        "Basement_SqFt":           basement_sqft,
        "Bedrooms":                bedrooms,
        "Bathrooms":               bathrooms,
        "Garage_Cars":             garage_cars,
        "Fireplaces":              fireplaces,
        "Has_Pool":                has_pool,
        "Overall_Qual":            overall_qual,
        "Neighborhood":            _encode(neighborhood, _NEIGHBORHOOD_CLASSES),
        "House_Condition":         _encode(condition,    _CONDITION_CLASSES),
        "Garage_Type":             _encode(garage_type,  _GARAGE_CLASSES),
        "Heating_Type":            _encode(heating,      _HEATING_CLASSES),
        "House_Age":               house_age,
        "Remodel_Age":             remodel_age,
        "Remodeled":               remodeled,
        "Price_Per_SqFt_approx":   price_per_sqft_approx,
        "Total_Rooms":             total_rooms,
    }])

    log_price = model.predict(row.values)[0]
    return float(np.expm1(log_price))
