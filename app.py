import pandas as pd
import streamlit as st
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


@st.cache_resource
def train_model():
    housing = fetch_california_housing(as_frame=True)
    features = housing.data
    target = housing.target

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=150, max_depth=18, random_state=42, n_jobs=-1
    )
    model.fit(x_train, y_train)

    test_predictions = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, test_predictions),
        "mae": mean_absolute_error(y_test, test_predictions) * 100_000,
    }
    return model, metrics


model, metrics = train_model()

st.title("🏠 California House Price Predictor")
st.write(
    "Estimate the median value of a California home using neighbourhood-level "
    "information from the California Housing dataset."
)

with st.sidebar:
    st.header("Model performance")
    st.metric("Test R² score", f"{metrics['r2']:.3f}")
    st.metric("Mean absolute error", f"${metrics['mae']:,.0f}")
    st.caption("Results are calculated on a held-out 20% test set.")

left, right = st.columns(2)

with left:
    median_income = st.number_input(
        "Median income (in $10,000s)", min_value=0.5, max_value=15.0, value=4.0, step=0.1
    )
    house_age = st.number_input(
        "Median house age (years)", min_value=1.0, max_value=52.0, value=25.0, step=1.0
    )
    average_rooms = st.number_input(
        "Average rooms per household", min_value=1.0, max_value=20.0, value=5.5, step=0.1
    )
    average_bedrooms = st.number_input(
        "Average bedrooms per household", min_value=0.5, max_value=10.0, value=1.1, step=0.1
    )

with right:
    population = st.number_input(
        "Block-group population", min_value=3.0, max_value=35_000.0, value=1_400.0, step=50.0
    )
    average_occupancy = st.number_input(
        "Average household occupancy", min_value=1.0, max_value=20.0, value=3.0, step=0.1
    )
    latitude = st.number_input(
        "Latitude", min_value=32.5, max_value=42.0, value=34.1, step=0.1
    )
    longitude = st.number_input(
        "Longitude", min_value=-124.5, max_value=-114.0, value=-118.2, step=0.1
    )

if st.button("Predict house value", type="primary", use_container_width=True):
    if average_bedrooms > average_rooms:
        st.error("Average bedrooms cannot be greater than average rooms.")
    else:
        input_data = pd.DataFrame(
            [
                {
                    "MedInc": median_income,
                    "HouseAge": house_age,
                    "AveRooms": average_rooms,
                    "AveBedrms": average_bedrooms,
                    "Population": population,
                    "AveOccup": average_occupancy,
                    "Latitude": latitude,
                    "Longitude": longitude,
                }
            ]
        )
        predicted_value = model.predict(input_data)[0] * 100_000
        st.success(f"Estimated median house value: ${predicted_value:,.0f}")
        st.caption(
            "This is an educational estimate based on historical district-level data, "
            "not a professional property valuation."
        )

