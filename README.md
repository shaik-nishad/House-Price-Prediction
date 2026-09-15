# California House Price Prediction

A machine-learning project that estimates median house values from neighbourhood-level information. I first built a small Linear Regression demo using five sample records. I then upgraded it using a real dataset, model evaluation and an interactive Streamlit interface.

## Why I built this

I wanted to understand how a regression model moves from sample data to a prediction that a user can interact with. The project helped me practise data handling with Pandas, train/test splitting, regression metrics and building a simple interface.

## Dataset

The project uses the California Housing dataset provided by Scikit-learn. It contains 20,640 rows and eight numerical input features. The target represents median house value in units of $100,000.

Input features:

- Median income
- Median house age
- Average rooms
- Average bedrooms
- Population
- Average occupancy
- Latitude
- Longitude

## Models compared

- Linear Regression: used as a simple baseline
- Random Forest Regressor: used to capture non-linear relationships

The same 80/20 train-test split and `random_state=42` are used for a reproducible comparison. Run `train_model.py` to see the actual MAE, RMSE and R² values.

## Technologies

- Python
- Pandas
- Scikit-learn
- Streamlit

## Run locally

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Project structure

```text
.
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── legacy/
    └── basic_house_price.py
```

## What I learned

- Why a model should be evaluated on data it did not see during training
- How MAE, RMSE and R² describe regression performance
- Why a baseline model is useful before trying a more flexible model
- How to connect a trained model to a simple user interface

## Limitations

- The dataset describes California districts from the 1990 census, so predictions are not current market valuations.
- The inputs describe neighbourhood averages, not every detail of an individual house.
- The application is an educational project and should not be used for financial decisions.

## Future improvements

- Add exploratory-data-analysis charts
- Compare additional regression models using cross-validation
- Add automated tests for input validation
- Track model settings and experiment results

## Original version

The first 22-line Linear Regression version is preserved in `legacy/basic_house_price.py` to show how the project developed.
