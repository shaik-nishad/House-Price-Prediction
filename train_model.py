from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    return {
        "Model": name,
        "MAE ($)": mean_absolute_error(y_test, predictions) * 100_000,
        "RMSE ($)": mean_squared_error(y_test, predictions) ** 0.5 * 100_000,
        "R2": r2_score(y_test, predictions),
    }


def main():
    housing = fetch_california_housing(as_frame=True)
    x = housing.data
    y = housing.target

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=18, random_state=42, n_jobs=-1
        ),
    }

    print(f"Dataset rows: {len(x):,}")
    print(f"Training rows: {len(x_train):,}")
    print(f"Testing rows: {len(x_test):,}\n")

    for name, model in models.items():
        result = evaluate_model(name, model, x_train, x_test, y_train, y_test)
        print(name)
        print(f"  MAE:  ${result['MAE ($)']:,.0f}")
        print(f"  RMSE: ${result['RMSE ($)']:,.0f}")
        print(f"  R²:   {result['R2']:.3f}\n")


if __name__ == "__main__":
    main()

