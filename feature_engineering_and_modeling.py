!pip install joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def preprocess_data(df):

    
    df = df[['genres', 'budget', 'revenue']].dropna().copy()
    df = df[df['revenue'] > 0]  # Ensure valid revenue values
    print(df.columns)

    df['log_budget'] = np.log1p(df['budget'])
    df['log_revenue'] = np.log1p(df['revenue'])

    
    genre_means = df.groupby('genres')['log_revenue'].mean().to_dict()
    df['genre_encoded'] = df['genres'].map(genre_means)

    return df.drop(columns=['genres', 'budget', 'revenue']), genre_means

def train_model(df):
    
    
    df, genre_means = preprocess_data(df)
    X, y = df.drop(columns=['log_revenue']), df['log_revenue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    joblib.dump(model, 'box_office_model.pkl')
    joblib.dump(genre_means, 'genre_means.pkl')

    y_pred = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_})
    feature_importance.sort_values(by='Coefficient', ascending=False, inplace=True)

    print(f"\n🎬 Movie Box Office Revenue Prediction Model 📊\n"
          f"------------------------------------------------\n"
          f"Model Performance:\n"
          f"- Mean Absolute Error (MAE): {metrics['MAE']:.2f}\n"
          f"- Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}\n"
          f"- R-Squared (R2): {metrics['R2']:.2f}\n")

    print("\n🔥 Key Influencing Factors:\n", feature_importance.to_string(index=False))

    plt.figure(figsize=(10, 5))
    sns.barplot(x=feature_importance['Coefficient'], y=feature_importance['Feature'])
    plt.title('Feature Importance in Predicting Box Office Revenue')
    plt.show()

    return model

def predict_revenue(genre, budget):
    
    
    model = joblib.load('box_office_model.pkl')
    genre_means = joblib.load('genre_means.pkl')

    genre_encoded = genre_means.get(genre, np.mean(list(genre_means.values())))  # Default to mean if unknown
    log_budget = np.log1p(budget)

    input_data = pd.DataFrame({'log_budget': [log_budget], 'genre_encoded': [genre_encoded]})
    revenue_pred = np.expm1(model.predict(input_data)[0])  # Convert log revenue back to actual revenue

    print(f"🎥 Estimated Box Office Revenue: ${revenue_pred:,.2f}")
    return revenue_pred

if __name__ == "__main__":
    
    train_model(df)

    genre = input("Enter movie genre: ")
    budget = float(input("Enter movie budget ($): "))
    predict_revenue(genre, budget)
