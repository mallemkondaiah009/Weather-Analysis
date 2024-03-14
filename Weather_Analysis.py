import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    return pd.read_csv(file_path)

def explore_data(df):
    print(df.head())
    print(df.info())
    print(df.describe())

def calculate_monthly_avg_max_temp(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    return df.groupby('Month')['MaxTemp'].mean()

def prepare_data_for_prediction(df):
    X = df[['MinTemp', 'MaxTemp']]
    y = df['Rainfall']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_linear_regression_model(X_train, X_test, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error for Rainfall Prediction: {mse}')

def analyze_rainfall_months(monthly_avg_max_temp):
    highest_rainfall_month = monthly_avg_max_temp.idxmax()
    lowest_rainfall_month = monthly_avg_max_temp.idxmin()
    print(f'Highest rainfall month: {highest_rainfall_month}, Lowest rainfall month: {lowest_rainfall_month}')

def visualize_data(df, monthly_avg_max_temp):
    sns.pairplot(df[['MinTemp', 'MaxTemp', 'Rainfall']])
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_avg_max_temp.index, monthly_avg_max_temp.values, marker='o')
    plt.xlabel('Month')
    plt.ylabel('Average Max Temperature')
    plt.title('Monthly Average Max Temperature')
    plt.grid(True)
    plt.show()

# Load the data
file_path = "Weather_Dataset.csv"

df = load_data(file_path)

# Explore the data
explore_data(df)

# Calculate average MaxTemp by month
monthly_avg_max_temp = calculate_monthly_avg_max_temp(df)

# Prepare data for prediction
X_train, X_test, y_train, y_test = prepare_data_for_prediction(df)

# Train a linear regression model
model = train_linear_regression_model(X_train, X_test, y_train)

# Evaluate the model
evaluate_model(model, X_test, y_test)

# Analyze rainfall months
analyze_rainfall_months(monthly_avg_max_temp)

# Visualize the data
visualize_data(df, monthly_avg_max_temp)
