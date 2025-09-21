import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("car data.csv")
print("Original Data (First 5 rows):")
print(df.head())

df.fillna(method='ffill', inplace=True)

categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

numerical_cols = ['Year', 'Present_Price', 'Kms_Driven']
for col in numerical_cols:
    if col in df.columns:
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[[col]])

df.to_csv("cleaned_car_data.csv", index=False)
print("\n Cleaned dataset saved as 'cleaned_car_data.csv'")

X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Absolute Error: {mae}")
print(f"R² Score: {r2}")

residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

if hasattr(model, 'coef_'):
    feature_importance = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=feature_importance.index, y=feature_importance.values)
    plt.title('Feature Importance')
    plt.xticks(rotation=90)
    plt.show()
