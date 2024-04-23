import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('weather.csv')

# Split features and target variable
X = df.drop(columns=['date', 'weather'])
y = df['weather']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Print classification report to evaluate model performance
print(classification_report(y_test, predictions))

# Define a dictionary containing weather data for a test instance
test_weather = {
    'precipitation': 0.0,
    'temp_max': 40,
    'temp_min': 30,
    'wind': 5.4
}

# Create a DataFrame for the test instance
test_df = pd.DataFrame([test_weather])

# Use the trained model to predict the weather for the test instance
model.predict(test_df)
