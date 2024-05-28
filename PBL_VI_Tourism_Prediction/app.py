# from flask import Flask, render_template, request
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.ensemble import RandomForestRegressor

# app = Flask(__name__)

# # Load the dataset
# data = pd.read_csv("./model_data.csv")
# if 'Unnamed: 0' in data.columns:
#     data.drop(columns=['Unnamed: 0'], inplace=True)

# # Encode categorical variables
# label_encoders = {}
# for column in data.columns:
#     if data[column].dtype == 'object':
#         label_encoders[column] = LabelEncoder()
#         data[column] = label_encoders[column].fit_transform(data[column])

# # Define features and target variable
# X = data.drop(columns=['total_cost'])
# y = data['total_cost']

# # Train a random forest regressor
# model = RandomForestRegressor()
# model.fit(X, y)

# # Route to render the form
# @app.route('/')
# def home():
#     return render_template('form.html')

# # Route to handle form submission and display prediction
# @app.route('/predict', methods=['POST'])
# def predict():
#     input_data = {}
#     for key, value in request.form.items():
#         input_data[key] = value
    
#     # Create DataFrame from input data
#     input_df = pd.DataFrame([input_data])
#     input_df = input_df[X.columns]
    
#     # Encode input data
#     for column, encoder in label_encoders.items():
#         if column in input_df.columns:
#             input_df[column] = encoder.transform(input_df[column])
    
#     # Predict total cost
#     total_cost = model.predict(input_df)
    
#     # return str(total_cost[0])
#     return render_template('result.html', total_cost=total_cost[0])

# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Define the exchange rate from Tanzanian Shilling to Indian Rupee
exchange_rate = 0.027  # For example, 1 Tanzanian Shilling = 0.027 Indian Rupees

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("./model_data.csv")
if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

# Encode categorical variables
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column])

# Define features and target variable
X = data.drop(columns=['total_cost'])
y = data['total_cost']

# Train a random forest regressor
model = RandomForestRegressor()
model.fit(X, y)

# Route to render the form
@app.route('/')
def home():
    return render_template('form.html')

# Route to handle form submission and display prediction
@app.route('/predict', methods=['POST'])
def predict():
    input_data = {}
    for key, value in request.form.items():
        input_data[key] = value
    
    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])
    input_df = input_df[X.columns]
    
    # Encode input data
    for column, encoder in label_encoders.items():
        if column in input_df.columns:
            input_df[column] = encoder.transform(input_df[column])
    
    # Predict total cost
    total_cost_tzs = model.predict(input_df)
    
    # Convert total cost from Tanzanian Shilling to Indian Rupees
    total_cost_inr = total_cost_tzs * exchange_rate
    
    print("Total Cost in Indian Rupees:", total_cost_inr)  # For debugging
    
    return render_template('result.html', total_cost_inr=total_cost_inr[0])

if __name__ == '__main__':
    app.run(debug=True)

