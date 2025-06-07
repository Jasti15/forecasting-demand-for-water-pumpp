from flask import Flask, render_template, redirect, request,flash
import mysql.connector
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from keras.models import load_model
from keras.metrics import MeanSquaredError
from sklearn.preprocessing import LabelEncoder


app = Flask(__name__)

mydb = mysql.connector.connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='',
    database='Groundwater'
)

mycur = mydb.cursor()




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/registration', methods=['POST', 'GET'])
def registration():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']
        phonenumber= request.form['phonenumber']
        age  = request.form['age']
        if password == confirmpassword:
            sql = 'SELECT * FROM users WHERE email = %s'
            val = (email,)
            mycur.execute(sql, val)
            data = mycur.fetchone()
            if data is not None:
                msg = 'User already registered!'
                return render_template('registration.html', msg=msg)
            else:
                sql = 'INSERT INTO users (name, email, password,`phone number`,age) VALUES (%s, %s, %s, %s,%s)'
                val = (name, email, password, phonenumber,age)
                mycur.execute(sql, val)
                mydb.commit()
                
                msg = 'User registered successfully!'
                return render_template('registration.html', msg=msg)
        else:
            msg = 'Passwords do not match!'
            return render_template('registration.html', msg=msg)
    return render_template('registration.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        sql = 'SELECT * FROM users WHERE email=%s'
        val = (email,)
        mycur.execute(sql, val)
        data = mycur.fetchone()

        if data:
            stored_password = data[2]
            if password == stored_password:
               msg = 'user logged successfully'
               return redirect("/upload")
            else:
                msg = 'Password does not match!'
                return render_template('login.html', msg=msg)
        else:
            msg = 'User with this email does not exist. Please register.'
            return render_template('login.html', msg=msg)
    return render_template('login.html')
                            
# Set a secret key for session management (for flash messages)
app.secret_key = 'bhuvana'

# Define the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Checks if the file extension is allowed (CSV only)."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

        file = request.files['file']

        # If no file is selected
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(request.url)

        # If file is selected and is CSV
        if file and allowed_file(file.filename):
            # Save the file to the upload folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Read the CSV file
            dataset = pd.read_csv(filename)
            dataset_dict = dataset.to_dict(orient='records')

            # Flash a success message
            flash('CSV file uploaded successfully!', 'CSV file uploaded successfully!')

            return render_template('upload.html', dataset=dataset_dict)

        else:
            flash('Only CSV files are allowed!', 'error')
            return redirect(request.url)

    return render_template('upload.html', dataset=None)













@app.route('/algo')
def algo():
    # Load and preprocess the dataset
    df = pd.read_csv('uploads\Forecasting_Dataset.csv')

    # Fill missing values with median (for numerical data) and mode (for categorical data)
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    #label encoding the data.
    # Convert categorical columns to numerical using label encoding if needed
    from sklearn.preprocessing import LabelEncoder
    # Store original column names
    original_columns = df.select_dtypes(include='object').columns

    # Initialize LabelEncoder
    label_encoders = {}

    # Apply LabelEncoder to each categorical variable
    for col in original_columns:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])

    # Print the mapping between original categories and numerical labels
    for col, encoder in label_encoders.items():
        print(f"Mapping for column '{col}':")
        for label, category in enumerate(encoder.classes_):
            print(f"Label {label}: {category}")
    
    # Drop 'S.no.' column and handle missing values
    df = df.drop(columns=['S.no.'])
    df.fillna(df.median(), inplace=True)

    # Define features and target variable
    X = df.drop(columns=['Net Ground Water Availability for future use'])
    y = df['Net Ground Water Availability for future use']
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Initialize models
    svm_model = SVR(kernel='rbf')
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

    # Train models
    svm_model.fit(X_train_scaled, y_train)
    rf_model.fit(X_train_scaled, y_train)
    xgb_model.fit(X_train_scaled, y_train)

    # Evaluate models
    models = {"SVM": svm_model, "Random Forest": rf_model, "XGBoost": xgb_model}
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        results[name] = {
            "MAE": mean_absolute_error(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "R2 Score": r2_score(y_test, y_pred)
        }

    # Train LSTM model
    X_train_lstm = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
    X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train_lstm, y_train, epochs=20, batch_size=16, validation_data=(X_test_lstm, y_test), verbose=1)

    # Predict using LSTM
    y_pred_lstm = lstm_model.predict(X_test_lstm)
    results["LSTM"] = {
        "MAE": mean_absolute_error(y_test, y_pred_lstm),
        "MSE": mean_squared_error(y_test, y_pred_lstm),
        "R2 Score": r2_score(y_test, y_pred_lstm)
    }

    # Stacking Model (meta learner: XGBoost)
    stacking_model = StackingRegressor(
        estimators=[('rf', rf_model), ('svm', svm_model), ('xgb', xgb_model)],
        final_estimator=XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42)
    )
    stacking_model.fit(X_train_scaled, y_train)

    # Predict using Stacking model
    y_pred_stacking = stacking_model.predict(X_test_scaled)
    results["Stacking"] = {
        "MAE": mean_absolute_error(y_test, y_pred_stacking),
        "MSE": mean_squared_error(y_test, y_pred_stacking),
        "R2 Score": r2_score(y_test, y_pred_stacking)
    }

    # Pass the results to the frontend template
    return render_template('algo.html', results=results)





@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        # Extract input values from the form
        data = {
            'Name of State': request.form['state'],
            'Name of District': request.form['district'],
            'Recharge from rainfall During Monsoon Season': float(request.form['rainfall_monsoon']),
            'Recharge from other sources During Monsoon Season': float(request.form['other_sources_monsoon']),
            'Recharge from rainfall During Non Monsoon Season': float(request.form['rainfall_non_monsoon']),
            'Recharge from other sources During Non Monsoon Season': float(request.form['other_sources_non_monsoon']),
            'Total Annual Ground Water Recharge': float(request.form['total_annual_recharge']),
            'Total Natural Discharges': float(request.form['total_natural_discharge']),
            'Annual Extractable Ground Water Resource': float(request.form['extractable_gw']),
            'Current Annual Ground Water Extraction For Irrigation': float(request.form['current_annual_extraction_irrigation']),
            'Current Annual Ground Water Extraction For Domestic & Industrial Use': float(request.form['current_annual_extraction_domestic']),
            'Total Current Annual Ground Water Extraction': float(request.form['Total_Current_Annual_Ground_Water_Extraction']),
            'Annual GW Allocation for Domestic Use as on 2025': float(request.form['annual_gw_allocation_domestic']),
            'Stage of Ground Water Extraction (%)': float(request.form['stage_of_gw_extraction'])
        }
        input_data = pd.DataFrame([data])

        # Load and preprocess the dataset
        df = pd.read_csv('uploads/Forecasting_Dataset.csv')
        df = df.drop(columns=['S.no.'])  # Assuming 'S.no.' is not needed

        # Handling missing values
        # Numeric columns: fill with median
        numeric_cols = df.select_dtypes(include=[np.number])
        df[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.median())
        
        # Categorical columns: handle separately if needed
        categorical_cols = df.select_dtypes(include=['object'])
        for col in categorical_cols:
            df[col] = df[col].fillna('Unknown')  # Or use another method like mode
        
        # Encoding categorical variables
        label_encoders = {}
        for col in categorical_cols.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col])
            if col in input_data:
                input_data[col] = label_encoders[col].transform(input_data[col])

        # Splitting data
        X = df.drop(columns=['Net Ground Water Availability for future use'])
        y = df['Net Ground Water Availability for future use']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # This line is actually unnecessary in this context

        # Prediction model
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)

        # Make prediction for the input data
        input_data_scaled = scaler.transform(input_data)  # Transform input data
        prediction = rf_model.predict(input_data_scaled)

        # Render prediction result to the frontend
        return render_template('prediction.html', prediction=prediction[0])

    # If the request is GET, show the prediction form
    return render_template('prediction.html')



if __name__ == '__main__':
    app.run(debug=True)
