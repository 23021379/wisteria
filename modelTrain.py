import pandas as pd
import re
import numpy as np
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error

# numbers = [str(i) for i in range(1, 389)]
# print(numbers)
# time.sleep(10)

data = pd.read_csv(r"[REDACTED_BY_SCRIPT]")  # Replace "your_data.csv"
reference_date = pd.to_datetime('2000-01-01')

date_columns = []
for col in data.columns:
    # Try converting to datetime; if it works, it's a date column
    try:
        # Attempt conversion to datetime, handling potential errors
        test_series = pd.to_datetime(data[col], format='%m-%Y', errors='coerce') #or dayfirst=True depending on format
        if test_series.notna().any(): #check that not all values failed to parse
            date_columns.append(col)
    except (ValueError, TypeError):
        pass  # Not a date column

print(f"[REDACTED_BY_SCRIPT]")
reference_date = pd.to_datetime('2000-01-01')
for col in date_columns:
    #Convert successful date columns to date time objects
    data[col] = pd.to_datetime(data[col], format='%m-%Y', errors='coerce')
    #Calculate days since a reference date
    
    data[col] = (data[col] - reference_date).dt.days
    try:
        data[col] = data[col].str.replace(' months', '')
    except:
        pass
    try:
        data[col] = data[col].str.replace(' years', '')
    except:
        pass
    try:
        data[col] = data[col].str.replace(' days', '')
    except:
        pass
for col in data.columns:
    try:
        data[col] = data[col].str.replace('-', '0.000001')
    except:
        pass
    try:
        data[col] = data[col].str.replace('M', '')
        number = float(data[col])
        data[col]=number * 1000000
    except:pass
    try:
        data[col] = data[col].str.replace('£', '')
    except:pass
    try:
        data[col] = data[col].str.replace(',', '')
    except:pass
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if data[col].isna().all():
            data[col] = np.nan
    else:pass


data = data.fillna(data.mean())
data = data.fillna(0.000001)
x = data.drop("3", axis=1)  # Replace "target" with your target column name
y = data["3"]

feature_cols = [str(i) for i in range(1, data.shape[1])]
num_features_to_select = 50  # Start with a smaller number, e.g., 100 - Tune this!
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
X_selected = selector.fit_transform(x, y) #Apply feature selection to X
selected_feature_indices = selector.get_support(indices=True)
selected_feature_names = [feature_cols[i] for i in selected_feature_indices]

# Convert back to DataFrame for easier handling (optional, but can be helpful)
X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)
mean_label = y.mean()
X_train, X_val, y_train, y_val = train_test_split(X_selected_df, y, test_size=0.2, random_state=42)


x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)
y_train = y_scaler.fit_transform(y_train.values.reshape(-1, 1)) #scale training labels
y_val = y_scaler.transform(y_val.values.reshape(-1, 1)) #Scale testing labels

# Get the number of selected features (after feature selection)
num_features = X_train.shape[1]

# 3. Define Model (simpler architecture)
model = keras.Sequential([
    keras.layers.Input(shape=(num_features,)),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l1(0.001)), #Try L1 regularization to reduce weights
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3), #A moderate dropout value
    keras.layers.Dense(1)  # Output layer (no activation for regression)
])

# 4. Compile Model
model.compile(optimizer='adam',
              loss='mean_squared_error',  # Mean Squared Error for regression
              metrics=['mean_absolute_error']) # Metric is mean abs error

# 5. Train Model
history = model.fit(X_train, y_train,
                    epochs=50,   # Reduce epochs to prevent overfitting
                    batch_size=32,
                    validation_data=(X_val, y_val))


loss, mae = model.evaluate(X_val, y_val) # Evaluate on the validation set
print(f"[REDACTED_BY_SCRIPT]")
print(f"[REDACTED_BY_SCRIPT]")
print(f"mae/mean_label={mae/mean_label}")
print(f"[REDACTED_BY_SCRIPT]")

y_pred_scaled = model.predict(X_val)  # Get predictions on the scaled test set
y_pred = y_scaler.inverse_transform(y_pred_scaled)
mae = mean_absolute_error(y_val, y_pred)
print(f"[REDACTED_BY_SCRIPT]")





data = pd.read_csv(r"[REDACTED_BY_SCRIPT]")  # Replace "your_data.csv"
reference_date = pd.to_datetime('2000-01-01')

date_columns = []
for col in data.columns:
    # Try converting to datetime; if it works, it's a date column
    try:
        # Attempt conversion to datetime, handling potential errors
        test_series = pd.to_datetime(data[col], format='%m-%Y', errors='coerce') #or dayfirst=True depending on format
        if test_series.notna().any(): #check that not all values failed to parse
            date_columns.append(col)
    except (ValueError, TypeError):
        pass  # Not a date column

print(f"[REDACTED_BY_SCRIPT]")
reference_date = pd.to_datetime('2000-01-01')
for col in date_columns:
    #Convert successful date columns to date time objects
    data[col] = pd.to_datetime(data[col], format='%m-%Y', errors='coerce')
    #Calculate days since a reference date
    
    data[col] = (data[col] - reference_date).dt.days
    try:
        data[col] = data[col].str.replace(' months', '')
    except:
        pass
    try:
        data[col] = data[col].str.replace(' years', '')
    except:
        pass
    try:
        data[col] = data[col].str.replace(' days', '')
    except:
        pass
for col in data.columns:
    try:
        data[col] = data[col].str.replace('-', '0.000001')
    except:
        pass
    try:
        data[col] = data[col].str.replace('M', '')
        number = float(data[col])
        data[col]=number * 1000000
    except:pass
    try:
        data[col] = data[col].str.replace('£', '')
    except:pass
    try:
        data[col] = data[col].str.replace(',', '')
    except:pass
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')
        if data[col].isna().all():
            data[col] = np.nan
    else:pass
#data = data.fillna(data.mean())
data = data.fillna(0.000001)
data = data.drop('3', axis=1)

data=data[selected_feature_names]
new_data_scaled = x_scaler.transform(data)
prediction = model.predict(new_data_scaled)

y_pred = y_scaler.inverse_transform(prediction)
print(f"[REDACTED_BY_SCRIPT]")

# 7. Save Model
model.save('[REDACTED_BY_SCRIPT]')
