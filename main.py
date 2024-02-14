import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

Sequential = tf.keras.models.Sequential
Dense = tf.keras.layers.Dense
Adam = tf.keras.optimizers.legacy.Adam


# ----------------------------------------------------------------------------------------------------
# BELOW IS CODE TO LOAD AND FILTER WEARING ANF NOT WEARING DATA
# Number of hours/minutes can be specified in the functions

def load_and_filter_wearing_data(recumbent_hours=None):

    activities_sorted = pd.read_csv('data/model/wearing/ActivitiesSorted.csv')
    activities_with_name = activities_sorted.dropna(subset=['Activity']).copy()
    activities_with_name['Label'] = activities_with_name['Activity']
    activities_with_name.drop('Activity', axis=1, inplace=True)

    # Load Cleared_Recumbent1.csv and filter for recumbent (1)
    cleared_recumbent = pd.read_csv('data/model/wearing/Cleared_Recumbent1.csv')
    recumbent_filtered = cleared_recumbent[cleared_recumbent['Recumbent'] == 1].copy()
    recumbent_filtered['Label'] = 'recumbent'
    
    # Adjust the number of hours of recumbent data if specified
    if recumbent_hours is not None:
        recumbent_filtered['Date and Time'] = pd.to_datetime(recumbent_filtered['Date and Time'])
        start_time = recumbent_filtered['Date and Time'].min()
        end_time = start_time + pd.Timedelta(hours=recumbent_hours)
        recumbent_filtered = recumbent_filtered[(recumbent_filtered['Date and Time'] >= start_time) & (recumbent_filtered['Date and Time'] <= end_time)]

    usual_walking_combined = pd.DataFrame()

    # Load and filter Cleared_UsualWalking CSVs
    for i in range(1, 6):  # versions 1 through 5
        try:
            usual_walking = pd.read_csv(f'data/model/wearing/Cleared_UsualWalking{i}.csv')
            walking_filtered = usual_walking[usual_walking['Usual Walking'] == 1].copy()
            walking_filtered['Label'] = 'walking'
            usual_walking_combined = pd.concat([usual_walking_combined, walking_filtered], ignore_index=True)
        except FileNotFoundError:
            print(f'Cleared_UsualWalking{i}.csv not found. Skipping...')

    recumbent_filtered = recumbent_filtered[['Date and Time', ' X-Value', ' Y-Value', ' Z-Value', 'Label']]
    usual_walking_combined = usual_walking_combined[['Date and Time', ' X-Value', ' Y-Value', ' Z-Value', 'Label']]
    activities_with_name = activities_with_name[['Date and Time', ' X-Value', ' Y-Value', ' Z-Value', 'Label']]

    # Combine recumbent, walking, and activities datasets
    combined_data = pd.concat([recumbent_filtered, usual_walking_combined, activities_with_name], ignore_index=True)

    return combined_data



def load_and_filter_not_wearing_data(duration_limits=None):
    file_paths = {
        'faceDown': 'data/model/notWearing/faceDown.csv',
        'faceLeft': 'data/model/notWearing/faceLeft.csv',
        'faceRight': 'data/model/notWearing/faceRight.csv',
        'faceUp': 'data/model/notWearing/faceUp.csv'
    }
    
    # Default duration limits if not specified
    if duration_limits is None:
        duration_limits = {
            'faceDown': 150,
            'faceLeft': 120,
            'faceRight': 120,
            'faceUp': 150
        }

    combined_not_wearing = pd.DataFrame()

    for position, file_path in file_paths.items():
        try:
            temp_df = pd.read_csv(file_path)
            temp_df['Date and Time'] = pd.to_datetime(temp_df['Date and Time'])
            # Filter based on the specified duration limit for each position
            if position in duration_limits:
                start_time = temp_df['Date and Time'].min()
                max_duration = pd.Timedelta(minutes=duration_limits[position])
                end_time = start_time + max_duration
                temp_df = temp_df[(temp_df['Date and Time'] >= start_time) & (temp_df['Date and Time'] <= end_time)]
            temp_df['Label'] = position
            combined_not_wearing = pd.concat([combined_not_wearing, temp_df], ignore_index=True)
        except FileNotFoundError:
            print(f'{file_path} not found. Skipping...')

    return combined_not_wearing




# ----------------------------------------------------------------------------------------------------
# BELOW IS CODE TO COMBINE THE PROCESSED DATA




# Specify your duration limits here, if different from the defaults
custom_duration_limits = {
    'faceDown': 30,  
    'faceLeft': 20,  
    'faceRight': 20, 
    'faceUp': 30    
}

# combined_not_wearing_data_custom_limits = load_and_filter_not_wearing_data(custom_duration_limits)
combined_not_wearing_data = load_and_filter_not_wearing_data()

# Adjust the number of hours for recumbent data here (e.g., 2 hours)
# combined_data_wearing = load_and_filter_wearing_data(recumbent_hours=2)
combined_data_wearing = load_and_filter_wearing_data()

# For double checking purposes
combined_not_wearing_data.to_csv('combined_dataset_not_wearing.csv', index=False)

# For double checking purposes
combined_data_wearing.to_csv('combined_dataset_wearing.csv', index=False)

# Assign binary labels: 0 for not wearing, 1 for wearing
combined_not_wearing_data['Label'] = 0
combined_data_wearing['Label'] = 1

# Combine the two datasets
combined_df = pd.concat([combined_data_wearing, combined_not_wearing_data], ignore_index=True)

# For double checking purposes
combined_df.to_csv('combined_dataset.csv', index=False)




# ----------------------------------------------------------------------------------------------------
# BELOW IS THE CODE CREATE 5-MINUTES INTERVALS



combined_df['Date and Time'] = pd.to_datetime(combined_df['Date and Time'])

# Sort to ensure correct window grouping
combined_df.sort_values('Date and Time', inplace=True)

# Define a function to compute the vector magnitude
def vector_magnitude(row):
    return np.sqrt(row[' X-Value']**2 + row[' Y-Value']**2 + row[' Z-Value']**2)

# Compute vector magnitude for each row
combined_df['Vector_Magnitude'] = combined_df.apply(vector_magnitude, axis=1)

# Function to process each 5-minute window
def process_window(window):
    # Check if all labels are the same within the window; if not, return None
    if window['Label'].nunique() != 1:
        return None
    
    # Calculate mean of X, Y, Z
    means = window[[' X-Value', ' Y-Value', ' Z-Value']].mean().to_dict()
    
    # Temporarily add 'Time_Bin' column for 30-second interval grouping
    window = window.copy()
    window['Time_Bin'] = window['Date and Time'].dt.floor('30s')
    std_devs = window.groupby('Time_Bin')['Vector_Magnitude'].std().mean()
    
    return {
        'Mean_X': means[' X-Value'], 
        'Mean_Y': means[' Y-Value'], 
        'Mean_Z': means[' Z-Value'], 
        'Avg_Std_Dev_VM': std_devs, 
        'Label': window['Label'].iloc[0]
    }

# Group by 5-minute intervals and apply the processing function
processed_data = []
for _, window in combined_df.groupby(pd.Grouper(key='Date and Time', freq='5min')):
    result = process_window(window)
    if result:
        processed_data.append(result)

# Convert the processed data into a DataFrame
processed_df = pd.DataFrame(processed_data)

# For double checking purposes
processed_df.to_csv('processed_dataset.csv', index=False)




# ----------------------------------------------------------------------------------------------------
# BELOW IS CODE FOR MODEL DEVELOPMENT



# Split the Data into features (X) and labels (y)
X = processed_df[['Mean_X', 'Mean_Y', 'Mean_Z', 'Avg_Std_Dev_VM']].values
y = processed_df['Label'].values

# Split the data into training and testing sets
# 20% of the data is reserved for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess
# Here, we can experiment with scalers e.g.:
# scaler = MinMaxScaler()

# (In my opinion) StandardScaler is better because it is less sensisitve to outliers and are ideal for mean and standard deviation robustness in range
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Defining
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)), # 64 neurons with 1 feature set
    Dense(32, activation='relu'), # 32 neurons
    Dense(1, activation='sigmoid') # output layer 1
])

# Compiling
# Adaptive Moment Estimation used to minimize the loss
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training
# validation_split reserves a portion of the training data for validation to monitor the model's performance on unseen data during training.
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=10)

# Evaluating
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')



# ----------------------------------------------------------------------------------------------------
# BELOW IS FOR TESTING NEW DATA (UNCOMMENT FOR TESTING NEW DATA)





# new_df = pd.read_csv('')

# # Adjusted to include time columns
# def process_window_with_time(window):
#     # Check if all labels are the same within the window; if not, return None
#     if window['Label'].nunique() != 1:
#         return None
    
#     start_time = window['Date and Time'].min()
    
#     # Calculate mean of X, Y, Z
#     means = window[[' X-Value', ' Y-Value', ' Z-Value']].mean().to_dict()
    
#     # Temporarily add 'Time_Bin' column for 30-second interval grouping
#     window = window.copy()
#     window['Time_Bin'] = window['Date and Time'].dt.floor('30s') 
#     std_devs = window.groupby('Time_Bin')['Vector_Magnitude'].std().mean()
    
#     return {
#         'Start_Date_and_Time': start_time,
#         'Mean_X': means[' X-Value'], 
#         'Mean_Y': means[' Y-Value'], 
#         'Mean_Z': means[' Z-Value'], 
#         'Avg_Std_Dev_VM': std_devs, 
#         'Label': window['Label'].iloc[0]
#     }



# new_df['Date and Time'] = pd.to_datetime(new_df['Date and Time'])
# new_df.sort_values('Date and Time', inplace=True)
# new_df['Vector_Magnitude'] = new_df.apply(vector_magnitude, axis=1)
# new_processed_data = []
# for _, window in combined_df.groupby(pd.Grouper(key='Date and Time', freq='5min')):
#     result = process_window_with_time(window)
#     if result:
#         new_processed_data.append(result)

# new_processed_df = pd.DataFrame(new_processed_data)

# features = new_processed_df[['Mean_X', 'Mean_Y', 'Mean_Z', 'Avg_Std_Dev_VM']].values
# scaler = StandardScaler()
# scaled_features = scaler.fit_transform(features)
# predictions = model.predict(scaled_features)
# predicted_labels = (predictions > 0.5).astype(int).flatten()

# new_processed_df['Probability'] = predictions
# new_processed_df['Predicted_Label'] = predicted_labels

# new_processed_df.to_csv('predicted_dataset.csv', index=False)