import pandas as pd

def load_and_filter_data(recumbent_hours=None):
    # Load ActivitiesSorted.csv and filter for rows with activities
    activities_sorted = pd.read_csv('data/wearing/ActivitiesSorted.csv')
    activities_with_name = activities_sorted.dropna(subset=['Activity']).copy()
    activities_with_name['Label'] = activities_with_name['Activity']
    activities_with_name.drop('Activity', axis=1, inplace=True)

    # Load Cleared_Recumbent1.csv and filter for recumbent (1)
    cleared_recumbent = pd.read_csv('data/wearing/Cleared_Recumbent1.csv')
    recumbent_filtered = cleared_recumbent[cleared_recumbent['Recumbent'] == 1].copy()
    recumbent_filtered['Label'] = 'recumbent'
    
    # Adjust the number of hours of recumbent data if specified
    if recumbent_hours is not None:
        recumbent_filtered['Date and Time'] = pd.to_datetime(recumbent_filtered['Date and Time'])
        start_time = recumbent_filtered['Date and Time'].min()
        end_time = start_time + pd.Timedelta(hours=recumbent_hours)
        recumbent_filtered = recumbent_filtered[(recumbent_filtered['Date and Time'] >= start_time) & (recumbent_filtered['Date and Time'] <= end_time)]

    # Initialize an empty DataFrame for usual walking data
    usual_walking_combined = pd.DataFrame()

    # Load and filter Cleared_UsualWalking CSVs
    for i in range(1, 6):  # versions 1 through 5
        try:
            usual_walking = pd.read_csv(f'data/wearing/Cleared_UsualWalking{i}.csv')
            walking_filtered = usual_walking[usual_walking['Usual Walking'] == 1].copy()
            walking_filtered['Label'] = 'walking'
            usual_walking_combined = pd.concat([usual_walking_combined, walking_filtered], ignore_index=True)
        except FileNotFoundError:
            print(f'Cleared_UsualWalking{i}.csv not found. Skipping...')

    # Ensure all dataframes have the same columns before concatenation
    recumbent_filtered = recumbent_filtered[['Date and Time', ' X-Value', ' Y-Value', ' Z-Value', 'Label']]
    usual_walking_combined = usual_walking_combined[['Date and Time', ' X-Value', ' Y-Value', ' Z-Value', 'Label']]
    activities_with_name = activities_with_name[['Date and Time', ' X-Value', ' Y-Value', ' Z-Value', 'Label']]

    # Combine recumbent, walking, and activities datasets
    combined_data = pd.concat([recumbent_filtered, usual_walking_combined, activities_with_name], ignore_index=True)

    return combined_data

# Adjust the number of hours for recumbent data here (e.g., 2 hours)
combined_data_wearing = load_and_filter_data(recumbent_hours=2)

combined_data_wearing.to_csv('combined_dataset_wearing.csv', index=False)

print(combined_data_wearing.head())
