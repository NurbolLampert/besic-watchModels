import pandas as pd
from datetime import datetime

file_path = 'rawNotWearing/PT-DEMO_Accelerometer_29(left-right).csv'  
df = pd.read_csv(file_path)

df['Date and Time'] = pd.to_datetime(df['Date and Time'], format='%Y/%m/%d %H:%M:%S:%f')

start_time = datetime.strptime('2024/02/13 01:30:00:000', '%Y/%m/%d %H:%M:%S:%f')
end_time = datetime.strptime('2024/02/13 03:30:00:000', '%Y/%m/%d %H:%M:%S:%f')

filtered_df = df[(df['Date and Time'] >= start_time) & (df['Date and Time'] <= end_time)]

filtered_df['Label'] = 'faceRight'

output_file_path = 'data/notWearing/faceRight.csv'  
filtered_df.to_csv(output_file_path, index=False)

print(f'Filtered data saved to {output_file_path}')
