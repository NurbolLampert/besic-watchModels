# besic-watchModels
## Fiji's binary model folder (/binaryClassifierFiji):
main.py -> accelReading.py (the json file is the data) -> classifierTraining.py (using scikit-learn, numpy, pandas)

## Nurb's binary model:

### In main.py
Loading -> Filtering -> Processing into Windows -> Model Development -> Testing
Loss: 0.087
Accuracy: 0.964

### data/wearing: 
 - Recumbent ~ 9 hours (in main.py in the load_and_filter function, the number of hours used for the model can be adjusted e.g recumbent_hours=2)
 - Walking ~ 2 hours and 40 minutes
 - Standing = 25 minutes
 - Sitting = 25 minutes
 - Going Upstairs = 25 minutes
 - Going Downstairs = 25 minutes
 - Brushing Teeth = 25 minutes
 - Object Picking = 25 minutes
 - Total: ~ 14 hours and 10 minutes (can be adjusted hence recumbent state could be adjusted to 4 hours = 9 hours and 10 minutes total)

### data/notWearing
 - Watch face up ~ 2 hours and 30 minutes
 - Watch face down ~ 2 hours and 30 minutes
 - Watch face left ~ 2 hours
 - Watch face right ~ 2 hours
 - Total:  ~ 9 hours