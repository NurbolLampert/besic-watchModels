# besic-watchModels
## Fiji's binary model folder (/binaryClassifierFiji):
main.py -> accelReading.py (the json file is the data) -> classifierTraining.py (using scikit-learn)

## Nurb's binary model:

### In main.py

### data/wearing: 
Recumbent ~ 9 hours (in main.py in the load_and_filter function, the number of hours used for the model can be adjusted e.g recumbent_hours=2)
Walking ~ 2 hours and 40 minutes
Standing = 25 minutes
Sitting = 25 minutes
Going Upstairs = 25 minutes
Going Downstairs = 25 minutes
Brushing Teeth = 25 minutes
Object Picking = 25 minutes

### data/notWearing