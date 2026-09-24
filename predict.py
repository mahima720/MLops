import joblib
import pandas as pd

model = joblib.load("telecom_tower_model.pkl")

new_data = pd.DataFrame({
    'Temperature_C':[11.2],
        'Battery_Voltage':[55],
        'Power_Consumption_W':[0.5],
        'Signal_Strength_Percent':[88],
        'Fan_Speed_RPM':[65],
        'Humidity_Percent':[95],
        'Traffic_Load':[25],
        'Tower_Age_Years':[12]
    })

prediction = model.predict(new_data)

if prediction[0] == 1:
    print("Hardware Failure Predicted")
else:
    print("Tower is Healthy")
