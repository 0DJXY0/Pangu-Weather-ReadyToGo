from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from datetime import datetime, timedelta
import sys
import os

date_time = datetime(
    year=2023,
    month=7,
    day=10,
    hour=23,
    minute=0)
# time = '23:00'

# The date and time of the final approaches
date_time_final = datetime(
    year=2023,
    month=7,
    day=11,
    hour=23,
    minute=0)

pred_dir = os.path.join(
    os.path.join(os.getcwd(), "results"),
    (date_time.strftime("%Y-%m-%d-%H-%M") + "to" + date_time_final.strftime("%Y-%m-%d-%H-%M"))
)

label_dir = os.path.join(
    os.path.join(os.getcwd(), "forecasts"),
    date_time_final.strftime("%Y-%m-%d-%H-%M")
)

label = np.load(os.path.join(label_dir, 'input_upper.npy')).astype(np.float32)
label_surface = np.load(os.path.join(label_dir, 'input_surface.npy')).astype(np.float32)

pred = np.load(os.path.join(final_result_dir, 'output_upper_'+current_date_time.strftime("%Y-%m-%d-%H-%M"))).astype(np.float32)
pred_surface = np.load(os.path.join(final_result_dir, 'output_surface_' + current_date_time.strftime("%Y-%m-%d-%H-%M"))).astype(np.float32)

print("MSE: ",mean_squared_error(label[0,:,:],pred[0,:,:]))
print("MAPE: ",mean_absolute_percentage_error(label[0,:,:],pred[0,:,:]))
print("MSE_surface: ",mean_squared_error(label_surface[0,:,:],pred_surface[0,:,:]))
print("MAPE_surface: ",mean_absolute_percentage_error(label_surface[0,:,:],pred_surface[0,:,:]))
