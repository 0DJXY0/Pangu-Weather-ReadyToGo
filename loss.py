from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
from datetime import datetime, timedelta
import torch
import os

date_time = datetime(
    year=2007,
    month=2,
    day=1,
    hour=0,
    minute=0)
# time = '23:00'

# The date and time of the final approaches
date_time_final = datetime(
    year=2007,
    month=2,
    day=2,
    hour=0,
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

pred = np.load(os.path.join(pred_dir, 'output_upper_'+ (date_time_final.strftime("%Y-%m-%d-%H-%M")) + ".npy")).astype(np.float32)
pred_surface = np.load(os.path.join(pred_dir, 'output_surface_' + (date_time_final.strftime("%Y-%m-%d-%H-%M")) + ".npy")).astype(np.float32)

label = torch.from_numpy(label)
pred = torch.from_numpy(pred)
label_surface = torch.from_numpy(label_surface)
pred_surface = torch.from_numpy(pred_surface)
print("MAPE_upper: ",torch.norm(pred-label,p='fro')/torch.norm(label,p='fro'))
print("MAPE_surface: ",torch.norm(pred_surface-label_surface,p='fro')/torch.norm(label_surface,p='fro'))

# print("MSE_upper: ",mean_squared_error(label[0,:,:],pred[0,:,:]))
# print("MAPE_upper: ",mean_absolute_percentage_error(label[0,:,:],pred[0,:,:]))
# print("MSE_surface: ",mean_squared_error(label_surface[0,:,:],pred_surface[0,:,:]))
# print("MAPE_surface: ",mean_absolute_percentage_error(label_surface[0,:,:],pred_surface[0,:,:]))
