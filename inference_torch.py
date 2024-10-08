import os
import numpy as np
import torch
from model_lite_cpu import PanguModel
from datetime import datetime, timedelta
# Use GPU or CPU
use_GPU = False

# The date and time of the initial field
# date = '2023-07-03'
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

final_result_dir = os.path.join(
    os.path.join(os.getcwd(), "results"),
    (date_time.strftime("%Y-%m-%d-%H-%M") + "to" + date_time_final.strftime("%Y-%m-%d-%H-%M"))
)
os.makedirs(final_result_dir,exist_ok=True)

temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir,exist_ok=True)
## copy forecast files to temp dir

model_24 = 'models/lite24_2007.pt' # 24h
model_6 = 'models/pangu_weather_6.onnx' # 6h
model_3 = 'models/pangu_weather_3.onnx' # 3h
model_1 = 'models/pangu_weather_1.onnx' # 1h



# The directory for forecasts
forecast_dir = os.path.join(
    os.path.join(os.getcwd(), "forecasts"),
    ## replace to prevent invaild char ":"
    date_time.strftime("%Y-%m-%d-%H-%M")
)
# Calculate the order of models should be used to generate the final result
time_difference_in_hour = (date_time_final - date_time).total_seconds() / 3600
current_date_time = date_time
last_date_time = None
model_used = None
start = True
ort_session = None
jump = False
while time_difference_in_hour >= 1:
    print(time_difference_in_hour)
    last_model = model_used
    if time_difference_in_hour >= 24:
        model_used = model_24
        time_difference_in_hour -= 24
        current_date_time += timedelta(hours=24)
        print("24")
    elif time_difference_in_hour >= 6:
        model_used = model_6
        time_difference_in_hour -= 6
        current_date_time += timedelta(hours=6)
        print("6")
    elif time_difference_in_hour >= 3:
        model_used = model_3
        time_difference_in_hour -= 3
        current_date_time += timedelta(hours=3)
        print("3")
    elif time_difference_in_hour >= 1:
        model_used = model_1
        time_difference_in_hour -= 1
        current_date_time += timedelta(hours=1)
        print("1")
    if model_used == last_model:
        jump = True
    else:
        jump = False
    print(current_date_time.strftime("%Y-%m-%d-%H-%M"))

    if not jump:
        # Load the model
        model = PanguModel()
        model.load_state_dict(torch.load(model_used))
        model.eval()

        # Initialize onnxruntime session for Pangu-Weather Models
        if use_GPU:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model.to(device)

    print("start")
    # Load the upper-air numpy arrays
    # Load the surface numpy arrays
    input = None
    input_surface = None
    if start:
        input = np.load(os.path.join(forecast_dir, 'input_upper.npy')).astype(np.float32)
        input_surface = np.load(os.path.join(forecast_dir, 'input_surface.npy')).astype(np.float32)
    else:
        input = np.load(os.path.join(final_result_dir, 'output_upper_'+last_date_time.strftime("%Y-%m-%d-%H-%M")+'.npy')).astype(np.float32)
        input_surface = np.load(os.path.join(final_result_dir, 'output_surface_'+last_date_time.strftime("%Y-%m-%d-%H-%M")+'.npy')).astype(np.float32)
    print('inference')
    # Run the inference session
    output, output_surface = model(input,input_surface)
    output, output_surface = output.detach().numpy(), output_surface.detach().numpy()
    # Save the results
    np.save(os.path.join(final_result_dir, 'output_upper_'+current_date_time.strftime("%Y-%m-%d-%H-%M")), output)
    np.save(os.path.join(final_result_dir, 'output_surface_' + current_date_time.strftime("%Y-%m-%d-%H-%M")), output_surface)
    last_date_time = current_date_time
    start = False
