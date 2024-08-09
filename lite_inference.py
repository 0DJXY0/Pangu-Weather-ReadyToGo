import os
import numpy as np
from datetime import datetime, timedelta



def Inference(input, input_surface, forecast_range):
    '''Inference code, describing the algorithm of inference using models with different lead times.
    PanguModel24, PanguModel6, PanguModel3 and PanguModel1 share the same training algorithm but differ in lead times.
    Args:
      input: input tensor, need to be normalized to N(0, 1) in practice
      input_surface: target tensor, need to be normalized to N(0, 1) in practice
      forecast_range: iteration numbers when roll out the forecast model
    '''

    # Load 4 pre-trained models with different lead times
    PanguModel24 = LoadModel(ModelPath24)

    # Load mean and std of the weather data
    weather_mean, weather_std, weather_surface_mean, weather_surface_std = LoadStatic()

    # Store initial input for different models
    input_24, input_surface_24 = input, input_surface

    # Using a list to store output
    output_list = []

    # Note: the following code is implemented for fast inference of [1,forecast_range]-hour forecasts -- if only one lead time is requested, the inference can be much faster.
    for i in range(forecast_range):
        # switch to the 24-hour model if the forecast time is 24 hours, 48 hours, ..., 24*N hours
        if (i + 1) % 24 == 0:
            # Switch the input back to the stored input
            input, input_surface = input_24, input_surface_24

            # Call the model pretrained for 24 hours forecast
            output, output_surface = PanguModel24(input, input_surface)

            # Restore from uniformed output
            output = output * weather_std + weather_mean
            output_surface = output_surface * weather_surface_std + weather_surface_mean

            # Stored the output for next round forecast
            input_24, input_surface_24 = output, output_surface

        else:
            print('forecast range is not divided by 24')

        # Stored the output for next round forecast
        input, input_surface = output, output_surface

        # Save the output
        output_list.append((output, output_surface))
    return output_list




# The date and time of the initial field
# date = '2023-07-03'
date_time = datetime(
    year=2023,
    month=7,
    day=11,
    hour=23,
    minute=0)
# time = '23:00'

# The date and time of the final approaches
date_time_final = datetime(
    year=2023,
    month=7,
    day=17,
    hour=11,
    minute=0)

final_result_dir = os.path.join(
    os.path.join(os.getcwd(), "results"),
    (date_time.strftime("%Y-%m-%d-%H-%M") + "to" + date_time_final.strftime("%Y-%m-%d-%H-%M"))
)
os.makedirs(final_result_dir,exist_ok=True)

temp_dir = os.path.join(os.getcwd(), "temp")
os.makedirs(temp_dir,exist_ok=True)
## copy forecast files to temp dir

# The directory for forecasts
forecast_dir = os.path.join(
    os.path.join(os.getcwd(), "forecasts"),
    ## replace to prevent invaild char ":"
    date_time.strftime("%Y-%m-%d-%H-%M")
)
target_dir = os.path.join(
    os.path.join(os.getcwd(), "forecasts"),
    ## replace to prevent invaild char ":"
    date_time_final.strftime("%Y-%m-%d-%H-%M")
)

# Calculate the order of models should be used to generate the final result
forecast_range = (date_time_final - date_time).total_seconds() / 3600

print("start")
# Load the upper-air numpy arrays
# Load the surface numpy arrays

input = np.load(os.path.join(forecast_dir, 'input_upper.npy')).astype(np.float32)
input_surface = np.load(os.path.join(forecast_dir, 'input_surface.npy')).astype(np.float32)
target = np.load(os.path.join(target_dir, 'input_upper.npy')).astype(np.float32)
target_surface = np.load(os.path.join(target_dir, 'input_surface.npy')).astype(np.float32)


# Run the inference session
output, output_surface = Inference(input, input_surface, forecast_range)
# Save the results
np.save(os.path.join(final_result_dir, 'output_upper_'+ date_time_final.strftime("%Y-%m-%d-%H-%M")), output)
np.save(os.path.join(final_result_dir, 'output_surface_' + date_time_final.strftime("%Y-%m-%d-%H-%M")), output_surface)

