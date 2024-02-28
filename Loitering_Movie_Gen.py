# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:08:07 2024

@author: JJARPA
"""

import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage


def add_calculated_columns(df):
    df['cog_change'] = df['COG'].diff().abs()
    df['delta_cog_heading'] = (df['COG'] - df['Heading']).abs()
    return df

def read_excel_file(file_path, columns):
    try:
        df = pd.read_csv(file_path, usecols=columns)
        return df
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    
    
def plot_vessel_course(time_windows_df, B, G, Fc, Fchd, max_lat, min_lat, max_lon, min_lon):
    # Create a map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(8, 6))

    # Set the extent of the plot based on the maximum and minimum 'LAT' and 'LON' values
    ax.set_extent([min_lon, max_lon, min_lat, max_lat])

    # Add features to the map
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')

    # Plot vessel course
    for i in range(len(time_windows_df)-1):
        start_point = (time_windows_df['LON'].iloc[i], time_windows_df['LAT'].iloc[i])
        end_point = (time_windows_df['LON'].iloc[i+1], time_windows_df['LAT'].iloc[i+1])
        if i == len(time_windows_df)-2:
            # Plot red arrow for the last point
            ax.annotate('', xy=end_point, xytext=start_point, arrowprops=dict(
                arrowstyle="fancy", 
                color='red',
                lw=2)
                , size=20)
        else:
            # Plot blue marker for intermediate points
            ax.plot(end_point[0], end_point[1], marker='.', color='blue', markersize=1)

    # Plot green square at the first point
    ax.plot(time_windows_df['LON'].iloc[0], time_windows_df['LAT'].iloc[0], marker='s', markersize=8, color='green')

    # Add latitude and longitude gridlines
    ax.gridlines(draw_labels=True, linestyle='--')

    # Add country borders and names
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, edgecolor='blue')
    ax.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
    ax.add_feature(cfeature.COASTLINE, edgecolor='black')
    ax.add_feature(cfeature.RIVERS)

    # Add legend
    ax.legend()

    # Set title
    ax.set_title(f"Vessel Course from {time_windows_df['BaseDateTime'].iloc[0]} to {time_windows_df['BaseDateTime'].iloc[-1]}\nB: {round(B, 1)}, G: {round(G, 1)}, Fc: {round(Fc, 1)}, Fchd: {round(Fchd, 1)}")

    return fig



def Area_Nm2(df):
    # Calculate the latitude and longitude differences
    lat_diff = df['LAT'].max() - df['LAT'].min()
    lon_diff = df['LON'].max() - df['LON'].min()
    
    # Convert latitude and longitude differences to nautical miles
    lat_diff_nm = lat_diff * 60  # 1 degree = 60 Nautical Miles
    lon_diff_nm = lon_diff * 60  # 1 degree = 60 Nautical Miles
    
    # Calculate the area in square nautical miles
    B = lat_diff_nm * lon_diff_nm
    
    return B

def GeoDist_Nm(df, initial_dist, cumulated_dist=0):
    # Calculate the initial distance if not provided
    
    start_lat, start_lon = df.iloc[0]['LAT'], df.iloc[0]['LON']
    end_lat, end_lon = df.iloc[1]['LAT'], df.iloc[1]['LON']
    init_dist = geodesic((start_lat, start_lon), (end_lat, end_lon)).nautical

    # Calculate the Geodesic Distance
    G = cumulated_dist
    if cumulated_dist == 0:
        for i in range(len(df) - 1):
            start_lat, start_lon = df.iloc[i]['LAT'], df.iloc[i]['LON']
            end_lat, end_lon = df.iloc[i + 1]['LAT'], df.iloc[i + 1]['LON']
            G += geodesic((start_lat, start_lon), (end_lat, end_lon)).nautical
    else:
        start_lat, start_lon = df.iloc[-2]['LAT'], df.iloc[-2]['LON']
        end_lat, end_lon = df.iloc[-1]['LAT'], df.iloc[-1]['LON']
        G += geodesic((start_lat, start_lon), (end_lat, end_lon)).nautical - initial_dist

    return G, init_dist

    

def Func_c(df, B):
    """
    The frequent change of courses in a loitering trajectory may include minor
    ones to the relatively large change of courses that occur in extreme 
    turnings. The rate of course change can be described by comparing ΔC with 
    the maximum course change (180°). Considering the area of the bounding box
    B enclosing the loitering trajectory, the speed S of the ship, and the rate
    of course change, the score of loitering F(c) can be expressed as Eq. (6).
    The unit for B and S are Nm2 and knots respectively. Here, the score of 
    loitering F(c) is proportional to the rate of course change and inversely
    proportional to the area of the enclosing bounding box.

    Parameters
    ----------
    df : Pandas Dataframe
        Frame with 24 hours (configurable) of vessel info: SOG,COG, LAT, LONG,
        ΔCOG,BaseTimeDate.

    Returns
    -------
    result : Float
        Loitering Detection Parameter.

    """
        
    # Calculate the result
    result = (df['cog_change'].sum() * df['SOG'].sum()) / (180 * B)
    
    return result


def Func_chd(df, B, G):
            
    result = (df['cog_change'].sum() * df['delta_cog_heading'].sum() * df['SOG'].sum()) / (B * G)
    
    return result



    
# Specify the columns to be loaded
columns_to_load = ['LAT', 'LON', 'MMSI', 'BaseDateTime', 'VesselType', 'Status', 'SOG', 'COG', 'Heading', 'label']

# Example usage
file_path = "vessel_calculations_2024_02_23_14_53_47.csv"
df = read_excel_file(file_path, columns_to_load)

if df is not None:
    print("Data loaded successfully!")
    # Sort the DataFrame by 'MMSI' and 'BaseDateTime'
    df_sorted = df.sort_values(by=['MMSI', 'BaseDateTime'])
    print(df_sorted.head())  # Display the first few rows of the sorted DataFrame
    # Count unique MMSI values
    unique_mmsi_count = df_sorted['MMSI'].nunique()
    print(f"Number of unique MMSI elements: {unique_mmsi_count}")
    
    # Count unique MMSI values with Status = 1
    unique_mmsi_status_1_count = df_sorted[df_sorted['Status'] == 1]['MMSI'].nunique()
    print(f"Number of unique MMSI elements with Status = 1: {unique_mmsi_status_1_count}")
    
    # Count total different values of "label" column
    unique_label_count = df_sorted['label'].nunique()
    print(f"Number of unique values in 'label' column: {unique_label_count}")
    
    # Count the quantity of each label value
    label_counts = df_sorted['label'].value_counts()
    print("Quantity of each label value:")
    print(label_counts)
    
    # Count the number of different MMSI values for each unique label value
    for label_value in df_sorted['label'].unique():
        mmsi_count_for_label = df_sorted[df_sorted['label'] == label_value]['MMSI'].nunique()
        print(f"Number of different MMSI values for label '{label_value}': {mmsi_count_for_label}")
else:
    print("Failed to load data.")



# Apply the function to add the columns to the original DataFrame
df_calculations = add_calculated_columns(df_sorted)

# Display the first few rows of the new DataFrame
print(df_calculations.head())

# Define the time window duration in hours
time_window_hours = 12

# Convert 'BaseDateTime' column to datetime format
df_calculations['BaseDateTime'] = pd.to_datetime(df_calculations['BaseDateTime'], unit='s')


# Initialize dictionaries to store the maximum values of Fc and Fchd for each MMSI
max_Fc = {}
max_Fchd = {}

def process_mmsi(mmsi, df_calculations, time_window_hours):
    # Initialize an empty DataFrame to store the results
    time_windows_df = pd.DataFrame()
    
    # Initialize variables for Geodesoc Distance calculation.
    initial_dist = 0
    cumulated_dist = 0

    mmsi_data = df_calculations[df_calculations['MMSI'] == mmsi].copy()
    mmsi_data.sort_values('BaseDateTime', inplace=True)
    mmsi_data.reset_index(drop=True, inplace=True)
    
    # Get the maximum and minimum 'LAT' and 'LON' values
    max_lat = mmsi_data['LAT'].max()
    min_lat = mmsi_data['LAT'].min()
    max_lon = mmsi_data['LON'].max()
    min_lon = mmsi_data['LON'].min()

    # Check if the MMSI data has at least two timestamps
    if len(mmsi_data) < 2:
        print(f"Skipping MMSI {mmsi}. Insufficient data points for time window calculation.")
        return None, None

    # Calculate the time difference between the first and last timestamp
    time_diff_hours = (mmsi_data['BaseDateTime'].iloc[-1] - mmsi_data['BaseDateTime'].iloc[0]).total_seconds() / 3600

    # Check if the time difference is greater than the time window
    if time_diff_hours < time_window_hours:
        print(f"Skipping MMSI {mmsi}. Time difference between first and last timestamp is less than {time_window_hours} hours.")
        return None, None

    # Initialize variables to store the maximum Fc and Fchd for the current MMSI
    max_Fc_mmsi = 0
    max_Fchd_mmsi = 0

    for i in range(len(mmsi_data)):
        current_timestamp = mmsi_data.loc[i, 'BaseDateTime']
        end_timestamp = current_timestamp + pd.Timedelta(hours=time_window_hours)

        time_window_data = mmsi_data[(mmsi_data['BaseDateTime'] >= current_timestamp) &
                                      (mmsi_data['BaseDateTime'] < end_timestamp)].copy()

        # Check if the time window is valid (difference between first and last timestamp is at least 24 hours)
        if (time_window_data['BaseDateTime'].iloc[-1] - time_window_data['BaseDateTime'].iloc[0]) >= pd.Timedelta(hours=time_window_hours - 1): # 1 hour of tolerance
            time_window_data['WindowStart'] = current_timestamp
            time_window_data['WindowEnd'] = end_timestamp
            
            time_windows_df = pd.concat([time_windows_df, time_window_data], ignore_index=True)
            #print(time_windows_df.head())
            #print(len(time_windows_df))
            # Plotting Vessel

            # Calculate Area
            B = Area_Nm2(time_window_data)

            # Calculate Geodesic Distance
            G, init_dist = GeoDist_Nm(time_window_data, initial_dist, cumulated_dist)
            cumulated_dist = G
            initial_dist = init_dist

            Fc = Func_c(time_window_data, B)

            Fchd = Func_chd(time_window_data, B, G)

            # Update max_Fc and max_Fchd values for the current MMSI
            max_Fc_mmsi = max(max_Fc_mmsi, Fc)
            max_Fchd_mmsi = max(max_Fchd_mmsi, Fchd)
            
            # Generate the plot for the current time window
            fig = plot_vessel_course(time_windows_df, B, G, Fc, Fchd, max_lat, min_lat, max_lon, min_lon)
            # Convert the plot to a numpy array
            frame = mplfig_to_npimage(fig)
            plt.close(fig)
            # Append the frame to the list of frames
            frames.append(frame)
            time_windows_df = pd.DataFrame() #cleaning df
        else:
            print(f"Sliding Time window for MMSI {mmsi} end.")
            break

    return max_Fc_mmsi, max_Fchd_mmsi


# Create a list to store each frame
frames = []

# Example usage
mmsi = 319200700    # Specify the MMSI you want to process
max_Fc_mmsi, max_Fchd_mmsi = process_mmsi(mmsi, df_calculations, time_window_hours)

if max_Fc_mmsi is not None and max_Fchd_mmsi is not None:
    print(f"MMSI: {mmsi}, Max Fc: {max_Fc_mmsi}, Max Fchd: {max_Fchd_mmsi}")

# Create a video clip from the list of frames
fps = 24  # Frames per second
clip = VideoClip(lambda t: frames[min(int(t*fps), len(frames)-1)], duration=len(frames)/fps)

# Save the video clip
clip.write_videofile('vessel_course.mp4', fps=fps)
