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
import datetime
import numpy as np
from sklearn.ensemble import IsolationForest

########################
# Function Definitions #
########################

def calculate_anomaly_scores(max_Fc_or_max_Fchd):
    # Convert the dictionary values to a numpy array
    X = np.array(list(max_Fc_or_max_Fchd.values())).reshape(-1, 1)

    # Fit the Isolation Forest model
    clf = IsolationForest(random_state=0)
    clf.fit(X)

    # Compute anomaly scores
    anomaly_scores = clf.decision_function(X)

    # Create a dictionary with the anomaly scores
    anomaly_scores_dict = {key: score for key, score in zip(max_Fc_or_max_Fchd.keys(), anomaly_scores)}

    return anomaly_scores_dict



def entropy(series):
    p = series.value_counts(normalize=True)
    return -np.sum(p * np.log2(p))

def calculate_weights(series1, series2):
    entropy1 = entropy(series1)
    entropy2 = entropy(series2)

    weight1 = (1 - entropy1) / (2 - entropy1 - entropy2)
    weight2 = (1 - entropy2) / (2 - entropy1 - entropy2)

    return weight1, weight2

def combine_columns(max_Fc, max_Fchd):
    keys = max_Fc.keys()
    result = {}
    for key in keys:
        weight1, weight2 = calculate_weights(pd.Series([max_Fc[key]]), pd.Series([max_Fchd[key]]))
        combined_value = max_Fc[key] * weight1 + max_Fchd[key] * weight2
        result[key] = combined_value
    return result

# Calculate ΔC and ΔH:
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
    

def generate_png_vessel_course(time_windows_df):
    # Create a map
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    #ax.set_extent([-180, 180, -90, 90])

    """
    # Add features to the map
    ax.add_feature(cfeature.COASTLINE, edgecolor='gray')
    ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='gray')
    ax.add_feature(cfeature.RIVERS, edgecolor='blue')
    """
    
    # Plot vessel course
    for mmsi in time_windows_df['MMSI'].unique():
        vessel_data = time_windows_df[time_windows_df['MMSI'] == mmsi]
        ax.plot(vessel_data['LON'], vessel_data['LAT'], marker='o', markersize=2, linewidth=1, label=f'MMSI: {mmsi}')

    # Add latitude and longitude gridlines
    ax.gridlines(draw_labels=True, linestyle='--')

    """
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
    """
    # Show the map
    plt.show()



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



##########################################################
# 1.Load and Order the AIS dataset by MMSI and TimeStamp #
##########################################################
    
# Specify the columns to be loaded
columns_to_load = ['LAT', 'LON', 'MMSI', 'BaseDateTime', 'VesselType', 'Status', 'SOG', 'COG', 'Heading', 'label']

# Example usage
file_path = "v3_jan_jul_2021_all_dataset_res30_3days_stats.csv"
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

###########################################
# 2. Calculate ΔC and ΔH of the equations #
###########################################
# Apply the function to add the columns to the original DataFrame
df_calculations = add_calculated_columns(df_sorted)

# Display the first few rows of the new DataFrame
print(df_calculations.head())

date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

# Save the DataFrame to a CSV file
file_name = f"vessel_calculations_{date_time}.csv"
df_calculations.to_csv(file_name, index=False)

# Print the file name for confirmation
print(f"DataFrame saved to '{file_name}'")


# Define the time window duration in hours
time_window_hours = 24

# Convert 'BaseDateTime' column to datetime format
df_calculations['BaseDateTime'] = pd.to_datetime(df_calculations['BaseDateTime'], unit='s')

# Initialize an empty DataFrame to store the results
time_windows_df = pd.DataFrame()




# Initialize dictionaries to store the maximum values of Fc and Fchd for each MMSI
max_Fc = {}
max_Fchd = {}

##############################
# 3. Iterate over all MMSI’s #
##############################

# Iterate over each unique MMSI in df_calculations
for mmsi in df_calculations['MMSI'].unique():
    # Initialize variavles for Geodesoc Distance calculation.
    initial_dist = 0
    cumulated_dist = 0
    
    mmsi_data = df_calculations[df_calculations['MMSI'] == mmsi].copy()
    mmsi_data.sort_values('BaseDateTime', inplace=True)
    mmsi_data.reset_index(drop=True, inplace=True)

    # Check if the MMSI data has at least two timestamps
    if len(mmsi_data) < 2:
        print(f"Skipping MMSI {mmsi}. Insufficient data points for time window calculation.")
        continue

    # Calculate the time difference between the first and last timestamp
    time_diff_hours = (mmsi_data['BaseDateTime'].iloc[-1] - mmsi_data['BaseDateTime'].iloc[0]).total_seconds() / 3600

    # Check if the time difference is greater than the time window
    if time_diff_hours < time_window_hours:
        print(f"Skipping MMSI {mmsi}. Time difference between first and last timestamp is less than {time_window_hours} hours.")
        continue

    # Initialize variables to store the maximum Fc and Fchd for the current MMSI
    max_Fc_mmsi = 0
    max_Fchd_mmsi = 0
    
    

    ##############################################################################
    # 4 Slide the time window across all the data of the MMSI. Calculate B and G #
    ##############################################################################

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
            #generate_png_vessel_course(time_windows_df)
            
            ########################################################
            # 5.With ΔC, ΔH, B and G, calculate F(c) and F (c,h,d) #
            ########################################################
            
            # Calculate Area
            B = Area_Nm2(time_windows_df)
            
            # Calculate Geodesic Distance
            G, init_dist = GeoDist_Nm(time_windows_df, initial_dist, cumulated_dist)
            #print(f"Geodesic Distance: {G}")
            cumulated_dist = G 
            initial_dist = init_dist
            
            # Calculate Fc
            Fc = Func_c(time_windows_df, B)
            #print(f"Fc Loitering : {Fc}")
            # calculate Fcdh
            Fchd = Func_chd(time_windows_df, B, G)
            #print(f"Fchd Loitering : {Fchd}")
            
            ##########################################################
            # 5. choose the biggest  F(c) and F(c,h,d). and store it #
            ##########################################################
                        
            # Update max_Fc and max_Fchd values for the current MMSI
            max_Fc_mmsi = max(max_Fc_mmsi, Fc)
            max_Fchd_mmsi = max(max_Fchd_mmsi, Fchd)
            
            time_windows_df = pd.DataFrame() #cleaning df
        else:
            print(f"Sliding Time window for MMSI {mmsi} end.")
            break
        
    
    ##################################################
    # 6. Store dictionary with all F(c) and F(c,h,d) #
    ##################################################
    
    # Store the maximum Fc and Fchd for the current MMSI
    max_Fc[mmsi] = max_Fc_mmsi
    max_Fchd[mmsi] = max_Fchd_mmsi

print("max_Fc:")
for key, value in list(max_Fc.items())[:5]:
    print(f"{key}: {value}")

# Print the first few lines of max_Fchd
print("\nmax_Fchd:")
for key, value in list(max_Fchd.items())[:5]:
    print(f"{key}: {value}")

###################################################################   
# 7. Isolation Forest ovet all F(c) and F(c,h,d). get sf1 and sf2 #
###################################################################

# calculate the anomaly score of max_Fc & max_Fchd dictionaries
anomaly_scores_max_Fc = calculate_anomaly_scores(max_Fc)
anomaly_scores_max_Fchd = calculate_anomaly_scores(max_Fchd)

print("anomaly_scores_max_Fc:")
for key, value in list(anomaly_scores_max_Fc.items())[:5]:
    print(f"{key}: {value}")



#################################################################
# 8. Entropy Weight Method (EWM) over sf1 and sf2. get I(f1,f2) #
#################################################################

# Combine anomaly_scores_max_Fc and anomaly_scores_max_Fchd list of anomalies scores using Entropy Weight Method
EWM_anomaly_scores_max_Fc_max_Fchd = combine_columns(anomaly_scores_max_Fc, anomaly_scores_max_Fchd)

# Get the current date and time
date_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

###############################################
# 9. Store and rank sf1 and sf2. get I(f1,f2) #
###############################################

# Combine the dictionaries and anomaly scores into a single DataFrame
df_dict = pd.DataFrame({
    'MMSI': list(max_Fc.keys()),
    'Max_Fc': list(max_Fc.values()),
    'Max_Fchd': list(max_Fchd.values()),
    'Anomaly_Score_Max_Fc': list(anomaly_scores_max_Fc.values()),
    'Anomaly_Score_Max_Fchd': list(anomaly_scores_max_Fchd.values()),
    'EWM_Anomaly_Score_Max_Fc_Max_Fchd': list(EWM_anomaly_scores_max_Fc_max_Fchd.values())
})

# Save the DataFrame to a CSV file
file_name = f"Loitering_indexes_{date_time}.csv"
df_dict.to_csv(file_name, index=False)

# Print the file name for confirmation
print(f"Dictionary saved to '{file_name}'")


# Sort the DataFrame by 'EWM_Anomaly_Score_Max_Fc_Max_Fchd' in ascending order
sorted_df = df_dict.sort_values(by='EWM_Anomaly_Score_Max_Fc_Max_Fchd', ascending=True)

# Print the 5 most negative rows
print("5 most negative rows based on 'EWM_Anomaly_Score_Max_Fc_Max_Fchd':")
print(sorted_df[['MMSI', 'Max_Fc', 'Max_Fchd', 'Anomaly_Score_Max_Fc', 'Anomaly_Score_Max_Fchd', 'EWM_Anomaly_Score_Max_Fc_Max_Fchd']].head(5))

# Print the 5 most positive rows
print("\n5 most positive rows based on 'EWM_Anomaly_Score_Max_Fc_Max_Fchd':")
print(sorted_df[['MMSI', 'Max_Fc', 'Max_Fchd', 'Anomaly_Score_Max_Fc', 'Anomaly_Score_Max_Fchd', 'EWM_Anomaly_Score_Max_Fc_Max_Fchd']].tail(5))
