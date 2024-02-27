# LoiteringDetector
This project is to devise a strategy to identify vessels that are loitering in a specific area known for smuggling. The approach is based on "Loitering behavior detection by spatiotemporal characteristics quantification based on the dynamic features of Automatic identification System (AIS) messages." by Wayan Mahardhika Wijaya and Yasuhiro Nakamura

In order to test the program:
1. Download all files in the same directory.
2. unzip 'v3_jan_jul_2021_all_dataset_res30_3days_stats.zip'.
3. open 'Loitering_Detector.py'.
4. Configure the variable:
5. time_window_hours = 24 (12, 24 or 36 hours)
6. RUN. You will obtain a console output like this:
   
Data loaded successfully!
        MMSI  BaseDateTime  VesselType  Status  ...   SOG    COG  Heading  label
0  211311970    1625316226          70       0  ...  18.4  160.1    162.0      1
1  211311970    1625316291          70       0  ...  18.5  159.2    162.0      1
2  211311970    1625316357          70       0  ...  18.4  159.1    162.0      1
3  211311970    1625316423          70       0  ...  18.5  159.9    163.0      1
4  211311970    1625316490          70       0  ...  18.5  158.9    162.0      1

[5 rows x 10 columns]
Number of unique MMSI elements: 137
Number of unique MMSI elements with Status = 1: 0
Number of unique values in 'label' column: 2
Quantity of each label value:
label
 1    91603
-1    62340
Name: count, dtype: int64
Number of different MMSI values for label '1': 112
Number of different MMSI values for label '-1': 25
        MMSI  BaseDateTime  VesselType  ...  label  cog_change  delta_cog_heading
0  211311970    1625316226          70  ...      1         NaN                1.9
1  211311970    1625316291          70  ...      1         0.9                2.8
2  211311970    1625316357          70  ...      1         0.1                2.9
3  211311970    1625316423          70  ...      1         0.8                3.1
4  211311970    1625316490          70  ...      1         1.0                3.1

[5 rows x 12 columns]
DataFrame saved to 'vessel_calculations_2024_02_26_11_38_26.csv'
Skipping MMSI 211311970. Time difference between first and last timestamp is less than 24 hours.
Sliding Time window for MMSI 211327410 end.
Sliding Time window for MMSI 211335760 end.
Sliding Time window for MMSI 215071000 end.
Sliding Time window for MMSI 219031000 end.
Skipping MMSI 219155000. Time difference between first and last timestamp is less than 24 hours.
Skipping MMSI 220415000. Time difference between first and last timestamp is less than 24 hours.
........
Skipping MMSI 636090967. Time difference between first and last timestamp is less than 24 hours.
Skipping MMSI 636091959. Time difference between first and last timestamp is less than 24 hours.
max_Fc:
211327410: 564.1014812973774
211335760: 361.66151317743146
215071000: 3.9544335376954116
219031000: 0
220593000: 1374.2619555112494

max_Fchd:
211327410: 417076085.79720855
211335760: 155669829.10367316
215071000: 6699.901962041226
219031000: 0
220593000: 873617051.8396381
anomaly_scores_max_Fc:
211327410: -0.08175186734594309
211335760: -0.04898182812653673
215071000: 0.16609184498976992
219031000: 0.16065462762119842
220593000: 0.01828600914472539
Dictionary saved to 'Loitering_indexes_2024_02_26_11_44_41.csv'
5 most negative rows based on 'EWM_Anomaly_Score_Max_Fc_Max_Fchd':
         MMSI  ...  EWM_Anomaly_Score_Max_Fc_Max_Fchd
46  565747000  ...                          -0.236336
32  373932000  ...                          -0.231074
23  357051000  ...                          -0.203917
9   303352000  ...                          -0.157054
35  431496000  ...                          -0.153165

[5 rows x 6 columns]

5 most positive rows based on 'EWM_Anomaly_Score_Max_Fc_Max_Fchd':
         MMSI  ...  EWM_Anomaly_Score_Max_Fc_Max_Fchd
13  319200700  ...                           0.179389
14  319819000  ...                           0.179462
52  636019627  ...                           0.179715
33  374077000  ...                           0.179817
12  311000421  ...                           0.179875

[5 rows x 6 columns]
7. Note that the program generated two files: 'vessel_calculations_2024_02_26_11_38_26.csv' (intermediate calculations) and 'Loitering_indexes_2024_02_26_11_44_41.csv' (all loitering scores for each MMSI, for further analysis).

Now you can choose any of the MSSI to be the input of 'Loitering_Tester.py' program.

1. Locate and replace the variable mmsi = 357051000 (with the MMSI you want to analyse).
2.  RUN.

The program will generate a sequence of plots (charts). One for each time-frame analysis. You can easily observ the trajectory of the vessels and check if the algorithm works well.

3. Similarly, you can run the program 'Loitering_Movie_Gen.py' which generate a movie of the vessel trajectory using the same data.
   Just need to replace the same variable  mmsi = 357051000 with the desired MMSSI.

   
