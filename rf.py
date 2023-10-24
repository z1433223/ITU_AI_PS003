import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split
# Replace 'your_file.parquet' with the actual path to your Parquet file

df = pq.read_table(source='cellular_dataframe.parquet').to_pandas()

count = 0

# Define the feature list
feature_list = ['ping_ms', 'jitter', 'Latitude', 'Longitude', 'Altitude', 'speed_kmh', 'COG', 'precipIntensity', 'precipProbability', 'temperature', 'dewPoint', 'humidity', 'pressure', 'windSpeed', 'cloudCover', 'uvIndex', 'visibility', 'Traffic Jam Factor', 'PCell_RSRP_1', 'PCell_RSRP_2', 'PCell_RSRP_max', 'PCell_RSRQ_1', 'PCell_RSRQ_2', 'PCell_RSRQ_max', 'PCell_RSSI_1', 'PCell_RSSI_2', 'PCell_RSSI_max', 'PCell_SNR_1', 'PCell_SNR_2', 'PCell_E-ARFCN', 'PCell_Downlink_Num_RBs', 'PCell_Downlink_TB_Size', 'PCell_Downlink_RBs_MCS_0', 'PCell_Downlink_RBs_MCS_1', 'PCell_Downlink_RBs_MCS_2', 'PCell_Downlink_RBs_MCS_3', 'PCell_Downlink_RBs_MCS_4', 'PCell_Downlink_RBs_MCS_5', 'PCell_Downlink_RBs_MCS_6', 'PCell_Downlink_RBs_MCS_7', 'PCell_Downlink_RBs_MCS_8', 'PCell_Downlink_RBs_MCS_9', 'PCell_Downlink_RBs_MCS_10', 'PCell_Downlink_RBs_MCS_11', 'PCell_Downlink_RBs_MCS_12', 'PCell_Downlink_RBs_MCS_13', 'PCell_Downlink_RBs_MCS_14', 'PCell_Downlink_RBs_MCS_15', 'PCell_Downlink_RBs_MCS_16', 'PCell_Downlink_RBs_MCS_17', 'PCell_Downlink_RBs_MCS_18', 'PCell_Downlink_RBs_MCS_19', 'PCell_Downlink_RBs_MCS_20', 'PCell_Downlink_RBs_MCS_21', 'PCell_Downlink_RBs_MCS_22', 'PCell_Downlink_RBs_MCS_23', 'PCell_Downlink_RBs_MCS_24', 'PCell_Downlink_RBs_MCS_25', 'PCell_Downlink_RBs_MCS_26', 'PCell_Downlink_RBs_MCS_27', 'PCell_Downlink_RBs_MCS_28', 'PCell_Downlink_RBs_MCS_29', 'PCell_Downlink_RBs_MCS_30', 'PCell_Downlink_RBs_MCS_31', 'PCell_Downlink_Average_MCS', 'PCell_Cell_ID', 'PCell_Downlink_frequency', 'PCell_Downlink_bandwidth_MHz', 'PCell_Cell_Identity', 'PCell_TAC', 'PCell_Band_Indicator', 'PCell_MCC', 'PCell_MNC_Digit', 'PCell_MNC', 'PCell_Allowed_Access', 'PCell_freq_MHz', 'SCell_RSRP_1', 'SCell_RSRP_2', 'SCell_RSRP_max', 'SCell_RSRQ_1', 'SCell_RSRQ_2', 'SCell_RSRQ_max', 'SCell_RSSI_1', 'SCell_RSSI_2', 'SCell_RSSI_max', 'SCell_SNR_1', 'SCell_SNR_2', 'SCell_E-ARFCN', 'SCell_Downlink_Num_RBs', 'SCell_Downlink_TB_Size', 'SCell_Downlink_RBs_MCS_0', 'SCell_Downlink_RBs_MCS_1', 'SCell_Downlink_RBs_MCS_2', 'SCell_Downlink_RBs_MCS_3', 'SCell_Downlink_RBs_MCS_4', 'SCell_Downlink_RBs_MCS_5', 'SCell_Downlink_RBs_MCS_6', 'SCell_Downlink_RBs_MCS_7', 'SCell_Downlink_RBs_MCS_8', 'SCell_Downlink_RBs_MCS_9', 'SCell_Downlink_RBs_MCS_10', 'SCell_Downlink_RBs_MCS_11', 'SCell_Downlink_RBs_MCS_12', 'SCell_Downlink_RBs_MCS_13', 'SCell_Downlink_RBs_MCS_14', 'SCell_Downlink_RBs_MCS_15', 'SCell_Downlink_RBs_MCS_16', 'SCell_Downlink_RBs_MCS_17', 'SCell_Downlink_RBs_MCS_18', 'SCell_Downlink_RBs_MCS_19', 'SCell_Downlink_RBs_MCS_20', 'SCell_Downlink_RBs_MCS_21', 'SCell_Downlink_RBs_MCS_22', 'SCell_Downlink_RBs_MCS_23', 'SCell_Downlink_RBs_MCS_24', 'SCell_Downlink_RBs_MCS_25', 'SCell_Downlink_RBs_MCS_26', 'SCell_Downlink_RBs_MCS_27', 'SCell_Downlink_RBs_MCS_28', 'SCell_Downlink_RBs_MCS_29', 'SCell_Downlink_RBs_MCS_30', 'SCell_Downlink_RBs_MCS_31', 'SCell_Downlink_Average_MCS', 'SCell_Cell_ID', 'SCell_Downlink_frequency', 'SCell_Downlink_bandwidth_MHz', 'SCell_Cell_Identity', 'SCell_TAC', 'SCell_Band_Indicator', 'SCell_MCC', 'SCell_MNC_Digit', 'SCell_MNC', 'SCell_freq_MHz']

# Select features
selected_features_df = df[feature_list]

# Separate the target variable (assuming 'target' is the column name)
selected_features_df = df[feature_list]

target = df['datarate']

selected_features_df.fillna(0, inplace=True)

target_imputer = SimpleImputer(strategy='mean')
target = target_imputer.fit_transform(target.values.reshape(-1, 1)).flatten()


# Initialize a RandomForestRegressor
regressor = RandomForestRegressor()

# Fit
regressor.fit(selected_features_df, target)

feature_importances = regressor.feature_importances_

feature_importance_tuples = list(zip(feature_list, feature_importances))

# Sort features by importance
sorted_features = sorted(feature_importance_tuples, key=lambda x: x[1], reverse=True)

print("Feature Importances:")
for feature, importance in sorted_features:
    print(f"{feature}: {importance:.4f}")