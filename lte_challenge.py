from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import pyarrow.parquet as pq
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mean_squared_error, binary_crossentropy
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the dataset
cell_df = pq.read_table(source='cellular_dataframe.parquet').to_pandas()
filtered_data = cell_df.query("direction == 'downlink' & measured_qos == 'datarate'")
filtered_data = filtered_data.dropna(subset = 'datarate')

train_data = filtered_data.query("operator == 1")
test_data = filtered_data.query("operator == 2")

# Select the chosen features
qos_column = 'datarate'


selected_features = [
    
    'PCell_RSRP_max',
     'PCell_RSRQ_max',
     'PCell_RSSI_max',
     'PCell_SNR_1',
     'PCell_SNR_2',
     'PCell_Downlink_Num_RBs',
     'PCell_Downlink_TB_Size',
     'PCell_Downlink_Average_MCS',
     'SCell_RSRP_max',
     'SCell_RSRQ_max',
     'SCell_RSSI_max',
     'SCell_SNR_1',
     'SCell_SNR_2',
     'SCell_Downlink_Num_RBs',
     'SCell_Downlink_TB_Size',
     'SCell_Downlink_Average_MCS',
     'Traffic Jam Factor'
]

x_train_source, y_train_source = train_data[selected_features], train_data[qos_column]
x_test_target, y_test_target = test_data[selected_features], test_data[qos_column]


imputer = SimpleImputer(strategy='mean')
x_train_source = imputer.fit_transform(x_train_source)
x_test_target = imputer.transform(x_test_target)

source_domain_labels = np.zeros(x_train_source.shape[0])
target_domain_labels = np.ones(x_test_target.shape[0])
domain_labels = np.concatenate((source_domain_labels, target_domain_labels))

combined_data = np.concatenate((x_train_source, x_test_target))

input_layer = Input(shape=(x_train_source.shape[1],))

feature_extractor_hidden = Dense(100, activation='relu')(input_layer)
shared_features = Dense(100, activation='relu')(feature_extractor_hidden)
regression_hidden = Dense(100, activation='relu')(shared_features)
regression_output = Dense(1)(regression_hidden)
domain_hidden = Dense(100, activation='relu')(shared_features)
domain_output = Dense(1, activation='sigmoid')(domain_hidden)

regression_model = Model(inputs=input_layer, outputs=regression_output)
combined_model = Model(inputs=input_layer, outputs=[regression_output, domain_output])

# Create gradient reversal layer
def grad_reverse(x):
    return -1.0 * x

gradient_reversal_layer = Lambda(grad_reverse)

domain_adversarial_output = gradient_reversal_layer(domain_output)

combined_model.compile(
    optimizer=Adam(learning_rate=0.05),
    loss=['mean_squared_error', 'binary_crossentropy'],
    loss_weights=[1.0, 1.0]
)

x_train_source_mean = x_train_source.mean()
x_train_source_std = x_train_source.std()
x_train_source_normalized = (x_train_source - x_train_source_mean) / x_train_source_std
x_test_target_normalized = (x_test_target - x_train_source_mean) / x_train_source_std
y_train_source_mean = y_train_source.mean()
y_train_source_std = y_train_source.std()
y_train_source_normalized = (y_train_source - y_train_source_mean) / y_train_source_std

combined_model.fit(
    x_train_source_normalized,
    [y_train_source_normalized, source_domain_labels],
    epochs=64,
    batch_size=32
)

y_pred_target_normalized, _ = combined_model.predict(x_test_target_normalized)

y_pred_target = (y_pred_target_normalized * y_train_source_std) + y_train_source_mean

r2_target = r2_score(y_test_target, y_pred_target)

print("R2 Score on Target Domain:", r2_target)


# input_layer = Input(shape=(x_train_source.shape[1],))

# feature_extractor_hidden = Dense(100, activation='relu')(input_layer)
# shared_features = Dense(100, activation='relu')(feature_extractor_hidden)
# regression_hidden = Dense(100, activation='relu')(shared_features)
# regression_output = Dense(1)(regression_hidden)
# domain_hidden = Dense(100, activation='relu')(shared_features)
# domain_output = Dense(1, activation='sigmoid')(domain_hidden)

# regression_model = Model(inputs=input_layer, outputs=regression_output)
# combined_model = Model(inputs=input_layer, outputs=[regression_output, domain_output])

# def grad_reverse(x):
#     return -1.0 * x

# gradient_reversal_layer = Lambda(grad_reverse)
# reversed_domain_output = gradient_reversal_layer(domain_hidden)
# domain_adversarial_output = Dense(1, activation='sigmoid')(reversed_domain_output)

# domain_adversarial_model = Model(inputs=input_layer, outputs=domain_adversarial_output)

# combined_model.compile(
#     optimizer=Adam(learning_rate=0.05),
#     loss=['mean_squared_error', 'binary_crossentropy'],
#     loss_weights=[1.0, 1.0]
# )

# domain_adversarial_model.compile(
#     optimizer=Adam(learning_rate=0.05),
#     loss='binary_crossentropy'
# )

# x_train_source_mean = x_train_source.mean()
# x_train_source_std = x_train_source.std()
# y_train_source_mean = y_train_source.mean()
# y_train_source_std = y_train_source.std()
# x_train_source_normalized = (x_train_source - x_train_source_mean) / x_train_source_std
# x_test_target_normalized = (x_test_target - x_train_source_mean) / x_train_source_std
# y_train_source_normalized = (y_train_source - y_train_source_mean) / y_train_source_std

# combined_model.fit(
#     x_train_source_normalized,
#     [y_train_source_normalized, source_domain_labels],
#     epochs=64,
#     batch_size=32
# )

# domain_adversarial_model.fit(
#     combined_data,
#     domain_labels,
#     epochs=64,
#     batch_size=32
# )

# y_pred_target_normalized, _ = combined_model.predict(x_test_target_normalized)

# y_pred_target = (y_pred_target_normalized * y_train_source_std) + y_train_source_mean

# r2_target = r2_score(y_test_target, y_pred_target)

# print("R2 Score on Target Domain:", r2_target)
