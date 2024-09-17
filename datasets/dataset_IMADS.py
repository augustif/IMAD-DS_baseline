# built-in
import os
from time import gmtime, strftime

# libraries
import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

#custom
from datasets.dataset_utilities import *

class DatasetFromSegmentFiles(Dataset):
    def __init__(self, csv_file, root_dir, sensors = ['imp23absu_mic', 'ism330dhcx_acc', 'ism330dhcx_gyr'], transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the sensor data files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.sensors = sensors
        self.labels_map = {
            'normal': 0,
            'anomaly': 1
        }        

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        sensor_data = []
        row = self.annotations.iloc[idx, :]

        for sensor in self.sensors:
            sensor_file = os.path.join(self.root_dir,row[sensor] )
            sensor_df = pd.read_parquet(sensor_file)
            sensor_data.append(sensor_df.values.astype('float32'))
        
        # Concatenate all sensor data along the last dimension
        sample = torch.tensor(sensor_data, dtype=torch.float32)
        label = self.labels_map[row['anomaly_label']]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

class DatasetFromNumpy(Dataset):
    """
    Custom Dataset for handling multi-sensor data.
    """
    sensor_dict = {
            'imp23absu_mic': {
                'fs': 16000,
                'number_of_channel': 1
            },
            'ism330dhcx_acc': {
                'fs': 7063,  # Estimated sampling rate calculated by averaging time deltas across all files
                'number_of_channel': 3
            },
            'ism330dhcx_gyro': {
                'fs': 7063,  # Estimated sampling rate calculated by averaging time deltas across all files
                'number_of_channel': 3
            }
        }
    # Duration of initial data time affected by the gyroscope warm-up period
    gyroscope_warm_up_time = pd.to_timedelta('35ms')

    # List of label names to be extracted from the dataset
    label_names = [
        'segment_id',
        'split_label',
        'anomaly_label',
        'domain_shift_op',
        'domain_shift_env'
        ]  
    
    def __init__(self, X, y, device, transform=None):
        """
        Initialize the CustomDataset with sensor data.

        Parameters:
        X (list): List of numpy arrays, where each array contains data from a different sensor.
        y (list): List of string, where each string represent the label of i-th element of X arrays.
        transform (str): Normalization method ('std', 'min-max', 'std_window', or 'min-max_window').

        """
        self.X = X
        self.y = y

        # Ensure that X is a list of numpy arrays
        if not all(isinstance(x, np.ndarray) for x in self.X):
            raise ValueError("All elements of X must be numpy arrays")

        self.X = [torch.from_numpy(x).to(device) for x in self.X]

        # apply transform (data normalization) function on the whole dataset
        # during initialization to speed up __get_item__()
        if transform is not None:
            self.X = self.normalize_data(self.X, transform)

    def __len__(self):
        """
        Return the length of the dataset.

        Returns:
        int: Length of the dataset, which is the length of the first sensor's data.
        """
        return len(self.y)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset at the specified index.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        list: A list of samples from each sensor at the specified index.
        """
        return [x[idx] for x in self.X], self.y[idx]

    @staticmethod
    def normalize_data(X, normalisation):
        """
        Normalize the training, validation, and test datasets using the specified normalization method.

        Parameters:
        X (list): List of numpy arrays, one per sensor.
        normalisation (str): Normalization method ('std', 'min-max', 'std_window', or 'min-max_window').

        Returns:
        tuple: Normalized datasets.
        """

        sensor_count = len(X)
        channel_counts = [X[i].shape[1] for i in range(sensor_count)]

        if normalisation == 'std':
            # Calculate means and standard deviations across all samples and windows for each sensor
            means_ = [X[i].mean(2).mean(0).reshape(
                1, channel_counts[i], 1) for i in range(sensor_count)]
            stds_ = [X[i].std(2).mean(0).reshape(
                1, channel_counts[i], 1) for i in range(sensor_count)]

            # Standardize each dataset using the calculated means and standard deviations
            X = [standardize(X[i], means_[i], stds_[i])
                    for i in range(sensor_count)]

        elif normalisation == 'min-max':
            # Calculate min and max values across all samples and windows for each sensor
            mins_ = [X[i].min(2).min(0).reshape(
                1, channel_counts[i], 1) for i in range(sensor_count)]
            maxs_ = [X[i].max(2).max(0).reshape(
                1, channel_counts[i], 1) for i in range(sensor_count)]

            # Apply min-max scaling to each dataset using the calculated min and max values
            X = [min_max_scale(X[i], mins_[i], maxs_[i])
                    for i in range(sensor_count)]

        elif normalisation == 'std_window':
            # Apply standardization within each window for each sensor
            X = [standardize_window(X[i]) for i in range(sensor_count)]

        elif normalisation == 'min-max_window':
            # Apply min-max scaling within each window for each sensor
            X = [min_max_scale_window(X[i])
                    for i in range(sensor_count)]
        
        return X

    def load_windows_dataset(self, path, label_names, sensors):
        """
        Load training and testing datasets from HDF5 files.

        Parameters:
        train_path (str): Path to the training dataset HDF5 file.
        test_path (str): Path to the testing dataset HDF5 file.
        label_names (list): List of label names to extract from the HDF5 files.
        sensors (dict): dict containing sensors to extract from the HDF5 files. Sensor names must be the dict keys.

        Returns:
        tuple: A tuple containing the following elements:
            - X_raw (list): List of numpy arrays containing raw data for each sensor.
            - y_raw (pd.DataFrame): DataFrame containing labels.
        """
        with h5py.File(path, 'r') as f:
            # Extract raw training data for each sensor
            X_raw = [f[sensor][:] for sensor in sensors]
            # Extract and decode training labels
            Y_raw = pd.DataFrame([[s.decode(
                'utf-8') for s in f[label_name][:].flatten()] for label_name in label_names]).T
            Y_raw.columns = label_names
            
        return X_raw, Y_raw

    def process_windows_dataset(self, split_type, metadata, sensor_dict, output_folder, window_size_ts, gyroscope_warm_up_time):
        # Loop through each dataset split type ('train' and 'test') with
        # corresponding metadata

        # Define the save path for the HDF5 file
        save_path = '{}/{}_dataset_window_{:.3f}s.h5'.format(
            output_folder,
            split_type,
            window_size_ts.total_seconds()
        )
        
        if os.path.exists(save_path):
            print(f"File exists: {save_path}. Aborted creation")
            return
        else:
            print(f"Creation of file {save_path}")

        # Open the HDF5 file in write mode
        with h5py.File(save_path, 'w') as h5file:
            # ================================================================ INIT
            # Initialize datasets dictionary to store HDF5 datasets
            datasets = {}

            # Create datasets for each sensor defined in sensor_dict
            for sensor in sensor_dict.keys():
                window_length = sensor_dict[sensor]['window_length']
                number_of_channel = sensor_dict[sensor]['number_of_channel']

                # Create a dataset for each sensor with specified shape and
                # chunking
                datasets[sensor] = h5file.create_dataset(
                    sensor,
                    shape=(0, number_of_channel, window_length),
                    maxshape=(None, number_of_channel, window_length),
                    chunks=True
                )

            # Create additional datasets for segment ID and various labels

            # dataset containing the index of corresponding segment
            datasets['segment_id'] = h5file.create_dataset(
                'segment_id',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # dataset containing split labels
            datasets['split_label'] = h5file.create_dataset(
                'split_label',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # dataset containing anomaly labels
            datasets['anomaly_label'] = h5file.create_dataset(
                'anomaly_label',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # dataset containing operational domain shift labels
            datasets['domain_shift_op'] = h5file.create_dataset(
                'domain_shift_op',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # dataset containing environmental domain shift labels
            datasets['domain_shift_env'] = h5file.create_dataset(
                'domain_shift_env',
                shape=(0, 1),
                maxshape=(None, 1),
                chunks=True,
                dtype=h5py.string_dtype(encoding='utf-8')
            )

            # ============================================  DATA SEGMENTATION INTO
            # Every row of the Metadata represent the i-th segment of one specific recording:
            # the same segment is recorded for all sensors, named in the same way and its path
            # is linked in the appropriate column of the dataframe

            # Iterate over all segments in the metadata
            for file_index in range(len(metadata)):
                try:
                    print(
                        f'Completed: {file_index / (len(metadata)-1)*100:.2f}%',
                        end='\r')

                    # Load and process data for each sensor
                    for sensor in sensor_dict:
                        sensor_df = pd.read_parquet(metadata[sensor][file_index])
                        sensor_df['Time'] = pd.to_datetime(
                            sensor_df['Time'], unit='s')
                        sensor_df.set_index('Time', inplace=True)
                        sensor_df.sort_index(inplace=True)

                        sensor_dict[sensor]['data_raw'] = sensor_df
                        sensor_dict[sensor]['max_ts'] = sensor_df.index[-1]
                        sensor_dict[sensor]['min_ts'] = sensor_df.index[0]

                    # Determine the time range for the segment: makes sure that
                    # there is available data for all sensors
                    max_ts_list = [sensor_dict[sensor]['max_ts']
                                for sensor in sensor_dict]
                    min_ts_list = [sensor_dict[sensor]['min_ts']
                                for sensor in sensor_dict]

                    start_timestamp = max(
                        sensor_dict['ism330dhcx_gyro']['min_ts'] +
                        gyroscope_warm_up_time,
                        max(min_ts_list))
                    end_timestamp = min(max_ts_list)

                    # Extract labels for the segment
                    segment_id = metadata['segment_id'][file_index]
                    split_label = metadata['split_label'][file_index]
                    anomaly_label = metadata['anomaly_label'][file_index]
                    domain_shift_op = metadata['domain_shift_op'][file_index]
                    domain_shift_env = metadata['domain_shift_env'][file_index]

                    flag = 1
                    number_of_window = (
                        end_timestamp - start_timestamp) // window_size_ts

                    # Iterate over each sensor to process the data into windows
                    for sensor in sensor_dict:
                        sensor_df = sensor_dict[sensor]['data_raw']
                        num_points_per_window = sensor_dict[sensor]['window_length']
                        num_channel = sensor_dict[sensor]['number_of_channel']

                        # Iterate over each window in the segment
                        for window_idx in range(number_of_window):
                            start = start_timestamp + window_idx * window_size_ts
                            end = start + window_size_ts
                            sensor_df_window = sensor_df[start:end].values

                            # Zero-pad or truncate the window to match the expected
                            # length
                            l = len(sensor_df_window)
                            if l < num_points_per_window:
                                pad_size = num_points_per_window - l
                                padding = np.zeros((pad_size, num_channel))
                                sensor_df_window = np.vstack(
                                    [sensor_df_window, padding])
                            else:
                                sensor_df_window = sensor_df_window[:num_points_per_window, :]

                            # Resize and store the windowed data in the HDF5
                            # dataset
                            current_size = datasets[sensor].shape[0]
                            datasets[sensor].resize(current_size + 1, axis=0)
                            datasets[sensor][-1] = sensor_df_window.T

                            if flag:
                                current_size = datasets['segment_id'].shape[0]

                                datasets['segment_id'].resize(
                                    current_size + 1, axis=0)
                                datasets['segment_id'][-1] = segment_id

                                datasets['split_label'].resize(
                                    current_size + 1, axis=0)
                                datasets['split_label'][-1] = split_label

                                datasets['anomaly_label'].resize(
                                    current_size + 1, axis=0)
                                datasets['anomaly_label'][-1] = anomaly_label

                                datasets['domain_shift_op'].resize(
                                    current_size + 1, axis=0)
                                datasets['domain_shift_op'][-1] = domain_shift_op

                                datasets['domain_shift_env'].resize(
                                    current_size + 1, axis=0)
                                datasets['domain_shift_env'][-1] = domain_shift_env

                        flag = 0
                except Exception as e:
                    print('could not read file index {}'.format(file_index), e)

class DatasetTrain(DatasetFromNumpy):
    """
    Custom Dataset for handling multi-sensor data.

    Attributes:
    X (list): List of numpy arrays, where each array contains data from a different sensor.
    """

    def __init__(self, machine = 'BrushlessMotor', window_size_ms = 100, params = None):
        """
        Initialize the CustomDataset with sensor data.

        Parameters:
        X (list): List of numpy arrays, where each array contains data from a different sensor.
        """
        self.machine = machine
        # Initializations
        self.input_folder = f'data/{self.machine}'
        self.output_folder = f'data/{self.machine}/windowed'
        os.makedirs(self.output_folder, exist_ok=True)

        # constants
        self.window_size_ts = pd.to_timedelta(f'{window_size_ms}ms')
        
        # update sensor window length in samples
        for sensor in self.sensor_dict.keys():
            sensor = self.sensor_dict[sensor]
            sensor['window_length'] = int(
                sensor['fs'] * self.window_size_ts.total_seconds())

        # load metadata
        normal_source_train = pd.read_csv(
            f'{self.input_folder}/train/attributes_normal_source_train.csv',
            index_col=0)
        normal_target_train = pd.read_csv(
            f'{self.input_folder}/train/attributes_normal_target_train.csv',
            index_col=0)

        metadata = pd.concat(
            [normal_source_train, normal_target_train], axis=0).reset_index(drop=True)
        
        # create segment id column
        metadata['segment_id'] = metadata['imp23absu_mic'].apply(
            lambda x: x.replace('imp23absu_mic_', ''))

        # add custom dataset path to each filepath in the Metadata dataframes
        for sensor in self.sensor_dict.keys():
            metadata[sensor] = self.input_folder + '/train/' + metadata[sensor]
        
        self.metadata = metadata

        # create windows dataset if not present yet
        self.process_windows_dataset('train', metadata, self.sensor_dict, self.output_folder, self.window_size_ts, self.gyroscope_warm_up_time)

        X, y = self.load_windows_dataset(
            path ='data/{}/windowed/train_dataset_window_{:.3f}s.h5'.format(
                self.machine,
                self.window_size_ts.total_seconds()
            ),
            label_names = self.label_names,
            sensors= self.sensor_dict
            )
        
        # Combine anomaly labels and domain shift labels to form a combined label
        y['combined_label'] = y['anomaly_label'] + \
            y['domain_shift_op'] + y['domain_shift_env']

        # Split training data into training and validation sets, maintaining the
        # stratified distribution of the combined label
        train_indices, valid_indices, _, _ = train_test_split(
            range(len(y)),
            y,
            stratify=y['combined_label'],
            test_size=params['valid_size'],
            random_state=params['seed']
        )

        # Select the training and validation data based on the indices
        X_train = [sensor_data[train_indices] for sensor_data in X]
        X_valid = [sensor_data[valid_indices] for sensor_data in X]
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_valid = y.iloc[valid_indices].reset_index(drop=True)
        
        super().__init__(X_train, y_train['anomaly_label'].to_list(), params['device'], params['normalisation'])
        self.X_valid = X_valid
        self.y_valid = y_valid

    def get_valid_dataset(self):
        return self.X_valid, self.y_valid

class DatasetTest(DatasetFromNumpy):
    """
    Custom Dataset for handling multi-sensor data.

    Attributes:
    X (list): List of numpy arrays, where each array contains data from a different sensor.
    """

    def __init__(self, machine = 'BrushlessMotor', window_size_ms = 100, params = None):
        """
        Initialize the CustomDataset with sensor data.

        Parameters:
        X (list): List of numpy arrays, where each array contains data from a different sensor.
        """
        self.machine = machine
        # Initializations
        self.input_folder = f'data/{self.machine}'
        self.output_folder = f'data/{self.machine}/windowed'
        os.makedirs(self.output_folder, exist_ok=True)

        # constants
        self.window_size_ts = pd.to_timedelta(f'{window_size_ms}ms')
        
        # update sensor window length in samples
        for sensor in self.sensor_dict.keys():
            sensor = self.sensor_dict[sensor]
            sensor['window_length'] = int(
                sensor['fs'] * self.window_size_ts.total_seconds())
        
        # load metadata
        normal_source_test = pd.read_csv(
            f'{self.input_folder}/test/attributes_normal_source_test.csv',
            index_col=0)
        anomaly_source_test = pd.read_csv(
            f'{self.input_folder}/test/attributes_anomaly_source_test.csv',
            index_col=0)
        normal_target_test = pd.read_csv(
            f'{self.input_folder}/test/attributes_normal_target_test.csv',
            index_col=0)
        anomaly_target_test = pd.read_csv(
            f'{self.input_folder}/test/attributes_anomaly_target_test.csv',
            index_col=0)

        metadata = pd.concat([normal_source_test,
                                anomaly_source_test,
                                normal_target_test,
                                anomaly_target_test],
                                axis=0).reset_index(drop=True)
        
        # create segment id column
        metadata['segment_id'] = metadata['imp23absu_mic'].apply(
            lambda x: x.replace('imp23absu_mic_', ''))

        # add custom dataset path to each filepath in the Metadata dataframes
        for sensor in self.sensor_dict.keys():
            metadata[sensor] = self.input_folder + '/test/' + metadata[sensor]
        
        self.metadata = metadata

        # create windows dataset if not present yet
        self.process_windows_dataset('test', metadata, self.sensor_dict, self.output_folder, self.window_size_ts, self.gyroscope_warm_up_time)
         
        # load all dataset in memory to spped up training 
        X, y = self.load_windows_dataset(
            path ='data/{}/windowed/test_dataset_window_{:.3f}s.h5'.format(
                self.machine,
                self.window_size_ts.total_seconds()
            ),
            label_names = self.label_names,
            sensors= self.sensor_dict
        )
        
        # Combine anomaly labels and domain shift labels to form a combined label
        y['combined_label'] = y['anomaly_label'] + \
            y['domain_shift_op'] + y['domain_shift_env']
        
        super().__init__(X,y['anomaly_label'].to_list(), params['device'], params['normalisation'])
    
if __name__ == '__main__':

    PARAMS = {
    'layer_dims': [2048, 2048, 2048, 16],
    'lr': 0.0001,
    'criterion': 'MSE',
    'batch_size': 1024,
    'num_epochs': 1000,
    'patience': 3,
    'normalisation': 'std_window',
    'valid_size': 0.1,
    'seed': 1995
    }

    ds = DatasetTrain(machine="BrushlessMotor", window_size_ms=100, params=PARAMS)
