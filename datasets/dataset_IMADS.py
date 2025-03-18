# built-in
import os
from time import gmtime, strftime

# libraries
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

#custom
from datasets.dataset_utilities import download_file, unzip_7z_file

class IMADSBaseDataset(Dataset):
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

    label_names = [
        'segment_id',
        'split_label',
        'anomaly_label',
        'domain_shift_op',
        'domain_shift_env'
        ]
    # Duration of initial data time affected by the gyroscope warm-up period
    gyroscope_warm_up_time = pd.to_timedelta('35ms')

    def set_sensor_dict(self, sensor_dict):
        self.sensor_dict = sensor_dict    

    def update_sensors_dict(self, sensor_dict = None, sensors_enabled = None):
        # update sensors_dict based on enabled_sensors
        if not sensors_enabled:
            raise ValueError("sensors_enabled cannot be empty")
        else:
            if sensor_dict:
                self.set_sensor_dict(sensor_dict)
                print(f"Overridden sensor dict: {self.sensor_dict}")

            prev_sensor_dict = self.sensor_dict.copy()
            self.sensor_dict = {k: v for k, v in self.sensor_dict.items() if k in sensors_enabled}
            if sensor_dict  != prev_sensor_dict:
                print(f"Sensor dict updated: {self.sensor_dict}")


    def __init__(self, 
                 X: list[np.array], 
                 y: list[str], 
                 device: str, 
                 sensors_enabled: list[str] = ['all'],
                 sensor_dict: dict = None, 
                 label_names: list['str'] = None, 
                 transform_pipeline: object = None
                 ):
        """
        Initialize the Dataset with sensor data.

        Parameters:
        sensor dict (dict): Dictionary containing sensor names and their respective attributes.
        label_names (list): List of label names to extract from the dataset.
        X (list): List of numpy arrays, where each array contains data from a different sensor.
        y (list): List of string, where each string represent the label of i-th element of X arrays.
        transform (str): Normalization method ('std', 'min-max', 'std_window', or 'min-max_window').        
        """

        # update sensor_dict based on enabled_sensors
        self.update_sensors_dict(sensor_dict, sensors_enabled)

        if label_names is not None:
            self.label_names = label_names

        self.X = X
        self.y = y

        # Ensure that X is a list of numpy arrays
        if not all(isinstance(x, np.ndarray) for x in self.X):
            raise ValueError("All elements of X must be numpy arrays")

        self.X = [torch.from_numpy(x).to(device) for x in self.X]

        # apply transform (data normalization) function on the whole dataset
        # during initialization to speed up __get_item__()
        if transform_pipeline is not None:
            self.X = transform_pipeline.transform(self.X)
    
    def set_transform_pipeline(self, pipeline):
        self.transform_pipeline = pipeline

    def apply_preprocess_pipeline(self):
        if self.transform_pipeline is None:
                raise ValueError("No transform pipeline set. Call set_transform_pipeline() before apply_preprocess_pipeline()")
        self.X = self.transform_pipeline.transform(self.X)
    
    @staticmethod
    def check_or_get_data(machine: str = 'BrushlessMotor', data_folder: Path = Path('data')):
        """
        Check if data is already downloaded, if not, download and extract it.

        Parameters:
        machine (str): Name of the machine (e.g., 'BrushlessMotor').
        data_folder (Path): Path to the data folder.
        """
        # Check if the data folder already contains the necessary files
        required_files = [
            data_folder / machine / 'train/attributes_normal_source_train.csv',
            data_folder / machine / 'train/attributes_normal_target_train.csv',
            data_folder / machine / 'test/attributes_normal_source_test.csv',
            data_folder / machine / 'test/attributes_anomaly_source_test.csv',
            data_folder / machine / 'test/attributes_normal_target_test.csv',
            data_folder / machine / 'test/attributes_anomaly_target_test.csv'
        ]
        
        if all(file.exists() for file in required_files):
            print(f"Data for {machine} already exists in {data_folder}.")
            return
        
        # If any required file is missing, download and extract the data
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        local_filename = data_folder / Path(f'{machine}.7z')

        download_file(
            url=f'https://zenodo.org/record/12665499/files/{machine}.7z',
            local_filename=local_filename
        )
        unzip_7z_file(file_path=local_filename, extract_to=data_folder)

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
        list: A list of samples from each sensor at the specified index and the relevant labels
        """
        # self.X: list of n sensors numpy arrays, with shape (num_windows, num_channels, window_lenght) 
        x = [x[idx] for x in self.X]
        if isinstance(self.y, pd.DataFrame):
            y = self.y.iloc[idx].to_dict()
        elif isinstance(self.y, list):
            y = self.y[idx]
        # x: (stacked_sensors_size), y=(1) where stacked_sensors_size is: for each sensor sum(sensor_channels*sensor_window_length) 
        return x, y

    def load_windows(self, path, label_names, sensors):
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

    def to_windows(self, split_type, metadata, sensor_dict, output_folder, window_size_ts, gyroscope_warm_up_time):
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

class IMADSDatasetTrain(IMADSBaseDataset):
    """
    Custom Dataset for handling multi-sensor data.

    Attributes:
    X (list): List of numpy arrays, where each array contains data from a different sensor.
    """

    def __init__(self,
                 seed: int,
                 data_folder: Path = Path('data'),
                 sensors_enabled: list[str] = ['ism330dhcx_acc', 'ism330dhcx_gyro', 'imp23absu_mic'],
                 sensor_dict = None, 
                 label_names = None, 
                 machine = 'BrushlessMotor', 
                 window_size_ms: int = 100, 
                 device: str ='cpu',
                 transform_pipeline: object = None,
                 valid_size:int=0.1,
                 ):
        """
        Initialize the CustomDataset with sensor data.

        Parameters:
        X (list): List of numpy arrays, where each array contains data from a different sensor.
        """
        self.machine = machine

        # Initializations
        self.input_folder = data_folder / Path(self.machine)
        self.output_folder = data_folder / Path(self.machine) / Path('windowed')

        super().check_or_get_data(machine, data_folder)
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
        dummy_sensor = list(self.sensor_dict.keys())[0]
        metadata['segment_id'] = metadata[dummy_sensor].apply(
            lambda x: x.replace(dummy_sensor, ''))

        # add custom dataset path to each filepath in the Metadata dataframes
        for sensor in self.sensor_dict.keys():
            metadata[sensor] = str(self.input_folder) + '/train/' + metadata[sensor]
        
        self.metadata = metadata

        # create windows dataset if not present yet
        self.to_windows('train', metadata, self.sensor_dict, self.output_folder, self.window_size_ts, self.gyroscope_warm_up_time)

        # update sensors dict before loading to ensure only enabled sensors are loaded
        self.update_sensors_dict(sensor_dict, sensors_enabled)

        # X: list of n sensors --> (num_windows, num_channels, window_lenght)
        X, y = self.load_windows(
            path ='{}/train_dataset_window_{:.3f}s.h5'.format(
                self.output_folder,
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
            test_size=valid_size,
            random_state=seed
        )

        # Select the training and validation data based on the indices
        X_train = [sensor_data[train_indices] for sensor_data in X]
        X_valid = [sensor_data[valid_indices] for sensor_data in X]
        y_train = y.iloc[train_indices].reset_index(drop=True)
        y_valid = y.iloc[valid_indices].reset_index(drop=True)
        
        super().__init__(X_train, 
                        y_train['anomaly_label'].to_list(), 
                        sensors_enabled=sensors_enabled,
                        sensor_dict=sensor_dict,
                        label_names=label_names,
                        device=device, 
                        transform_pipeline=transform_pipeline)
        
        self.X_valid = X_valid
        self.y_valid = y_valid

    def get_valid_dataset(self):
        return self.X_valid, self.y_valid

class IMADSDatasetTest(IMADSBaseDataset):
    """
    Custom Dataset for handling multi-sensor data.

    Attributes:
    X (list): List of numpy arrays, where each array contains data from a different sensor.
    """

    def __init__(self,
                 data_folder: Path = Path('data'),
                 sensors_enabled: list[str] = ['all'], 
                 sensor_dict = None, 
                 label_names = None, 
                 machine = 'BrushlessMotor', 
                 window_size_ms: int = 100, 
                 device: str ='cpu',
                 transform_pipeline: object = None
                 ):
        """
        Initialize the CustomDataset with sensor data.

        Parameters:
        X (list): List of numpy arrays, where each array contains data from a different sensor.
        """
        self.machine = machine

        # Initializations
        self.input_folder = data_folder / Path(self.machine)
        self.output_folder = data_folder / Path(self.machine) / Path('windowed')

        super().check_or_get_data(machine, data_folder)
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
        dummy_sensor = list(self.sensor_dict.keys())[0]
        metadata['segment_id'] = metadata[dummy_sensor].apply(
            lambda x: x.replace(dummy_sensor, ''))

        # add custom dataset path to each filepath in the Metadata dataframes
        for sensor in self.sensor_dict.keys():
            metadata[sensor] = str(self.input_folder) + '/test/' + metadata[sensor]
        
        self.metadata = metadata

        # create windows dataset if not present yet
        self.to_windows('test', metadata, self.sensor_dict, self.output_folder, self.window_size_ts, self.gyroscope_warm_up_time)

        X, y = self.load_windows(
            path ='{}/test_dataset_window_{:.3f}s.h5'.format(
                self.output_folder,
                self.window_size_ts.total_seconds()
            ),
            label_names = self.label_names,
            sensors= self.sensor_dict
            )
        
        # Combine anomaly labels and domain shift labels to form a combined label
        y['combined_label'] = y['anomaly_label'] + \
            y['domain_shift_op'] + y['domain_shift_env']
        
        super().__init__(X, 
                        y,
                        sensors_enabled=sensors_enabled,
                        sensor_dict=sensor_dict,
                        label_names=label_names,
                        device=device, 
                        transform_pipeline=transform_pipeline)

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

    ds = IMADSDatasetTrain(machine="BrushlessMotor", window_size_ms=100, params=PARAMS)
