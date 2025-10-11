import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

warnings.filterwarnings('ignore')

class Pred_TimeSeriesDataset(Dataset):
    def __init__(
            self,
            path,
            split = 'test',
            seq_len = 24 * 4 * 4,
            label_len = 24 * 4,
            pred_len = 24 * 4, 
            scale = True,
            inverse = False,
            random_state = 42,
            is_timeencoded = True,
            frequency = 'd',
            features = 'MS',
            target = 'BP'
            ):
        assert split in ['train', 'val', 'test']
        self.path = path 
        self.split = split
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale
        self.random_state = random_state
        self.is_timeencoded = is_timeencoded
        self.frequency = frequency
        self.inverse = inverse
        self.features = 'MS'
        self.target = 'BP'

        self.scaler = StandardScaler()

        self.prepare_data()

    def prepare_data(self):
        df = pd.read_csv(self.path) # 1st col: date 
        df['Date'] = pd.to_datetime(df['Date'])

        indices = df.index.tolist()
        train_size = 0.6
        val_size = 0.2
        test_size = 0.2

        train_end = int(len(indices) * train_size)
        val_end = train_end + int(len(indices) * val_size)
        train_indices = indices[:train_end]
        if self.split == 'train':
            split_indices = train_indices
        elif self.split == 'val':
            split_indices = indices[train_end:val_end]
        if self.split == 'test':
            split_indices = indices[val_end:]
            
        df_split = df.loc[split_indices]

        data_columns = df_split.columns[1:]
        data = df_split[data_columns]
        # data_y = df_split[data_columns[4:]] # y: exclude input columns
        self.feature_names = data_columns

        data = torch.FloatTensor(data.values)

        if self.scale:
            df_data = df.loc[split_indices][self.feature_names].values
            self.scaler.fit(df_data)
            data = self.scaler.transform(df_data)
        else:
            data = data.values

        if self.inverse:
            data_y = df_data
        else:
            data_y = data
            # data_y = data[:,4:]

        tmp_stamp = df_split[['Date']]
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)
        df_stamp = pd.DataFrame(columns=['Date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])

        if not self.is_timeencoded:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            if self.frequency == 'h' or self.frequency == 't':
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            if self.frequency == 't':
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            df_stamp_data = df_stamp.drop('date', axis = 1).values
            
        else:
            df_stamp_data = time_features(pd.to_datetime(timestamp.Date.values), freq = self.frequency)
            df_stamp_data = df_stamp_data.transpose(1,0)

        self.time_series_x = torch.FloatTensor(data)
        self.time_series_y = torch.FloatTensor(data_y)
        self.timestamp = torch.FloatTensor(df_stamp_data)
    
    def __getitem__(self, index):
        x_begin_index = index
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index - self.label_len
        y_end_index = y_begin_index + self.label_len + self.pred_len

        x = self.time_series_x[x_begin_index:x_end_index]
        y = self.time_series_y[y_begin_index:y_end_index]
        
        x_timestamp = self.timestamp[x_begin_index:x_end_index]
        y_timestamp = self.timestamp[y_begin_index:y_end_index]

        # x.requires_grad_(True)
        # y.requires_grad_(True)
        # x_timestamp.requires_grad_(True)
        # y_timestamp.requires_grad_(True)
        return x, y, x_timestamp, y_timestamp

    def __len__(self):
        return len(self.time_series_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        data = self.scaler.inverse_transform(data)
        return data.cpu().detach().numpy()

    @property
    def num_features(self):
        return self.time_series_x.shape[1]

    @property
    def columns(self):
        return self.feature_names
    
class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        path,
        split='train',
        seq_len=24 * 4 * 4,
        label_len=24 * 4,
        pred_len=24 * 4,
        scale=True,
        inverse=False,
        random_state=42,
        is_timeencoded=True,
        frequency='d',
        features='MS',
        target='BP',
    ):
        assert split in ['train', 'val', 'test']
        self.path = path
        self.split = split
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scale = scale
        self.inverse = inverse
        self.random_state = random_state
        self.is_timeencoded = is_timeencoded
        self.frequency = frequency
        self.features = features
        self.target = target

        self.scaler = StandardScaler()
        self.prepare_data()

    def prepare_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.path)
        df['Date'] = pd.to_datetime(df['Date'])

        # Split data into train, validation, and test sets
        indices = df.index.tolist()
        train_end = int(len(indices) * 0.6)
        val_end = train_end + int(len(indices) * 0.2)

        if self.split == 'train':
            split_indices = indices[:train_end]
        elif self.split == 'val':
            split_indices = indices[train_end:val_end]
        else:  # 'test'
            split_indices = indices[val_end:]

        df_split = df.loc[split_indices]
        data_columns = df_split.columns[1:]  # Exclude the 'Date' column
        data = df_split[data_columns].values

        if self.scale:
            train_data = df.loc[indices[:train_end], data_columns].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        if self.inverse:
            data_y = df_split[data_columns].values
        else:
            data_y = data

        # Process time features
        timestamp = df_split[['Date']]
        if not self.is_timeencoded:
            timestamp['month'] = timestamp['Date'].dt.month
            timestamp['day'] = timestamp['Date'].dt.day
            timestamp['weekday'] = timestamp['Date'].dt.weekday
            if self.frequency in ['h', 't']:
                timestamp['hour'] = timestamp['Date'].dt.hour
            if self.frequency == 't':
                timestamp['minute'] = (timestamp['Date'].dt.minute // 15)
            timestamp_data = timestamp.drop(columns=['Date']).values
        else:
            timestamp_data = time_features(pd.to_datetime(timestamp['Date']), freq=self.frequency).transpose(1, 0)

        # Convert data to tensors
        self.time_series_x = torch.FloatTensor(data)
        self.time_series_y = torch.FloatTensor(data_y)
        self.timestamp = torch.FloatTensor(timestamp_data)
        self.feature_names = data_columns

    def __getitem__(self, index):
        x_begin_index = index
        x_end_index = x_begin_index + self.seq_len
        y_begin_index = x_end_index - self.label_len
        y_end_index = y_begin_index + self.label_len + self.pred_len

        x = self.time_series_x[x_begin_index:x_end_index]
        y = self.time_series_y[y_begin_index:y_end_index]
        x_timestamp = self.timestamp[x_begin_index:x_end_index]
        y_timestamp = self.timestamp[y_begin_index:y_end_index]

        return x, y, x_timestamp, y_timestamp

    def __len__(self):
        return len(self.time_series_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    @property
    def num_features(self):
        return self.time_series_x.shape[1]

    @property
    def columns(self):
        return self.feature_names

    
class AnomalyDataset(Dataset):
    def __init__(self, path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.prepare_data()
    def prepare_data(self):
        # Load and preprocess data
        df = pd.read_csv(self.path)
        df['Date'] = pd.to_datetime(df['Date'])

        # Split data into train, validation, and test sets
        indices = df.index.tolist()
        train_end = int(len(indices) * 0.6)
        val_end = train_end + int(len(indices) * 0.2)

        if self.split == 'train':
            split_indices = indices[:train_end]
        elif self.split == 'val':
            split_indices = indices[train_end:val_end]
        else:  # 'test'
            split_indices = indices[val_end:]

        df_split = df.loc[split_indices]
        data_columns = df_split.columns[1:]  # Exclude the 'Date' column
        data = df_split[data_columns].values

        if self.scale:
            train_data = df.loc[indices[:train_end], data_columns].values
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        if self.inverse:
            data_y = df_split[data_columns].values
        else:
            data_y = data

        # Process time features
        timestamp = df_split[['Date']]
        if not self.is_timeencoded:
            timestamp['month'] = timestamp['Date'].dt.month
            timestamp['day'] = timestamp['Date'].dt.day
            timestamp['weekday'] = timestamp['Date'].dt.weekday
            if self.frequency in ['h', 't']:
                timestamp['hour'] = timestamp['Date'].dt.hour
            if self.frequency == 't':
                timestamp['minute'] = (timestamp['Date'].dt.minute // 15)
            timestamp_data = timestamp.drop(columns=['Date']).values
        else:
            timestamp_data = time_features(pd.to_datetime(timestamp['Date']), freq=self.frequency).transpose(1, 0)

        # Convert data to tensors
        self.time_series_x = torch.FloatTensor(data)
        self.time_series_y = torch.FloatTensor(data_y)
        self.timestamp = torch.FloatTensor(timestamp_data)
        self.feature_names = data_columns


    def __len__(self):
        if self.mode == 'thre':
            return len(self.time_series_x -self.win_size) // self.win_size + 1
        else:
            return len(self.time_series_x -self.win_size) // self.step + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == 'test':
            return np.float32(self.time_series_x[index:index + self.win_size]), np.float32(self.test_labels[index:index + self.win_size])
        elif self.mode == 'thre':
            return np.float32(self.time_series_x[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
        else: 
            return np.float32(self.time_series_x[index:index + self.win_size]), np.float32(self.test_labels[:self.win_size])
            
if __name__== "__main__":
    data_set = TimeSeriesDataset(
        path='data/sbk_ad_selected.csv'
    )
    data_loader = DataLoader(
        data_set,
        batch_size=32,
        shuffle=True,
        num_workers= 0,
        drop_last=True
    )
    for batch_idx, (x, y, x_timestamp, y_timestamp) in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"x shape: {x.shape}")
        print(f"y shape: {y.shape}")
        print(f"x_timestamp shape: {x_timestamp.shape}")
        print(f"y_timestamp shape: {y_timestamp.shape}")
        break

