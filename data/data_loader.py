# -*- coding: utf-8 -*-
import pickle
import numpy as np
from scipy.sparse.linalg import eigs
import torch


class StandardScaler:
    """Standard scaler for data normalization."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class StandardScaler_Torch:
    """PyTorch version of standard scaler."""
    
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(data=mean, dtype=torch.float, device=device)
        self.std = torch.tensor(data=std, dtype=torch.float, device=device)
        self.mean = mean
        self.std = std
    
    def transform(self, data):
        return (data - self.mean) / self.std
    
    def inverse_transform(self, data):
        return (data * self.std) + self.mean


class DataLoader:
    """Custom data loader for OD matrix sequences."""
    
    def __init__(self, x_od_week, x_od_day, x_od_hour, x_od_out_week, 
                 y_od, y_od_out, batch_size, pad_with_last_sample=True, 
                 shuffle=False):
        self.batch_size = batch_size
        self.current_ind = 0
        
        if pad_with_last_sample:
            num_padding = (batch_size - (len(x_od_week) % batch_size)) % batch_size
            x_od_week_padding = np.repeat(x_od_week[-1:], num_padding, axis=0)
            x_od_day_padding = np.repeat(x_od_day[-1:], num_padding, axis=0)
            x_od_hour_padding = np.repeat(x_od_hour[-1:], num_padding, axis=0)
            x_od_out_week_padding = np.repeat(x_od_out_week[-1:], num_padding, axis=0)
            y_od_padding = np.repeat(y_od[-1:], num_padding, axis=0)
            y_od_out_padding = np.repeat(y_od_out[-1:], num_padding, axis=0)
            
            x_od_week = np.concatenate([x_od_week, x_od_week_padding], axis=0)
            x_od_day = np.concatenate([x_od_day, x_od_day_padding], axis=0)
            x_od_hour = np.concatenate([x_od_hour, x_od_hour_padding], axis=0)
            x_od_out_week = np.concatenate([x_od_out_week, x_od_out_week_padding], axis=0)
            y_od = np.concatenate([y_od, y_od_padding], axis=0)
            y_od_out = np.concatenate([y_od_out, y_od_out_padding], axis=0)
        
        self.size = len(x_od_week)
        self.num_batch = int(self.size // self.batch_size)
        self.x_od_week = x_od_week
        self.x_od_day = x_od_day
        self.x_od_hour = x_od_hour
        self.x_od_out_week = x_od_out_week
        self.y_od = y_od
        self.y_od_out = y_od_out
    
    def shuffle(self):
        permutation = np.random.permutation(self.size)
        self.x_od_week = self.x_od_week[permutation]
        self.x_od_day = self.x_od_day[permutation]
        self.x_od_hour = self.x_od_hour[permutation]
        self.x_od_out_week = self.x_od_out_week[permutation]
        self.y_od = self.y_od[permutation]
        self.y_od_out = self.y_od_out[permutation]
    
    def get_iterator(self):
        self.current_ind = 0
        
        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_od_i_week = self.x_od_week[start_ind:end_ind, ...]
                x_od_i_day = self.x_od_day[start_ind:end_ind, ...]
                x_od_i_hour = self.x_od_hour[start_ind:end_ind, ...]
                x_od_out_i_week = self.x_od_out_week[start_ind:end_ind, ...]
                y_od_i = self.y_od[start_ind:end_ind, ...]
                y_od_out_i = self.y_od_out[start_ind:end_ind, ...]
                yield (x_od_i_week, x_od_i_day, x_od_i_hour, 
                      x_od_out_i_week, y_od_i, y_od_out_i)
                self.current_ind += 1
        
        return _wrapper()


def scaled_Laplacian(W):
    """Compute scaled Laplacian matrix."""
    assert W.shape[0] == W.shape[1]
    
    D = np.diag(np.sum(W, axis=1))
    L = D - W
    lambda_max = eigs(L, k=1, which='LR')[0].real
    
    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    """Compute Chebyshev polynomials."""
    N = L_tilde.shape[0]
    cheb_polynomials = [np.identity(N), L_tilde.copy()]
    
    for i in range(2, K):
        cheb_polynomials.append(
            2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
    
    return cheb_polynomials


def load_pickle(pickle_file):
    """Load pickle file with error handling."""
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data', pickle_file, ':', e)
        raise
    return pickle_data


def load_graph_data(pkl_filename):
    """Load adjacency matrix from pickle file."""
    adj_mx = load_pickle(pkl_filename)
    return adj_mx.astype(np.float32)


def search_data(sequence_length, num_of_batches, label_start_idx,
                num_for_predict, units, points_per_hour):
    """Search for valid data indices in temporal sequences."""
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    
    if label_start_idx + num_for_predict > sequence_length:
        return None
    
    x_idx = []
    for i in range(1, num_of_batches + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            return None
    
    if len(x_idx) != num_of_batches:
        return None
    
    return x_idx[::-1]


def search_datah(sequence_length, num_of_batches, label_start_idx,
                 num_for_predict, units, points_per_hour):
    """Search for hourly data indices."""
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")
    
    if label_start_idx + num_for_predict > sequence_length:
        return None
    
    x_idx = []
    for i in range(1, num_of_batches + 1):
        if label_start_idx - 6 >= 0:
            x_idx.append((label_start_idx - 6, label_start_idx - 5, 
                         label_start_idx - 4, label_start_idx - 3, 
                         label_start_idx - 2, label_start_idx - 1))
        else:
            return None
    
    if len(x_idx) != num_of_batches:
        return None
    
    return x_idx[0]


def complete_od_matrix(time_matrix, raw_od_out_distribution, 
                      raw_edge_in_matrices, raw_flow_delay_expands,
                      raw_od_distributions, label_start_idx, hour_indices):
    """Complete OD matrix using delayed flow distribution."""
    index6, index5, index4, index3, index2, index1 = hour_indices
    
    P6 = raw_od_distributions[5][index6 - 756]
    P5 = raw_od_distributions[4][index5 - 756]
    P4 = raw_od_distributions[3][index4 - 756]
    P3 = raw_od_distributions[2][index3 - 756]
    P2 = raw_od_distributions[1][index2 - 756]
    P1 = raw_od_distributions[0][index1 - 756]
    
    for n1 in range(80):
        for n2 in range(80):
            n = int(time_matrix[n1][n2])
            od_out_t = raw_od_out_distribution[label_start_idx - 1][n1][n2]
            od_out_t_week = raw_od_out_distribution[label_start_idx - 1 - 756][n1][n2]
            
            if od_out_t_week == 0:
                continue
            else:
                od_t6 = P6[n1][n2]
                od_t5 = P5[n1][n2]
                od_t4 = P4[n1][n2]
                od_t3 = P3[n1][n2]
                od_t2 = P2[n1][n2]
                od_t1 = P1[n1][n2]
                proportion = (od_out_t - od_out_t_week) / od_out_t_week
                
                if n == 6:
                    P1[n1][n2] = od_t1 * (1 + proportion)
                    P2[n1][n2] = od_t2 * (1 + proportion)
                    P3[n1][n2] = od_t3 * (1 + proportion)
                    P4[n1][n2] = od_t4 * (1 + proportion)
                    P5[n1][n2] = od_t5 * (1 + proportion)
                    P6[n1][n2] = od_t6 * (1 + proportion)
                elif n == 5 and od_t5 != 0:
                    P1[n1][n2] = od_t1 * (1 + proportion)
                    P2[n1][n2] = od_t2 * (1 + proportion)
                    P3[n1][n2] = od_t3 * (1 + proportion)
                    P4[n1][n2] = od_t4 * (1 + proportion)
                    P5[n1][n2] = od_t5 * (1 + proportion)
                elif n == 4 and od_t4 != 0:
                    P1[n1][n2] = od_t1 * (1 + proportion)
                    P2[n1][n2] = od_t2 * (1 + proportion)
                    P3[n1][n2] = od_t3 * (1 + proportion)
                    P4[n1][n2] = od_t4 * (1 + proportion)
                elif n == 3 and od_t3 != 0:
                    P1[n1][n2] = od_t1 * (1 + proportion)
                    P2[n1][n2] = od_t2 * (1 + proportion)
                    P3[n1][n2] = od_t3 * (1 + proportion)
                elif n == 2 and od_t2 != 0:
                    P1[n1][n2] = od_t1 * (1 + proportion)
                    P2[n1][n2] = od_t2 * (1 + proportion)
                elif n == 1 and od_t1 != 0:
                    P1[n1][n2] = od_t1 * (1 + proportion)
                else:
                    continue
    
    for i in range(80):
        for P in [P6, P5, P4, P3, P2, P1]:
            sum_p = P[i].sum()
            if sum_p != 0:
                for j in range(80):
                    if P[i][j] != 0:
                        P[i][j] = P[i][j] / sum_p
    
    MD6 = raw_flow_delay_expands[5][index6] * P6
    MC6 = raw_edge_in_matrices[5][index6] + MD6
    MD5 = raw_flow_delay_expands[4][index5] * P5
    MC5 = raw_edge_in_matrices[4][index5] + MD5
    MD4 = raw_flow_delay_expands[3][index4] * P4
    MC4 = raw_edge_in_matrices[3][index4] + MD4
    MD3 = raw_flow_delay_expands[2][index3] * P3
    MC3 = raw_edge_in_matrices[2][index3] + MD3
    MD2 = raw_flow_delay_expands[1][index2] * P2
    MC2 = raw_edge_in_matrices[1][index2] + MD2
    MD1 = raw_flow_delay_expands[0][index1] * P1
    MC1 = raw_edge_in_matrices[0][index1] + MD1
    
    MC6, MC5, MC4, MC3, MC2, MC1 = (np.expand_dims(MC6, 0), 
                                    np.expand_dims(MC5, 0), 
                                    np.expand_dims(MC4, 0),
                                    np.expand_dims(MC3, 0), 
                                    np.expand_dims(MC2, 0), 
                                    np.expand_dims(MC1, 0))
    
    hour_sample = np.concatenate([MC6, MC5, MC4, MC3, MC2, MC1], axis=0)
    
    return hour_sample


def get_sample_indices(data_sequence, Metro_edge_matrix_out, time_matrix,
                      raw_od_out_distribution, raw_edge_in_matrices,
                      raw_flow_delay_expands, raw_od_distributions,
                      num_of_weeks, num_of_days, num_of_hours,
                      label_start_idx, num_for_predict, points_per_hour=12):
    """Extract sample indices for multi-granularity temporal data."""
    
    week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                              label_start_idx, num_for_predict,
                              7 * 18, points_per_hour)
    if not week_indices:
        return None
    
    day_indices = search_data(data_sequence.shape[0], num_of_days,
                             label_start_idx, num_for_predict,
                             18, points_per_hour)
    if not day_indices:
        return None
    
    hour_indices1 = search_data(data_sequence.shape[0], num_of_hours,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if not hour_indices1:
        return None
    
    hour_indices = search_datah(data_sequence.shape[0], 1,
                               label_start_idx, num_for_predict,
                               1, points_per_hour)
    if not hour_indices:
        return None
    
    hour_sample = complete_od_matrix(time_matrix, raw_od_out_distribution,
                                    raw_edge_in_matrices, raw_flow_delay_expands,
                                    raw_od_distributions, label_start_idx,
                                    hour_indices)
    
    i, j = hour_indices1[0]
    hour_sample1 = data_sequence[i:j]
    hour_sample = np.concatenate([hour_sample1, hour_sample], axis=0)
    
    x_od_week = np.concatenate([data_sequence[i:j] 
                               for i, j in week_indices], axis=0)
    x_od_out_week = np.concatenate([Metro_edge_matrix_out[i:j+6] 
                                   for i, j in week_indices], axis=0)
    
    x_od_day = np.concatenate([data_sequence[i:j] 
                              for i, j in day_indices], axis=0)
    
    x_od_hour = hour_sample
    x_od_hour1 = np.concatenate([data_sequence[i:j] 
                                for i, j in hour_indices1], axis=0)
    
    y_od = data_sequence[label_start_idx:label_start_idx + num_for_predict]
    y_od_out = Metro_edge_matrix_out[label_start_idx:label_start_idx + num_for_predict]
    
    return x_od_week, x_od_out_week, x_od_day, x_od_hour, x_od_hour1, y_od, y_od_out


def read_and_generate_dataset(Metro_edge_matrix, Metro_edge_matrix_out,
                              time_matrix, raw_od_out_distribution,
                              raw_edge_in_matrices, raw_flow_delay_expands,
                              raw_od_distributions, num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict, points_per_hour,
                              batch_size, test_batch_size, merge=False,
                              scaler_axis=(0, 1, 2, 3)):
    """Read data and generate training/validation/test datasets."""
    
    all_samples = []
    
    for idx in range(Metro_edge_matrix.shape[0]):
        sample = get_sample_indices(
            Metro_edge_matrix, Metro_edge_matrix_out, time_matrix,
            raw_od_out_distribution, raw_edge_in_matrices,
            raw_flow_delay_expands, raw_od_distributions,
            num_of_weeks, num_of_days, num_of_hours, idx,
            num_for_predict, points_per_hour)
        
        if not sample:
            continue
        
        x_od_week, x_od_out_week, x_od_day, x_od_hour, x_od_hour1, y_od, y_od_out = sample
        
        all_samples.append((
            np.expand_dims(x_od_week, axis=0),
            np.expand_dims(x_od_out_week, axis=0),
            np.expand_dims(x_od_day, axis=0),
            np.expand_dims(x_od_hour, axis=0),
            np.expand_dims(y_od, axis=0),
            np.expand_dims(y_od_out, axis=0)))
    
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)
    
    if not merge:
        training_set = [np.concatenate(i, axis=0)
                       for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                       for i in zip(*all_samples[:split_line2])]
    
    validation_set = [np.concatenate(i, axis=0)
                     for i in zip(*all_samples[split_line1:split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                  for i in zip(*all_samples[split_line2:])]
    
    train_x_od_week, train_x_od_out_week, train_x_od_day, train_x_od_hour, \
        train_y_od, train_y_od_out = training_set
    val_x_od_week, val_x_od_out_week, val_x_od_day, val_x_od_hour, \
        val_y_od, val_y_od_out = validation_set
    test_x_od_week, test_x_od_out_week, test_x_od_day, test_x_od_hour, \
        test_y_od, test_y_od_out = testing_set
    
    print('Training data: x_od_week: {}, x_od_day: {}, x_od_hour: {}, '
          'x_od_out_week: {}, y_od: {}, y_od_out: {}'.format(
              train_x_od_week.shape, train_x_od_day.shape, train_x_od_hour.shape,
              train_x_od_out_week.shape, train_y_od.shape, train_y_od_out.shape))
    print('Validation data: x_od_week: {}, x_od_day: {}, x_od_hour: {}, '
          'x_od_out_week: {}, y_od: {}, y_od_out: {}'.format(
              val_x_od_week.shape, val_x_od_day.shape, val_x_od_hour.shape,
              val_x_od_out_week.shape, val_y_od.shape, val_y_od_out.shape))
    print('Testing data: x_od_week: {}, x_od_day: {}, x_od_hour: {}, '
          'x_od_out_week: {}, y_od: {}, y_od_out: {}'.format(
              test_x_od_week.shape, test_x_od_day.shape, test_x_od_hour.shape,
              test_x_od_out_week.shape, test_y_od.shape, test_y_od_out.shape))
    
    scaler = StandardScaler(mean=train_x_od_day.mean(axis=scaler_axis),
                           std=train_x_od_day.std(axis=scaler_axis))
    
    train_x_od_week_norm = scaler.transform(train_x_od_week)
    val_x_od_week_norm = scaler.transform(val_x_od_week)
    test_x_od_week_norm = scaler.transform(test_x_od_week)
    
    train_x_od_day_norm = scaler.transform(train_x_od_day)
    val_x_od_day_norm = scaler.transform(val_x_od_day)
    test_x_od_day_norm = scaler.transform(test_x_od_day)
    
    train_x_od_hour_norm = scaler.transform(train_x_od_hour)
    val_x_od_hour_norm = scaler.transform(val_x_od_hour)
    test_x_od_hour_norm = scaler.transform(test_x_od_hour)
    
    train_x_od_out_week_norm = scaler.transform(train_x_od_out_week)
    val_x_od_out_week_norm = scaler.transform(val_x_od_out_week)
    test_x_od_out_week_norm = scaler.transform(test_x_od_out_week)
    
    train_y_od_norm = scaler.transform(train_y_od)
    val_y_od_norm = scaler.transform(val_y_od)
    test_y_od_norm = scaler.transform(test_y_od)
    
    train_y_od_out_norm = scaler.transform(train_y_od_out)
    val_y_od_out_norm = scaler.transform(val_y_od_out)
    test_y_od_out_norm = scaler.transform(test_y_od_out)
    
    data = {}
    data['y_train'] = train_y_od_norm
    data['od_out_y_train'] = train_y_od_out_norm
    data['y_val'] = val_y_od_norm
    data['od_out_y_val'] = val_y_od_out_norm
    data['y_test'] = test_y_od_norm
    data['od_out_y_test'] = test_y_od_out_norm
    
    data['train_loader'] = DataLoader(train_x_od_week_norm, train_x_od_day_norm,
                                      train_x_od_hour_norm, train_x_od_out_week_norm,
                                      train_y_od_norm, train_y_od_out_norm,
                                      batch_size, shuffle=True)
    data['val_loader'] = DataLoader(val_x_od_week_norm, val_x_od_day_norm,
                                    val_x_od_hour_norm, val_x_od_out_week_norm,
                                    val_y_od_norm, val_y_od_out_norm,
                                    test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(test_x_od_week_norm, test_x_od_day_norm,
                                     test_x_od_hour_norm, test_x_od_out_week_norm,
                                     test_y_od_norm, test_y_od_out_norm,
                                     test_batch_size, shuffle=False)
    data['scaler'] = scaler
    
    return data
