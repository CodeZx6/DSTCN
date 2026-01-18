# -*- coding: utf-8 -*-
import torch
import numpy as np
from utils.metrics import mean_absolute_percentage_error, get_MAE, get_RMSE


def run_model(model, data_iterator, device, seq_len, horizon, output_dim):
    """Execute model inference without gradient computation."""
    model.eval()
    y_od_pred_list = []
    
    for _, (x_od_week, x_od_day, x_od_hour, x_od_out_week, y_od, y_od_out) in enumerate(data_iterator):
        x_od_week = torch.tensor(x_od_week, dtype=torch.float, device=device)
        x_od_day = torch.tensor(x_od_day, dtype=torch.float, device=device)
        x_od_hour = torch.tensor(x_od_hour, dtype=torch.float, device=device)
        x_od_out_week = torch.tensor(x_od_out_week, dtype=torch.float, device=device)
        y_od = torch.tensor(y_od, dtype=torch.float, device=device)
        y_od_out = torch.tensor(y_od_out, dtype=torch.float, device=device)
        
        with torch.no_grad():
            y_od_pred = model([x_od_week, x_od_day, x_od_hour, x_od_out_week],
                             [y_od, y_od_out])
            if y_od_pred is not None:
                y_od_pred_list.append(y_od_pred.detach().cpu().numpy())
    
    return y_od_pred_list


def evaluate(model, dataset, dataset_type, device, seq_Len, horizon, 
            output_dim, detail=True, format_result=False):
    """Comprehensive model evaluation with multi-horizon metrics."""
    
    y_od_preds = run_model(model,
                          data_iterator=dataset['{}_loader'.format(dataset_type)].get_iterator(),
                          device=device, seq_len=seq_Len, horizon=horizon,
                          output_dim=output_dim)
    
    evaluate_category = []
    if len(y_od_preds) > 0:
        evaluate_category.append("od")
    
    results = {}
    for category in evaluate_category:
        print(category)
        if category == 'od':
            y_preds = y_od_preds
            scaler = dataset['scaler']
            gt = dataset['y_{}'.format(dataset_type)]
        
        y_preds = np.concatenate(y_preds, axis=0)
        
        horizon = 6
        for horizon_i in range(horizon):
            y_truth = scaler.inverse_transform(gt[:, horizon_i, :, :output_dim])
            y_pred = scaler.inverse_transform(
                y_preds[:y_truth.shape[0], horizon_i, :, :output_dim])
            
            mae = get_MAE(y_pred, y_truth)
            mape = mean_absolute_percentage_error(y_pred.flatten(), y_truth.flatten())
            rmse = get_RMSE(y_pred, y_truth)
            
            print('{} Horizon:{:d}, MAE:{:.6f}, MAPE:{:.6f}, RMSE:{:.6f}'.format(
                dataset_type, horizon_i + 1, mae, mape, rmse))
        
        y_truth = scaler.inverse_transform(gt[:, :, :, :output_dim])
        y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0], :, :, :output_dim])
        
        mae = get_MAE(y_pred, y_truth)
        mape = mean_absolute_percentage_error(y_pred.flatten(), y_truth.flatten())
        rmse = get_RMSE(y_pred, y_truth)
        
        print('{} Average Horizon, MAE:{:.6f}, MAPE:{:.6f}, RMSE:{:.6f}'.format(
            dataset_type, mae, mape, rmse))
        
        results['{}_MAE'.format(category)] = mae
        results['{}_MAPE'.format(category)] = mape
        results['{}_RMSE'.format(category)] = rmse
    
    return mae, rmse


def evaluate_val(model, dataset, dataset_type, device, seq_Len, horizon, 
                output_dim, detail=False):
    """Validation evaluation with reduced output."""
    
    y_od_preds = run_model(model,
                          data_iterator=dataset['{}_loader'.format(dataset_type)].get_iterator(),
                          device=device, seq_len=seq_Len, horizon=horizon,
                          output_dim=output_dim)
    
    evaluate_category = []
    if len(y_od_preds) > 0:
        evaluate_category.append("od")
    
    results = {}
    for category in evaluate_category:
        if category == 'od':
            y_preds = y_od_preds
            scaler = dataset['scaler']
            gt = dataset['y_{}'.format(dataset_type)]
        
        y_preds = np.concatenate(y_preds, axis=0)
        
        y_truth = scaler.inverse_transform(gt[:, :, :, :output_dim])
        y_pred = scaler.inverse_transform(y_preds[:y_truth.shape[0], :, :, :output_dim])
        
        mae = get_MAE(y_pred, y_truth)
        mape = mean_absolute_percentage_error(y_pred.flatten(), y_truth.flatten())
        rmse = get_RMSE(y_pred, y_truth)
        
        print('Val Average Horizon, MAE:{:.6f}, MAPE:{:.6f}, RMSE:{:.6f}'.format(
            mae, mape, rmse))
        
        results['{}_MAE'.format(category)] = mae
        results['{}_MAPE'.format(category)] = mape
        results['{}_RMSE'.format(category)] = rmse
    
    return mae, rmse
