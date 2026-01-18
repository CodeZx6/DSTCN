# -*- coding: utf-8 -*-
import numpy as np
import torch


def mean_absolute_percentage_error(y_pred, y_true):
    """MAPE metric with zero-value handling."""
    idx = np.nonzero(y_true)
    return np.mean(np.abs((y_true[idx] - y_pred[idx]) / y_true[idx]))


def get_MSE(pred, real):
    """Mean Squared Error."""
    return np.mean(np.power(real - pred, 2))


def get_MAE(pred, real):
    """Mean Absolute Error."""
    return np.mean(np.abs(real - pred))


def get_RMSE(pred, real):
    """Root Mean Squared Error."""
    return np.sqrt(get_MSE(pred=pred, real=real))


def compute_errors(pred, real):
    """Compute comprehensive error metrics."""
    mse = get_MSE(pred, real)
    mae = get_MAE(pred, real)
    mape = mean_absolute_percentage_error(pred.flatten(), real.flatten())
    rmse = np.sqrt(mse)
    return rmse, mae, mape


def masked_rmse_np(preds, labels, null_val=np.nan):
    """Masked RMSE for handling missing values."""
    return np.sqrt(masked_mse_np(preds=preds, labels=labels, null_val=null_val))


def masked_mse_np(preds, labels, null_val=np.nan):
    """Masked MSE for handling missing values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        rmse = np.square(np.subtract(preds, labels)).astype('float32')
        rmse = np.nan_to_num(rmse * mask)
        return np.mean(rmse)


def masked_mae_np(preds, labels, null_val=np.nan, mode='dcrnn'):
    """Masked MAE for handling missing values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(preds, labels)).astype('float32')
        mae = np.nan_to_num(mae * mask)
        if mode == 'dcrnn':
            return np.mean(mae)
        else:
            return np.mean(mae, axis=(0, 1))


def masked_mape_np(preds, labels, null_val=np.nan):
    """Masked MAPE for handling missing values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(labels)
        else:
            mask = np.not_equal(labels, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(
            preds, labels).astype('float32'), labels))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)


def MAE_torch(pred, true, mask_value=None):
    """PyTorch implementation of MAE with optional masking."""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MSE_torch(pred, true, mask_value=None):
    """PyTorch implementation of MSE with optional masking."""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def RMSE_torch(pred, true, mask_value=None):
    """PyTorch implementation of RMSE with optional masking."""
    return torch.sqrt(MSE_torch(pred, true, mask_value))
