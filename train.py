# -*- coding: utf-8 -*-
import os
import random
import argparse
import numpy as np
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_norm_

from core.models import MetroODPredictionNet
from data.data_loader import read_and_generate_dataset, StandardScaler_Torch
from utils.evaluator import evaluate, evaluate_val


class StepLR2(MultiStepLR):
    """StepLR scheduler with minimum learning rate constraint."""
    
    def __init__(self, optimizer, milestones, gamma=0.1, 
                last_epoch=-1, min_lr=2.0e-6):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.last_epoch = last_epoch
        self.min_lr = min_lr
        super(StepLR2, self).__init__(optimizer, milestones, gamma)
    
    def get_lr(self):
        lr_candidate = super(StepLR2, self).get_lr()
        if isinstance(lr_candidate, list):
            for i in range(len(lr_candidate)):
                lr_candidate[i] = max(self.min_lr, lr_candidate[i])
        else:
            lr_candidate = max(self.min_lr, lr_candidate)
        return lr_candidate


def seed_torch(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Device for training')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='Batch size during training')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                       help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=50, 
                       help='Number of training epochs')
    parser.add_argument('--data_path', type=str, 
                       default='./data', help='Path to data directory')
    parser.add_argument('--save_path', type=str, 
                       default='./checkpoints', help='Path to save checkpoints')
    parser.add_argument('--seed', type=int, default=12, 
                       help='Random seed')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, 
                       help='Maximum gradient norm for clipping')
    
    return parser.parse_args()


def load_data(args):
    """Load and preprocess data."""
    
    Metro_edge_matrix = np.load(os.path.join(
        args.data_path, 'raw_edge80_in_day.npy'))
    Metro_edge_matrix_out = np.load(os.path.join(
        args.data_path, 'raw_edge80_out_day.npy'))
    time_matrix = np.load(os.path.join(
        args.data_path, 'time_Matrix_regularization.npy'))
    raw_od_out_distribution = np.load(os.path.join(
        args.data_path, 'raw_od_out_distribution.npy'))
    
    raw_edge_in_matrices = [
        np.load(os.path.join(args.data_path, f'raw_edge80_in_matrix{i}.npy'))
        for i in range(1, 7)]
    
    raw_flow_delay_expands = [
        np.load(os.path.join(
            args.data_path, f'raw_flow80_in_day_delay{i}_version_expand.npy'))
        for i in range(1, 7)]
    
    raw_od_distributions = [
        np.load(os.path.join(args.data_path, f'raw_od_distribution{i}.npy'))
        for i in range(1, 7)]
    
    return (Metro_edge_matrix, Metro_edge_matrix_out, time_matrix,
           raw_od_out_distribution, raw_edge_in_matrices,
           raw_flow_delay_expands, raw_od_distributions)


def train_epoch(model, train_loader, optimizer, criterion, scaler_torch, 
               device, max_grad_norm):
    """Train for one epoch."""
    model.train()
    train_losses = []
    
    for _, (x_od_week, x_od_day, x_od_hour, x_od_out_week, y_od, y_od_out) in enumerate(train_loader):
        x_od_week = torch.tensor(x_od_week, dtype=torch.float, device=device)
        x_od_day = torch.tensor(x_od_day, dtype=torch.float, device=device)
        x_od_hour = torch.tensor(x_od_hour, dtype=torch.float, device=device)
        x_od_out_week = torch.tensor(x_od_out_week, dtype=torch.float, device=device)
        y_od = torch.tensor(y_od, dtype=torch.float, device=device)
        y_od_out = torch.tensor(y_od_out, dtype=torch.float, device=device)
        
        optimizer.zero_grad()
        
        y_od_pred = model([x_od_week, x_od_day, x_od_hour, x_od_out_week],
                         [y_od, y_od_out])
        
        y_od_pred = scaler_torch.inverse_transform(y_od_pred)
        y_od = scaler_torch.inverse_transform(y_od)
        
        loss = criterion(y_od_pred, y_od)
        
        loss.backward()
        clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        train_losses.append(loss.item())
    
    return np.mean(train_losses)


def main():
    """Main training procedure."""
    args = parse_args()
    
    seed_torch(args.seed)
    print(f'Random seed: {args.seed}')
    
    device = torch.device(args.device)
    
    print("Loading data...")
    (Metro_edge_matrix, Metro_edge_matrix_out, time_matrix,
     raw_od_out_distribution, raw_edge_in_matrices,
     raw_flow_delay_expands, raw_od_distributions) = load_data(args)
    
    dataset = read_and_generate_dataset(
        Metro_edge_matrix, Metro_edge_matrix_out, time_matrix,
        raw_od_out_distribution, raw_edge_in_matrices,
        raw_flow_delay_expands, raw_od_distributions,
        num_of_weeks=2, num_of_days=1, num_of_hours=2,
        num_for_predict=6, points_per_hour=6,
        batch_size=args.batch_size, test_batch_size=args.batch_size,
        merge=False)
    
    print("Initializing model...")
    model = MetroODPredictionNet(device)
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
    
    scaler = dataset['scaler']
    scaler_torch = StandardScaler_Torch(scaler.mean, scaler.std, device=device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, eps=1.0e-4)
    
    steps = [60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280]
    scheduler = StepLR2(optimizer=optimizer, milestones=steps, 
                       gamma=0.5, min_lr=1.0e-12)
    
    os.makedirs(args.save_path, exist_ok=True)
    
    best_val_mae = np.inf
    best_test_mae = np.inf
    best_test_rmse = np.inf
    best_mae_epoch = 0
    best_rmse_epoch = 0
    
    val_mae_history = []
    train_time_history = []
    
    print("Starting training...")
    for epoch in range(1, args.epochs + 1):
        start_time_train = time()
        
        dataset['train_loader'].shuffle()
        train_iterator = dataset['train_loader'].get_iterator()
        
        train_loss = train_epoch(model, train_iterator, optimizer, criterion,
                                scaler_torch, device, args.max_grad_norm)
        
        end_time_train = time()
        train_time = end_time_train - start_time_train
        train_time_history.append(train_time)
        
        print(f'Epoch: {epoch}, Training Loss: {train_loss:.4f}, '
              f'Time: {train_time:.2f}s')
        
        val_mae, val_rmse = evaluate_val(model=model, dataset=dataset,
                                        dataset_type='val', device=device,
                                        seq_Len=6, horizon=6, output_dim=80,
                                        detail=False)
        
        scheduler.step()
        
        if (epoch + 1) % 1 == 0:
            test_mae, test_rmse = evaluate(model=model, dataset=dataset,
                                          dataset_type='test', device=device,
                                          seq_Len=6, horizon=6, output_dim=80)
        
        val_mae_history.append(val_mae)
        
        if test_mae < best_test_mae:
            best_test_mae = test_mae
            best_mae_epoch = epoch
        
        if test_rmse < best_test_rmse:
            best_test_rmse = test_rmse
            best_rmse_epoch = epoch
        
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            params_filename = os.path.join(args.save_path,
                                          f'best_model_epoch_{epoch}_{val_mae:.4f}.pth')
            torch.save(model.state_dict(), params_filename)
            print(f'Saved best model to: {params_filename}')
    
    print("\n" + "="*50)
    print("Training finished")
    print(f"Average training time per epoch: {np.mean(train_time_history):.2f}s")
    
    bestid = np.argmin(val_mae_history)
    print(f"Best validation MAE at epoch {bestid + 1}: {val_mae_history[bestid]:.4f}")
    print(f"Best test MAE at epoch {best_mae_epoch}: {best_test_mae:.4f}")
    print(f"Best test RMSE at epoch {best_rmse_epoch}: {best_test_rmse:.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
