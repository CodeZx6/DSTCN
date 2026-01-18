"""
Metro OD Matrix Prediction Framework

A sophisticated deep learning framework for metro Origin-Destination (OD) 
matrix prediction with temporal-spatial attention mechanisms and delayed 
flow completion strategies.
"""

__version__ = '1.0.0'
__author__ = 'Research Team'

from core import MetroODPredictionNet
from data import read_and_generate_dataset, DataLoader
from utils import evaluate, evaluate_val

__all__ = [
    'MetroODPredictionNet',
    'read_and_generate_dataset',
    'DataLoader',
    'evaluate',
    'evaluate_val'
]
