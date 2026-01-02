import numpy as np
import tensorflow as tf
import torch
import joblib
import os
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class ModelUtils:
    """Utility functions for model training and evaluation"""
    
    @staticmethod
    def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Intersection over Union"""
        intersection = np.logical_and(y_true, y_pred).sum()
        union = np.logical_or(y_true, y_pred).sum()
        
        if union == 0:
            return 0.0
        
        return float(intersection / union)
    
    @staticmethod
    def calculate_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Dice coefficient"""
        intersection = np.logical_and(y_true, y_pred).sum()
        
        if (y_true.sum() + y_pred.sum()) == 0:
            return 1.0
        
        return float(2 * intersection) / float(y_true.sum() + y_pred.sum())
    
    @staticmethod
    def calculate_map(predictions: list, ground_truth: list, iou_threshold: float = 0.5) -> float:
        """Calculate Mean Average Precision"""
        # Simplified implementation
        # In production, use proper mAP calculation
        if not predictions:
            return 0.0
        
        # Count true positives
        true_positives = 0
        for pred in predictions:
            for gt in ground_truth:
                iou = ModelUtils.calculate_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_threshold and pred['class'] == gt['class']:
                    true_positives += 1
                    break
        
        precision = true_positives / len(predictions) if predictions else 0
        recall = true_positives / len(ground_truth) if ground_truth else 0
        
        # Simplified AP calculation
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: list, save_path: str = None):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_training_history(history: dict, save_path: str = None):
        """Plot training history for neural networks"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot loss
        axes[0].plot(history.get('loss', []), label='Training Loss')
        axes[0].plot(history.get('val_loss', []), label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot accuracy
        if 'accuracy' in history:
            axes[1].plot(history['accuracy'], label='Training Accuracy')
            axes[1].plot(history['val_accuracy'], label='Validation Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def save_model(model, path: str, model_type: str = 'tensorflow'):
        """Save trained model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if model_type == 'tensorflow':
            model.save(path)
        elif model_type == 'pytorch':
            torch.save(model.state_dict(), path)
        elif model_type == 'sklearn':
            joblib.dump(model, path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print(f"Model saved to {path}")
    
    @staticmethod
    def load_model(path: str, model_type: str = 'tensorflow'):
        """Load trained model"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if model_type == 'tensorflow':
            return tf.keras.models.load_model(path)
        elif model_type == 'pytorch':
            # Load model architecture first, then weights
            # This requires the model class to be defined
            raise NotImplementedError("PyTorch model loading requires model class definition")
        elif model_type == 'sklearn':
            return joblib.load(path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class EarlyStopping:
    """Early stopping callback for training"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss: float):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

if __name__ == '__main__':
    utils = ModelUtils()
    print("ModelUtils initialized successfully")