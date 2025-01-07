import torch
import torch.nn as nn
import os
from datetime import datetime
from src.config import Config
from src.dataset import get_dataloaders
from src.model import UNet
from src.train_val import train_epoch, validate_epoch, EarlyStopping
from src.test import ModelEvaluator

class ExperimentManager:
    def __init__(self):
        self.config = Config()
        self.setup_experiment_folder()
        
    def setup_experiment_folder(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = f'experiments/experiment_{timestamp}'
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f'{self.experiment_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{self.experiment_dir}/plots', exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, epoch, best_val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        path = f'{self.experiment_dir}/checkpoints/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['best_val_loss']

def main():
    # Initialize experiment manager
    experiment = ExperimentManager()
    config = experiment.config
    
    print("=== Starting Emotion Detection Training ===")
    print(f"Using device: {config.DEVICE}")
    
    # Get dataloaders
    print("\nPreparing datasets...")
    train_loader, val_loader = get_dataloaders(config)
    
    # Initialize model
    print("\nInitializing model...")
    model = UNet(config.CHANNELS, config.NUM_CLASSES).to(config.DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.LR_STEP_SIZE,
        gamma=config.LR_GAMMA
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        min_delta=config.MIN_DELTA
    )
    
    # Training metrics tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    
    print("\n=== Starting Training ===")
    # Training loop
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        print("-" * 20)
        
        # Training phase
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, config.DEVICE
        )
        
        # Validation phase
        val_loss, val_acc, val_f1 = validate_epoch(
            model, val_loader, criterion, config.DEVICE
        )
        
        # Print metrics
        print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint if validation improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            experiment.save_checkpoint(model, optimizer, epoch, best_val_loss)
            print(f"Saved checkpoint (Best val loss: {best_val_loss:.4f})")
        
        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered")
            break
    
    print("\n=== Training Completed ===")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    # Save final model
    final_model_path = f'{experiment.experiment_dir}/final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")
    
    # Testing phase
    print("\n=== Starting Model Evaluation ===")
    evaluator = ModelEvaluator(model, val_loader, config.DEVICE)
    
    # Run comprehensive evaluation
    predictions, targets, probabilities = evaluator.run_evaluation()
    
    # Save test results
    test_results = {
        'predictions': predictions,
        'targets': targets,
        'probabilities': probabilities,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc
    }
    torch.save(test_results, f'{experiment.experiment_dir}/test_results.pt')
    
    print("\n=== Experiment Completed ===")
    print(f"Results saved in: {experiment.experiment_dir}")

if __name__ == "__main__":
    # Create experiments directory if it doesn't exist
    os.makedirs('experiments', exist_ok=True)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise e
