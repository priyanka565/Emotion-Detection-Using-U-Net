import torch
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    predictions = []
    targets = []
    
    # Progress bar
    pbar = tqdm(train_loader, desc='Training')
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Mixup augmentation
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        batch_size = images.size()[0]
        index = torch.randperm(batch_size).to(device)
        mixed_x = lam * images + (1 - lam) * images[index, :]
        
        optimizer.zero_grad()
        outputs = model(mixed_x)
        
        loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[index])
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        targets.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    scheduler.step()
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == np.array(targets))
    f1 = f1_score(targets, predictions, average='weighted')
    
    return total_loss / len(train_loader), accuracy, f1

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    # Progress bar
    pbar = tqdm(val_loader, desc='Validation')
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    # Calculate metrics
    accuracy = np.mean(np.array(predictions) == np.array(targets))
    f1 = f1_score(targets, predictions, average='weighted')
    
    return total_loss / len(val_loader), accuracy, f1

class MetricTracker:
    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.train_f1_scores = []
        self.val_losses = []
        self.val_accuracies = []
        self.val_f1_scores = []
    
    def update(self, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.train_f1_scores.append(train_f1)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.val_f1_scores.append(val_f1)
    
    def get_best_metrics(self):
        best_epoch = np.argmin(self.val_losses)
        return {
            'best_epoch': best_epoch,
            'best_val_loss': self.val_losses[best_epoch],
            'best_val_acc': self.val_accuracies[best_epoch],
            'best_val_f1': self.val_f1_scores[best_epoch]
        }
