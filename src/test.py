import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

class ModelEvaluator:
    def __init__(self, model, test_loader, device, class_names=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names or [
            'Angry', 'Disgust', 'Fear', 'Happy', 
            'Sad', 'Surprise', 'Neutral'
        ]
        
    def evaluate(self):
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        print("Evaluating model...")
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader):
                images = images.to(self.device)
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(labels.numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def print_classification_metrics(self, predictions, targets):
        print("\n=== Classification Report ===")
        print(classification_report(
            targets, 
            predictions, 
            target_names=self.class_names,
            digits=4
        ))
        
    def plot_confusion_matrix(self, predictions, targets):
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(targets, predictions)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
    def plot_class_distribution(self, targets):
        plt.figure(figsize=(10, 6))
        pd.Series(targets).value_counts().plot(kind='bar')
        plt.title('Class Distribution in Test Set')
        plt.xlabel('Emotion Class')
        plt.ylabel('Count')
        plt.xticks(range(len(self.class_names)), self.class_names, rotation=45)
        plt.tight_layout()
        plt.savefig('class_distribution.png')
        plt.close()
        
    def analyze_misclassifications(self, predictions, targets, probabilities):
        misclassified_idx = np.where(predictions != targets)[0]
        print("\n=== Misclassification Analysis ===")
        print(f"Total misclassified samples: {len(misclassified_idx)}")
        
        # Analyze confidence of misclassifications
        misclassified_probs = probabilities[misclassified_idx]
        misclassified_pred_probs = np.max(misclassified_probs, axis=1)
        
        print(f"\nAverage confidence for misclassifications: {misclassified_pred_probs.mean():.4f}")
        print(f"Median confidence for misclassifications: {np.median(misclassified_pred_probs):.4f}")
        
    def calculate_per_class_metrics(self, predictions, targets):
        per_class_accuracy = {}
        for i, class_name in enumerate(self.class_names):
            mask = targets == i
            if np.sum(mask) > 0:
                accuracy = np.mean(predictions[mask] == targets[mask])
                per_class_accuracy[class_name] = accuracy
        
        print("\n=== Per-Class Accuracy ===")
        for class_name, accuracy in per_class_accuracy.items():
            print(f"{class_name}: {accuracy:.4f}")
            
    def run_evaluation(self):
        predictions, targets, probabilities = self.evaluate()
        
        # Print basic metrics
        self.print_classification_metrics(predictions, targets)
        
        # Generate visualizations
        self.plot_confusion_matrix(predictions, targets)
        self.plot_class_distribution(targets)
        
        # Detailed analysis
        self.analyze_misclassifications(predictions, targets, probabilities)
        self.calculate_per_class_metrics(predictions, targets)
        
        return predictions, targets, probabilities

def test_model(model_path, test_loader, device):
    """
    Main function to test the model
    """
    # Load model
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # Create evaluator
    evaluator = ModelEvaluator(model, test_loader, device)
    
    # Run evaluation
    predictions, targets, probabilities = evaluator.run_evaluation()
    
    return {
        'predictions': predictions,
        'targets': targets,
        'probabilities': probabilities
    }


