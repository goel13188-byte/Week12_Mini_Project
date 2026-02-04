import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import torch

def plot_confusion_matrix(y_true, y_pred, classes):
    """
    Generates and saves a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Model Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved to models/confusion_matrix.png")

def get_model_summary(model):
    """
    Returns the number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)