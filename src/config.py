class Config:
    # Data
    DATA_DIR = "data/fer2013"
    IMAGE_SIZE = 48
    NUM_CLASSES = 7
    
    # Training
    BATCH_SIZE = 64  # Increased batch size
    EPOCHS = 200     # More epochs
    LEARNING_RATE = 3e-4  # Modified learning rate
    TRAIN_SPLIT = 0.9
    
    # Model
    CHANNELS = 1
    
    # Early Stopping
    PATIENCE = 15    # Increased patience
    MIN_DELTA = 0.001
    
    # Optimizer
    WEIGHT_DECAY = 1e-5  # L2 regularization
    
    # Learning Rate Scheduler
    LR_STEP_SIZE = 30
    LR_GAMMA = 0.1
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
