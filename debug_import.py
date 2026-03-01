import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    from training.10_train_focal_loss import FocalLoss
    print("FocalLoss imported successfully")
except Exception as e:
    print(f"Error importing FocalLoss: {e}")
    import traceback
    traceback.print_exc()
