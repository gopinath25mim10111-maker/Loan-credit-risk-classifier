import subprocess
import sys

print("=" * 50)
print("LOAN CREDIT RISK CLASSIFIER")
print("=" * 50)

print("\nTraining models...")
subprocess.run([sys.executable, "src/train_model.py"], check=True)

print("\nEvaluating models...")
subprocess.run([sys.executable, "src/evaluate_model.py"], check=True)

print("\nProject completed successfully!")