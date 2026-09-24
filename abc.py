import subprocess
import sys

# This forces the exact Python running IDLE to install the package
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])
print("Installation complete! You can close this now.")
