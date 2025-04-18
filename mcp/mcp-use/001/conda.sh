# 1. Create a new environment named "devenv" with Python 3.11 and Node.js
conda create -n devenv python=3.11 nodejs 

# 2. Activate the environment
conda activate devenv 

# 3. Verify Python and Node installations
python --version   # e.g. Python 3.11.x
node --version     # e.g. v22.x.x
npm --version      # e.g. 9.x.x

pip install --no-cache-dir -r requirements.txt

python main.py
