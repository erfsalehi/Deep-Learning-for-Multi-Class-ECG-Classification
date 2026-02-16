# Setup Instructions

## Prerequisites
- Python 3.8+
- [PhysioNet Account](https://physionet.org/register/) (for accessing datasets)

## Installation

1. **Clone the repository** (if using git):
   ```bash
   git clone <repository-url>
   cd cardiovascular-ai-portfolio
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Setup

### PTB-XL Dataset
1. Create directory: `mkdir -p data/raw/ptbxl`
2. Download from PhysioNet:
   ```bash
   wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/ -P data/raw/ptbxl/
   ```
   *Alternatively, download manually and place files in `data/raw/ptbxl/`.*

### MIMIC-III (Project 2)
*Requires credentialed access.*
1. Complete CITI training.
2. Request access on PhysioNet.
3. Download matched subset to `data/raw/mimic3wdb/`.

## Running the Code

### Notebooks
Start Jupyter Lab:
```bash
jupyter lab
```

### Scripts
Run the starter script to verify setup:
```bash
python scripts/00_first_ecg.py
```
