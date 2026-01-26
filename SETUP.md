# Environment Setup

## Prerequisites

- **Python 3.8+** (Python 3.9 or 3.10 recommended)
- **pip** (Python package installer)
- **Git** (for version control)

## Step-by-Step Setup

### 1. Create a Virtual Environment 

Using a virtual environment isolates your project dependencies and prevents conflicts with other projects.

#### Option A: Using `venv` (Built-in, Recommended)

```bash
# Navigate to project directory
cd /home/up202306122/Hackathons/world-model

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### 3. Upgrade pip

Ensure you have the latest version of pip:

```bash
pip install --upgrade pip
```

### 4. Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

This will install:
- **PyTorch** - Deep learning framework for VAE and dynamics models
- **NumPy** - Numerical computing
- **Gymnasium** - Environment framework for the robotic arm
- **SciPy** - Scientific computing utilities
- **Matplotlib & Seaborn** - Visualization libraries
- **tqdm** - Progress bars for training
- **Pillow & torchvision** - Image processing (for optional image observations)