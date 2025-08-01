# 🚀 MLX Reinforcement Learning Dashboard

A reinforcement learning project using Apple's MLX framework with automated training and results visualization via GitHub Actions and GitHub Pages.

## ✨ Features

- 🧠 MLX-powered RL training optimized for Apple Silicon
- 📊 Automated results visualization with interactive charts
- 🔄 GitHub Actions CI/CD pipeline with UV package management
- 📱 Modern responsive web dashboard via GitHub Pages
- ⚡ Lightning-fast dependency management with UV

## 🚀 Quick Start

### Prerequisites
- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.12 (NOT 3.13 - MLX doesn't support it yet)
- UV package manager

### Local Development
```bash
# Clone the repository
git clone https://github.com/yourusername/mlx-rl-dashboard.git
cd mlx-rl-dashboard

# Create virtual environment with UV
uv venv --python 3.12
source .venv/bin/activate

# Install dependencies
uv pip install mlx numpy tqdm matplotlib gymnasium plotly jinja2

# Run training locally
python src/train.py --episodes 100
