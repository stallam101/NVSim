#!/bin/bash
# Sierpinski Environment Setup Script

echo "🔧 Setting up Sierpinski testing environment..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo ""
echo "To use:"
echo "  source venv/bin/activate    # Activate environment"
echo "  python3 sierpinski_test.py  # Run tests"
echo "  deactivate                  # When done"