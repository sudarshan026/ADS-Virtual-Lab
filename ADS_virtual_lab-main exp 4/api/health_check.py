#!/usr/bin/env python
"""Quick health check for ADS Virtual Lab backend"""

import sys
import os

print("=" * 60)
print("ADS Virtual Lab - Backend Health Check")
print("=" * 60)

# Check Python version
print(f"\n✓ Python {sys.version.split()[0]}")

# Check required packages
required_packages = [
    'flask', 'flask_cors', 'pandas', 'numpy', 'sklearn',
    'xgboost', 'scipy'
]

print("\nChecking dependencies...")
missing = []

for package in required_packages:
    try:
        __import__(package if package != 'sklearn' else 'sklearn')
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} (MISSING)")
        missing.append(package)

if missing:
    print(f"\n❌ Missing packages: {', '.join(missing)}")
    print(f"\nInstall with:")
    print(f"  pip install -r requirements.txt")
    sys.exit(1)

# Check dataset
print("\nChecking dataset...")
if os.path.exists("../adult.csv"):
    size_mb = os.path.getsize("../adult.csv") / (1024 * 1024)
    print(f"  ✓ adult.csv ({size_mb:.2f} MB)")
else:
    print(f"  ✗ adult.csv (NOT FOUND)")
    sys.exit(1)

# Try loading modules
print("\nChecking Flask app modules...")
try:
    from app import app
    print(f"  ✓ Flask app loaded")
except Exception as e:
    print(f"  ✗ Flask app failed: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All checks passed! Backend is ready.")
print("=" * 60)
print("\nStart the API with:")
print("  python app.py")
print("\nThe API will run on http://localhost:5000")
