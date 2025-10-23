#!/usr/bin/env python3
"""
Quick test to see if HuggingFace libraries can be imported
Run this in Colab to verify the installation
"""

print("Testing HuggingFace imports...")
print("=" * 60)

# Test 1: Normal import
print("\n1. Testing normal import:")
try:
    import datasets
    print(f"   datasets module: {datasets}")
    print(f"   has load_dataset? {hasattr(datasets, 'load_dataset')}")
    if hasattr(datasets, '__file__'):
        print(f"   location: {datasets.__file__}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 2: Using __import__
print("\n2. Testing __import__ with fromlist:")
try:
    hf_datasets = __import__('datasets', fromlist=['load_dataset'], level=0)
    print(f"   datasets module: {hf_datasets}")
    print(f"   has load_dataset? {hasattr(hf_datasets, 'load_dataset')}")
    if hasattr(hf_datasets, '__file__'):
        print(f"   location: {hf_datasets.__file__}")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 3: Transformers
print("\n3. Testing transformers:")
try:
    from transformers import AutoTokenizer
    print(f"   AutoTokenizer: {AutoTokenizer}")
    print(f"   ✅ SUCCESS")
except Exception as e:
    print(f"   ERROR: {e}")

# Test 4: Check what's in sys.modules
print("\n4. Checking sys.modules:")
import sys
datasets_modules = [k for k in sys.modules.keys() if 'datasets' in k.lower()]
print(f"   Modules with 'datasets': {datasets_modules[:5]}")

print("\n" + "=" * 60)
print("If you see 'has load_dataset? False', the libraries are NOT installed properly")
print("Run: !pip install datasets transformers")
