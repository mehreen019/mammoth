"""
Quick test script to verify NLP setup is correct.
Run this before the main training to catch issues early.
"""

import sys

def test_imports():
    """Test if all required libraries are available"""
    print("Testing imports...")

    errors = []

    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        errors.append("torch")
        print("✗ PyTorch not found")

    # Test transformers
    try:
        import transformers
        print(f"✓ Transformers {transformers.__version__}")
    except ImportError:
        errors.append("transformers")
        print("✗ Transformers not found")

    # Test datasets
    try:
        import datasets
        print(f"✓ Datasets {datasets.__version__}")
    except ImportError:
        errors.append("datasets")
        print("✗ Datasets not found")

    # Test tokenizers
    try:
        import tokenizers
        print(f"✓ Tokenizers {tokenizers.__version__}")
    except ImportError:
        errors.append("tokenizers")
        print("✗ Tokenizers not found")

    if errors:
        print(f"\n❌ Missing libraries: {', '.join(errors)}")
        print(f"Install with: pip install {' '.join(errors)}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True

def test_backbone():
    """Test if BERT backbone can be imported"""
    print("\nTesting BERT backbone...")

    try:
        from backbone.bert_multilingual import BERTMultilingualBackbone
        print("✓ BERT backbone import successful")
        return True
    except Exception as e:
        print(f"✗ BERT backbone import failed: {e}")
        return False

def test_dataset():
    """Test if Hindi-Bangla dataset can be imported"""
    print("\nTesting Hindi-Bangla NER dataset...")

    try:
        from datasets.seq_hindi_bangla_ner import SequentialHindiBanglaNER
        print("✓ Dataset import successful")
        return True
    except Exception as e:
        print(f"✗ Dataset import failed: {e}")
        return False

def test_model():
    """Test if ER-NLP model can be imported"""
    print("\nTesting ER-NLP model...")

    try:
        from models.er_nlp import ErNlp
        print("✓ ER-NLP model import successful")
        return True
    except Exception as e:
        print(f"✗ ER-NLP model import failed: {e}")
        return False

def test_bert_download():
    """Test if BERT model can be downloaded"""
    print("\nTesting BERT model download...")

    try:
        from transformers import AutoModel
        print("Downloading bert-base-multilingual-cased (this may take a while)...")
        model = AutoModel.from_pretrained('bert-base-multilingual-cased')
        print("✓ BERT model downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ BERT download failed: {e}")
        return False

def main():
    print("="*60)
    print("NLP Setup Test for Hindi-Bangla Continual Learning")
    print("="*60)
    print()

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Backbone", test_backbone()))
    results.append(("Dataset", test_dataset()))
    results.append(("Model", test_model()))
    results.append(("BERT Download", test_bert_download()))

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")

    all_passed = all(passed for _, passed in results)

    print("="*60)
    if all_passed:
        print("\n🎉 All tests passed! You're ready to run the demo.")
        print("\nRun the quick demo with:")
        print("python main.py --model er_nlp --dataset seq-hindi-bangla-ner --buffer_size 200 --batch_size 16 --n_epochs 3 --lr 0.00005")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues above before running.")
        print("Refer to HINDI_BANGLA_NER_GUIDE.md for installation instructions.")

    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
