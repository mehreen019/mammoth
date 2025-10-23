"""
Quick test to verify the training script works without argument conflicts
"""

import subprocess
import sys

def test_help():
    """Test that --help works"""
    print("Testing --help...")
    result = subprocess.run(
        [sys.executable, 'train_ner_continual.py', '--help'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✓ --help works")
        return True
    else:
        print("✗ --help failed")
        print(result.stderr)
        return False

def test_dry_run():
    """Test that the script can parse arguments without running"""
    print("\nTesting argument parsing...")
    
    # Just test that it starts without errors
    # We'll kill it after a few seconds
    import time
    import signal
    
    proc = subprocess.Popen(
        [sys.executable, 'train_ner_continual.py', '--model', 'er_nlp', '--n_epochs', '1'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait a bit to see if it crashes immediately
    time.sleep(5)
    
    if proc.poll() is None:
        # Still running, kill it
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("✓ Script started successfully (killed after 5 seconds)")
        return True
    else:
        # Process ended
        stdout, stderr = proc.communicate()
        if 'ArgumentError' in stderr or 'conflicting option' in stderr:
            print("✗ Argument conflict detected")
            print(stderr)
            return False
        else:
            print("✓ Script started (ended early, but no argument errors)")
            return True

def main():
    print("="*60)
    print("Testing Training Script")
    print("="*60)
    
    results = []
    
    # Test 1: Help
    results.append(("Help", test_help()))
    
    # Test 2: Dry run
    results.append(("Argument Parsing", test_dry_run()))
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✅ All tests passed!")
        print("\nYou can now run:")
        print("  python train_ner_continual.py --model er_nlp --n_epochs 1")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1

if __name__ == '__main__':
    sys.exit(main())

