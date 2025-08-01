import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import():
    """Test that we can import our training module"""
    try:
        import train
        assert True
    except ImportError:
        assert False, "Could not import train module"

def test_mlx_available():
    """Test that MLX is available"""
    try:
        import mlx.core as mx
        assert True
    except ImportError:
        # Skip test if MLX not available (e.g., not on Apple Silicon)
        print("MLX not available - skipping test")

if __name__ == "__main__":
    test_import()
    print("Basic tests passed!")
