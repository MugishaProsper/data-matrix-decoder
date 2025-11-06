#!/usr/bin/env python3
"""
Test script to verify the setup is working correctly
"""

def test_imports():
    """Test that all modules can be imported"""
    try:
        from src.data_matrix_decoder import preprocess_image, decode_datamatrix_from_image
        from src.utils import collect_image_files, save_results_to_csv
        from main import main
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available"""
    dependencies = [
        ('cv2', 'opencv-python'),
        ('PIL', 'Pillow'),
        ('numpy', 'numpy'),
        ('pylibdmtx.pylibdmtx', 'pylibdmtx'),
        ('flask', 'flask')
    ]
    
    all_good = True
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} not found")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("Testing Data Matrix Decoder setup...")
    print()
    
    imports_ok = test_imports()
    deps_ok = test_dependencies()
    
    print()
    if imports_ok and deps_ok:
        print("🎉 Setup is ready!")
        print()
        print("You can now run:")
        print("  python main.py <image_path>     # CLI interface")
        print("  python server.py               # Web server")
    else:
        print("❌ Setup has issues. Please install missing dependencies.")
        print("Run: pip install -r requirements.txt")