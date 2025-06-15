#!/usr/bin/env python3
import sys
import traceback
import importlib

def test_imports():
    modules = [
        'main', 'physics', 'agent', 'sensors', 'environment',
        'logger', 'analyzer', 'ascii_visualizer', 'config',
        'qik_os', 'bot_interface_impl'
    ]
    
    for module in modules:
        try:
            print(f"Importing {module}...")
            importlib.import_module(module)
            print(f"  Success: {module} imported")
        except Exception as e:
            print(f"  Error importing {module}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    print("Testing QIKI modules import...")
    test_imports()
    print("Import test complete.")
