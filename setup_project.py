"""
SETUP SCRIPT - PLACE IN: C:\Users\a\Documents\Daksh\Smart India Hackathon\setup_project.py

This script creates the complete folder structure for UrbanTrace AI
Run this FIRST after downloading the project files
"""

import os
import shutil
from pathlib import Path


def setup_project():
    """Create folder structure"""
    
    # Define project root (where this script is)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Setting up project in: {project_root}\n")
    
    # Define folders to create
    folders = [
        'src',
        'data/test_videos',
        'data/sample_output',
        'api',
        'web',
    ]
    
    # Create each folder
    for folder in folders:
        folder_path = os.path.join(project_root, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"✓ Created folder: {folder}")
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50 + "\n")
    
    print("1. Copy all .py files to src/ folder:")
    py_files = [
        'database.py',
        '1_detection.py',
        '2_anpr.py',
        '3_tracking.py',
        '4_association.py',
        '5_trajectory.py',
        '6_analytics.py',
        'utils.py'
    ]
    
    for py_file in py_files:
        print(f"   - {py_file}")
    
    print("\n2. Copy config files to root:")
    print("   - requirements.txt")
    print("   - config.yaml")
    print("   - test_detection.py")
    
    print("\n3. Verify structure:")
    print_structure(project_root)
    
    print("\n4. Create __init__.py files:")
    init_files = [
        'src/__init__.py',
        'api/__init__.py'
    ]
    
    for init_file in init_files:
        init_path = os.path.join(project_root, init_file)
        if not os.path.exists(init_path):
            Path(init_path).touch()
            print(f"   ✓ Created {init_file}")
    
    print("\n" + "="*50)
    print("Setup complete! 🎉")
    print("="*50)


def print_structure(root, prefix=""):
    """Print folder structure"""
    items = []
    try:
        items = sorted(os.listdir(root))
    except PermissionError:
        return
    
    # Filter out unwanted items
    skip = {'.git', '__pycache__', '.venv', 'venv', '.pyc', '.egg-info'}
    items = [i for i in items if i not in skip and not i.startswith('.')]
    
    for i, item in enumerate(items):
        path = os.path.join(root, item)
        is_last = i == len(items) - 1
        
        print(f"{prefix}{'└── ' if is_last else '├── '}{item}")
        
        if os.path.isdir(path) and not item.startswith('.'):
            extension = "    " if is_last else "│   "
            print_structure(path, prefix + extension)


if __name__ == "__main__":
    setup_project()
