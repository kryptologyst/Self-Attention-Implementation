#!/usr/bin/env python3
"""
GitHub Repository Preparation Script
===================================

This script prepares the project for GitHub repository creation.
"""

import os
import subprocess
import sys


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_git_status():
    """Check if this is a git repository"""
    try:
        subprocess.run("git status", shell=True, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def initialize_git_repo():
    """Initialize git repository if not already initialized"""
    if not check_git_status():
        print("🔄 Initializing git repository...")
        run_command("git init", "Initializing git repository")
        run_command("git add .", "Adding all files to git")
        run_command('git commit -m "Initial commit: Advanced Self-Attention Implementation"', "Creating initial commit")
        print("✅ Git repository initialized")
    else:
        print("✅ Git repository already exists")


def create_github_instructions():
    """Create instructions for GitHub repository creation"""
    instructions = """
# GitHub Repository Setup Instructions

## 1. Create GitHub Repository
1. Go to https://github.com/new
2. Repository name: `self-attention-implementation`
3. Description: `Advanced Self-Attention Implementation with PyTorch, Interactive UI, and Performance Analysis`
4. Set to Public
5. Don't initialize with README (we already have one)
6. Click "Create repository"

## 2. Connect Local Repository to GitHub
Run these commands in your project directory:

```bash
# Add remote origin (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/self-attention-implementation.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 3. Repository Settings
- Enable Issues
- Enable Wiki (optional)
- Add topics: `pytorch`, `attention`, `transformer`, `deep-learning`, `machine-learning`, `self-attention`

## 4. Create Release
1. Go to Releases → Create a new release
2. Tag version: `v1.0.0`
3. Release title: `Advanced Self-Attention Implementation v1.0.0`
4. Description: Copy from README.md features section

## 5. Optional: GitHub Pages
If you want to host the Streamlit app:
1. Go to Settings → Pages
2. Source: Deploy from a branch
3. Branch: main
4. Folder: / (root)

## 6. Community Standards
- Add CONTRIBUTING.md (optional)
- Add CODE_OF_CONDUCT.md (optional)
- Enable Discussions (optional)
"""
    
    with open("GITHUB_SETUP.md", "w") as f:
        f.write(instructions)
    
    print("✅ GitHub setup instructions created in GITHUB_SETUP.md")


def main():
    """Main preparation function"""
    print("🚀 Preparing Project for GitHub")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists("attention_implementation.py"):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Initialize git repository
    initialize_git_repo()
    
    # Create GitHub setup instructions
    create_github_instructions()
    
    # Show project structure
    print("\n📁 Project Structure:")
    print("├── attention_implementation.py    # Main implementation")
    print("├── app.py                        # Streamlit web interface")
    print("├── test_attention.py             # Comprehensive test suite")
    print("├── requirements.txt              # Dependencies")
    print("├── README.md                     # Documentation")
    print("├── LICENSE                       # MIT License")
    print("├── setup.py                      # Setup script")
    print("├── .gitignore                    # Git ignore rules")
    print("├── GITHUB_SETUP.md               # GitHub setup instructions")
    print("└── examples/                      # Usage examples")
    print("    ├── basic_usage.py")
    print("    ├── visualization_demo.py")
    print("    └── performance_analysis.py")
    
    print("\n🎉 Project is ready for GitHub!")
    print("\n📋 Next Steps:")
    print("1. Read GITHUB_SETUP.md for detailed instructions")
    print("2. Create GitHub repository")
    print("3. Push your code")
    print("4. Share with the community!")
    
    print("\n🔗 Useful Commands:")
    print("   python setup.py                 # Run setup")
    print("   python -m pytest test_attention.py -v  # Run tests")
    print("   streamlit run app.py            # Launch web UI")
    print("   python attention_implementation.py     # Run main demo")


if __name__ == "__main__":
    main()
