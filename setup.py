#!/usr/bin/env python3
"""
Setup script for Self-Attention Implementation
==============================================

This script helps set up the development environment and run tests.
"""

import subprocess
import sys
import os


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


def main():
    """Main setup function"""
    print("🚀 Setting up Self-Attention Implementation")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected")
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing dependencies"):
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Run tests
    if not run_command("python -m pytest test_attention.py -v", "Running tests"):
        print("⚠️  Some tests failed, but continuing...")
    
    # Check if Streamlit is available for the web UI
    try:
        import streamlit
        print("✅ Streamlit is available - web UI can be launched")
        print("   Run: streamlit run app.py")
    except ImportError:
        print("⚠️  Streamlit not available - web UI cannot be launched")
        print("   Install with: pip install streamlit")
    
    print("\n🎉 Setup completed!")
    print("\n📚 Available commands:")
    print("   python attention_implementation.py     # Run main demo")
    print("   python -m pytest test_attention.py -v   # Run tests")
    print("   streamlit run app.py                    # Launch web UI")
    print("   python examples/basic_usage.py           # Run basic examples")
    print("   python examples/visualization_demo.py   # Run visualization demo")
    print("   python examples/performance_analysis.py # Run performance analysis")


if __name__ == "__main__":
    main()
