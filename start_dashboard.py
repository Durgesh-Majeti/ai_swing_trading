"""
Helper script to start the Streamlit dashboard
"""

import subprocess
import sys
import os

def start_dashboard():
    """Start the Streamlit dashboard"""
    print("🚀 Starting Nifty 50 AI Swing Trader Dashboard...")
    print("=" * 60)
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    print()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_path = os.path.join(script_dir, "dashboard.py")
    
    try:
        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path], check=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_dashboard()

