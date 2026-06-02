import subprocess
import sys

def main():
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动应用失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()