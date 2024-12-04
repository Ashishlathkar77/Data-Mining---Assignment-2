import subprocess
import sys

def install_requirements():
    print("Installing dependencies from requirements.txt...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def run_script(script_name):
    print(f"Running {script_name}...")
    print("")
    subprocess.check_call([sys.executable, script_name])

def main():

    install_requirements()

    run_script("problem1.py")
    run_script("problem2.py")
    run_script("problem3.py")

    print("All problems have been executed successfully.")

if __name__ == "__main__":
    main()