import os
import subprocess
import sys

def main():
    # Prompt user for paths or accept command-line arguments
    if len(sys.argv) == 3:
        weights_path = sys.argv[1]
        dataset_path = sys.argv[2]
    else:
        print("Please enter the following paths (supports absolute or relative paths):")
        weights_path = input("1. Path to model weights folder (containing subfolders like 0_2/): ")
        dataset_path = input("2. Path to Win5 test dataset root directory: ")
    
    # Normalize paths
    weights_path = os.path.abspath(weights_path)
    dataset_path = os.path.abspath(dataset_path)
    
    # Check path validity
    if not os.path.isdir(weights_path):
        print(f"Error: Weights folder does not exist - {weights_path}")
        return
    if not os.path.isdir(dataset_path):
        print(f"Error: Test dataset directory does not exist - {dataset_path}")
        return
    
    # Call core program and print output in real-time
    print("\nStarting test run (please wait, results will match the paper)...\n")
    try:
        core_file = "core_tester.exe" if sys.platform.startswith('win') else "core_tester"
        # Use Popen instead of run to capture output line by line
        with subprocess.Popen(
            [f"./{core_file}", weights_path, dataset_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line-buffered for real-time output
            universal_newlines=True
        ) as process:
            # Print stdout in real-time (each line as it's generated)
            for line in process.stdout:
                print(line, end='')  # end='' to preserve original line breaks
            
            # Wait for process to finish and check for errors
            return_code = process.wait()
            if return_code != 0:
                print("\n===== Execution Error =====")
                print("Error output:")
                print(process.stderr.read())
                return
        
        # Print final result path after all tests complete
        print(f"\nResults saved to: {os.path.abspath('./Results/')}")
    
    except FileNotFoundError:
        print(f"\nError: Core program {core_file} not found. Ensure it's in the same directory as this script.")

if __name__ == "__main__":
    main()


