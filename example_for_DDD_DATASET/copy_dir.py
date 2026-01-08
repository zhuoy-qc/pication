import os
import shutil

def move_directories_with_pication_posebuster():
    # Create the target directory if it doesn't exist
    target_dir = "with-pication"
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        print(f"Created directory: {target_dir}")

    # Read the input file
    try:
        with open("directories_with_interactions.txt", "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print("Error: 'PDB_lists_withpi-cation-interactions.txt' not found in current directory")
        return

    # Process each line
    for line in lines:
        # Remove whitespace and newline characters
        dir_name = line.strip()

        # Skip empty lines
        if not dir_name:
            continue

        # Check if the directory exists in current directory
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            destination = os.path.join(target_dir, dir_name)
            try:
                # Copy the directory to the target directory
                shutil.copytree(dir_name, destination)
                print(f"Copied: {dir_name} -> {destination}")
            except FileExistsError:
                print(f"Destination already exists: {destination}")
            except Exception as e:
                print(f"Error copying {dir_name}: {e}")
        else:
            print(f"Directory not found: {dir_name}")

if __name__ == "__main__":
    move_directories_with_pication_posebuster()
