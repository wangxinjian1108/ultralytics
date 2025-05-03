import os

def rename_files_in_dir(directory, old_str, new_str):
    for root, _, files in os.walk(directory):
        for file in files:
            if old_str in file:
                old_path = os.path.join(root, file)
                new_file = file.replace(old_str, new_str)
                new_path = os.path.join(root, new_file)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    dir_path = input("Enter the directory path: ").strip()
    old_part = input("Enter the part of the filename to replace: ").strip()
    new_part = input("Enter the new part to replace with: ").strip()

    if os.path.isdir(dir_path):
        rename_files_in_dir(dir_path, old_part, new_part)
    else:
        print("Invalid directory path.")