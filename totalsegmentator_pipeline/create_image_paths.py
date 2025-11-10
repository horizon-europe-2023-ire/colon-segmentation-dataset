import os

def create_image_paths_file(input_dir, output_txt):
    with open(output_txt, 'w') as f:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".mha.gz") or file.endswith(".mha.zip"):
                    full_path = os.path.join(root, file)
                    f.write(full_path + "\n")

if __name__ == "__main__":
    input_dir = "/home/smp884/IRE/data/CT/converted"  # Input directory with converted images
    output_txt = "/home/smp884/IRE/image_paths.txt"   # Output text file to store the paths

    create_image_paths_file(input_dir, output_txt)
