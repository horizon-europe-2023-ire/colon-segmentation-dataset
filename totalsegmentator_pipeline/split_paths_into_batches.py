import os

def split_paths_into_batches(image_paths_file, output_dir, batch_size):
    with open(image_paths_file, 'r') as f:
        image_paths = f.read().splitlines()

    # Split image paths into batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_file = f"{output_dir}/batch_{i // batch_size + 1}.txt"

        with open(batch_file, 'w') as bf:
            bf.write("\n".join(batch_paths))

if __name__ == "__main__":
    image_paths_file = "/home/smp884/IRE/image_paths.txt"
    output_dir = "/home/smp884/IRE/batch_paths"
    batch_size = 10  # Adjust batch size as needed

    os.makedirs(output_dir, exist_ok=True)
    split_paths_into_batches(image_paths_file, output_dir, batch_size)
