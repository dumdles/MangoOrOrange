import os
from PIL import Image
import pillow_heif

def convert_heic_to_jpg(input_dir, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.heic'):
            # Construct full file path
            heic_path = os.path.join(input_dir, filename)
            jpg_filename = os.path.splitext(filename)[0] + '.jpg'
            jpg_path = os.path.join(output_dir, jpg_filename)
            
            # Open the HEIC image and convert it to JPG
            heif_file = pillow_heif.read_heif(heic_path)
            image = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data,
                "raw",
            )
            # Convert to RGB mode if necessary
            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            # Save as JPG
            image.save(jpg_path, 'JPEG')
            print(f"Converted {heic_path} to {jpg_path}")

if __name__ == "__main__":
    # Ask the user for the input and output directories
    input_directory = input("Please enter the path to your input directory: ")
    output_directory = input("Please enter the path to your output directory: ")

    # Convert HEIC files to JPG in the specified directories
    convert_heic_to_jpg(input_directory, output_directory)
