from PIL import Image


def process_image(input_path, output_path):
    # Open the image
    with Image.open(input_path) as img:
        # Rotate the image 90 degrees clockwise
        rotated_img = img.rotate(-90, expand=True)

        # Resize the image
        new_size = (240 * 3, 576 * 3)
        resized_img = rotated_img.resize(new_size, Image.LANCZOS)

        # Save the processed image
        resized_img.save(output_path)


# Usage
input_image = "/home/haoyang/project/haoyang/openvlp/test/marble2.jpg"
output_image = "/home/haoyang/project/haoyang/openvlp/test/marble2_processed.jpg"
process_image(input_image, output_image)
