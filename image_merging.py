import os
import datetime
from PIL import Image
from constants import PROCESSED_FOLDER, UPLOAD_FOLDER

def get_image_timestamp(image_path):
    try:
        image = Image.open(image_path)
        image_exif = image._getexif()
        if image_exif:
            timestamp = image_exif.get(36867)  # 36867 is the tag for DateTimeOriginal
            if timestamp:
                return datetime.datetime.strptime(timestamp, '%Y:%m:%d %H:%M:%S')
    except Exception as e:
        print(f"Error reading metadata from {image_path}: {e}")
    return None

def merge_images_side_by_side(image_paths, output_path):
    images = [Image.open(path) for path in image_paths]
    
    # Resize images to the same height, maintaining aspect ratio
    heights = [img.size[1] for img in images]
    max_height = min(heights)
    resized_images = []
    for img in images:
        aspect_ratio = img.size[0] / img.size[1]
        new_width = int(aspect_ratio * max_height)
        resized_img = img.resize((new_width, max_height))
        resized_images.append(resized_img)

    # Create a new image with the sum of widths and max height
    total_width = sum([img.size[0] for img in resized_images])
    merged_image = Image.new('RGB', (total_width, max_height))

    # Paste images into the new image
    x_offset = 0
    for img in resized_images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the merged image
    merged_image.save(output_path)

def merge_images_to_box_ratio(image_paths, output_path, final_aspect_ratio=(3, 4)):
    images = [Image.open(path) for path in image_paths]

    # Determine the size for each image to fit in the final 3:4 box ratio
    final_height = max(img.size[1] for img in images)  # use the tallest image height
    final_width_per_image = int(final_height * final_aspect_ratio[0] / final_aspect_ratio[1] / len(images))

    resized_images = []
    for img in images:
        # Resize the image to fit the required width while maintaining its aspect ratio
        img_aspect_ratio = img.size[0] / img.size[1]
        new_width = final_width_per_image
        new_height = int(new_width / img_aspect_ratio)

        if new_height > final_height:
            # If the resized height is greater than the final height, adjust the width instead
            new_height = final_height
            new_width = int(new_height * img_aspect_ratio)

        resized_img = img.resize((new_width, new_height), Image.NEAREST)

        # Pad the image if it doesn't fill the height
        if new_height < final_height:
            padded_img = Image.new('RGB', (new_width, final_height))
            top = (final_height - new_height) // 2
            padded_img.paste(resized_img, (0, top))
            resized_img = padded_img

        resized_images.append(resized_img)

    # Create a new image with the sum of widths and max height
    total_width = sum([img.size[0] for img in resized_images])
    merged_image = Image.new('RGB', (total_width, final_height))

    # Paste images into the new image
    x_offset = 0
    for img in resized_images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.size[0]

    # Save the merged image
    merged_image.save(output_path)

def merge_images(clusters):
    for cluster_key, filenames in clusters.items():
        # Extract image paths and sort them by creation date
        image_paths = [os.path.join(UPLOAD_FOLDER, filename) for filename in filenames]
        image_paths.sort(key=lambda path: get_image_timestamp(path))

        # Merge and save the images
        output_path = os.path.join(PROCESSED_FOLDER, f'merged_{cluster_key}.jpg')
        merge_images_to_box_ratio(image_paths, output_path, final_aspect_ratio=(1, 1))