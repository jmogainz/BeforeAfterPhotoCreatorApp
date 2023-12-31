import os
import datetime
from PIL import Image
from PIL import ExifTags

def apply_exif_orientation(image):
    rotated = False
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
            rotated = True
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
            rotated = True
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
            rotated = True
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass
    return image, rotated

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
    images = []
    for path in image_paths:
        img = Image.open(path)
        img, _ = apply_exif_orientation(img)
        images.append(img)
    
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

def merge_images_with_custom_ratio_compromising_original_ar(image_paths, output_path, merged_aspect_ratio=(3, 4)):
    images = []
    for path in image_paths:
        img = Image.open(path)
        img, _ = apply_exif_orientation(img)
        images.append(img)

    # Calculate the size of each image's slot in the merged image
    num_images = len(images)
    merged_height = max(img.size[1] for img in images)  # Use the tallest image height
    merged_width = int(merged_height * merged_aspect_ratio[0] / merged_aspect_ratio[1])
    slot_width = merged_width // num_images

    resized_images = []
    for img in images:
        # Resize the image to fit its slot
        resized_img = img.resize((slot_width, merged_height), Image.LANCZOS)
        resized_images.append(resized_img)

    # Create a new image for the merged result
    merged_image = Image.new('RGB', (merged_width, merged_height))

    # Paste images into the new image
    x_offset = 0
    for img in resized_images:
        merged_image.paste(img, (x_offset, 0))
        x_offset += slot_width

    output_directory = os.path.dirname(output_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the merged image
    merged_image.save(output_path)
    
def merge_images_with_custom_ratio_maintaining_original_ar(image_paths, output_path, merged_aspect_ratio=(3, 4)):
    images = []
    for path in image_paths:
        img = Image.open(path)
        img, _ = apply_exif_orientation(img)
        images.append(img)

    # Determine the size for each image to fit in the final 3:4 box ratio
    final_height = max(img.size[1] for img in images)  # use the tallest image height
    final_width_per_image = int(final_height * merged_aspect_ratio[0] / merged_aspect_ratio[1] / len(images))

    resized_images = []
    for img in images:
        # Resize and crop the image to fit the required width and height while maintaining the aspect ratio
        img_aspect_ratio = img.size[0] / img.size[1]
        target_aspect_ratio = final_width_per_image / final_height

        if img_aspect_ratio > target_aspect_ratio:
            # Image is wider than target aspect ratio, crop the width
            new_height = final_height
            new_width = int(new_height * img_aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            crop_width = int(final_height * target_aspect_ratio)
            left = (new_width - crop_width) // 2
            img = img.crop((left, 0, left + crop_width, new_height))
        else:
            # Image is taller than target aspect ratio, crop the height
            new_width = final_width_per_image
            new_height = int(new_width / img_aspect_ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)
            crop_height = int(new_width / target_aspect_ratio)
            top = (new_height - crop_height) // 2
            img = img.crop((0, top, new_width, top + crop_height))

        resized_images.append(img)

    # Create a new image with the sum of widths and max height
    total_width = sum([img.size[0] for img in resized_images])
    merged_image = Image.new('RGB', (total_width, final_height))

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

def merge_images(clusters, input_dir, output_dir):
    for cluster_key, filenames in clusters.items():
        # Extract image paths and sort them by creation date
        image_paths = [os.path.join(input_dir, filename) for filename in filenames]
        image_paths.sort(key=lambda path: get_image_timestamp(path))

        # Merge and save the images
        # aspect ratios: 4:3, 5:4, 16:10, 16:9, 3:2, 1:1, 3:4 (how the images are coming in rn)
        output_path = os.path.join(output_dir, f'merged_{cluster_key}.jpg')
        merge_images_with_custom_ratio_maintaining_original_ar(image_paths, output_path, merged_aspect_ratio=(3, 2))