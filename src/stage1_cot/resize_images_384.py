#!/usr/bin/env python3
# coding: utf-8
"""
Image Resize Script - Convert all images to 384×384 resolution
Features:
- Batch process agricultural disease dataset images
- Resize to 384×384 using smart padding to preserve content
- Convert all formats to JPG with high quality
- Intelligent background color filling based on dominant color
"""

import os
from PIL import Image
import glob
from tqdm import tqdm


def batch_resize_agriculture_dataset(dataset_root, target_size=384):
    """
    Batch process agricultural disease dataset images, resize to target_size×target_size
    Uses smart padding to preserve image content without cropping

    Args:
        dataset_root: Root directory of dataset
        target_size: Target size (default 384)
    """
    # Statistics
    stats = {
        'total_processed': 0,
        'successful': 0,
        'failed': 0,
        'skipped': 0,
        'format_changed': 0,
        'size_changed': 0
    }

    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif', '*.webp']:
        image_files.extend(glob.glob(os.path.join(dataset_root, '**', ext), recursive=True))

    print(f"Found {len(image_files)} image files")

    # Process with progress bar
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            with Image.open(image_path) as img:
                stats['total_processed'] += 1

                # Convert to RGB mode (avoid RGBA issues)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Get original size
                original_width, original_height = img.size

                # Target size
                target = (target_size, target_size)

                # Choose appropriate resize method based on original size
                img_resized = smart_resize(img, target)

                # Generate new JPG file path
                file_dir, file_name = os.path.split(image_path)
                file_base = os.path.splitext(file_name)[0]
                new_image_path = os.path.join(file_dir, file_base + ".jpg")

                # Save as JPG format with highest quality
                img_resized.save(new_image_path, 'JPEG', quality=100, optimize=True)

                # Remove original file if path changed
                if new_image_path != image_path:
                    os.remove(image_path)
                    stats['format_changed'] += 1

                stats['successful'] += 1

                # Record size change
                if original_width != target_size or original_height != target_size:
                    stats['size_changed'] += 1

        except Exception as e:
            stats['failed'] += 1
            print(f"Failed to process: {image_path} - {str(e)}")

    # Print statistics
    print(f"\nProcessing complete!")
    print(f"Total processed: {stats['total_processed']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Format changed: {stats['format_changed']}")
    print(f"Size adjusted: {stats['size_changed']}")


def smart_resize(img, target_size):
    """
    Smart image resize to preserve content
    Uses appropriate padding method to avoid cropping important content

    Args:
        img: PIL Image object
        target_size: Tuple of (width, height)

    Returns:
        Resized PIL Image object
    """
    target_width, target_height = target_size
    img_width, img_height = img.size

    # Calculate aspect ratios
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height

    # If aspect ratio is similar, use direct resize
    if abs(img_ratio - target_ratio) < 0.1:
        return img.resize(target_size, Image.Resampling.LANCZOS)

    # If aspect ratio differs significantly, use padding
    # For wide images (width much larger than height)
    if img_ratio > target_ratio * 1.2:
        return resize_with_horizontal_padding(img, target_size)

    # For tall images (height much larger than width)
    elif img_ratio < target_ratio * 0.8:
        return resize_with_vertical_padding(img, target_size)

    # Other cases: use proportional padding with center alignment
    else:
        return resize_with_proportional_padding(img, target_size)


def resize_with_horizontal_padding(img, target_size):
    """
    Preserve original height, pad left and right sides
    Suitable for wide images
    """
    target_width, target_height = target_size
    img_width, img_height = img.size

    # Calculate scale ratio (based on height)
    scale = target_height / img_height
    new_width = int(img_width * scale)

    # Resize image
    img_resized = img.resize((new_width, target_height), Image.Resampling.LANCZOS)

    # Create new image with smart background color
    bg_color = get_dominant_color(img)
    new_img = Image.new('RGB', target_size, bg_color)

    # Calculate paste position (centered)
    paste_x = (target_width - new_width) // 2

    # Paste resized image
    new_img.paste(img_resized, (paste_x, 0))

    return new_img


def resize_with_vertical_padding(img, target_size):
    """
    Preserve original width, pad top and bottom sides
    Suitable for tall images
    """
    target_width, target_height = target_size
    img_width, img_height = img.size

    # Calculate scale ratio (based on width)
    scale = target_width / img_width
    new_height = int(img_height * scale)

    # Resize image
    img_resized = img.resize((target_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with smart background color
    bg_color = get_dominant_color(img)
    new_img = Image.new('RGB', target_size, bg_color)

    # Calculate paste position (centered)
    paste_y = (target_height - new_height) // 2

    # Paste resized image
    new_img.paste(img_resized, (0, paste_y))

    return new_img


def resize_with_proportional_padding(img, target_size):
    """
    Resize maintaining aspect ratio, fill edges with smart background color
    """
    target_width, target_height = target_size
    img_width, img_height = img.size

    # Calculate scale ratio
    scale = min(target_width / img_width, target_height / img_height)

    # Calculate resized dimensions
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)

    # Resize image
    img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create new image with smart background color
    bg_color = get_dominant_color(img)
    new_img = Image.new('RGB', target_size, bg_color)

    # Calculate paste position (centered)
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2

    # Paste resized image
    new_img.paste(img_resized, (paste_x, paste_y))

    return new_img


def get_dominant_color(img):
    """
    Get dominant color of image for smart background filling
    Uses thumbnail for performance

    Args:
        img: PIL Image object

    Returns:
        RGB tuple of dominant color
    """
    # Create thumbnail for performance
    thumbnail = img.copy()
    thumbnail.thumbnail((50, 50))

    # Get color histogram
    palette = thumbnail.getcolors(50 * 50)

    if not palette:
        return (0, 0, 0)  # Default to black

    # Find most common color
    palette.sort(key=lambda x: x[0], reverse=True)

    # Return most common color
    return palette[0][1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Resize images to 384×384')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--target_size', type=int, default=384,
                       help='Target size (default: 384)')

    args = parser.parse_args()

    batch_resize_agriculture_dataset(args.dataset_path, args.target_size)
