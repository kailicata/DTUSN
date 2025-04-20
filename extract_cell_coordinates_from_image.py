import cv2
import numpy as np
import json


def extract_coordinates(image_path, scale_bar_length_um, output_json):
    # Load image
    img = cv2.imread(image_path)

    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Create color masks
    red_mask = (img_rgb[:, :, 0] == 255) & (img_rgb[:, :, 1] == 0) & (img_rgb[:, :, 2] == 0)
    green_mask = (img_rgb[:, :, 0] == 0) & (img_rgb[:, :, 1] == 255) & (img_rgb[:, :, 2] == 0)
    yellow_mask = (img_rgb[:, :, 0] == 255) & (img_rgb[:, :, 1] == 255) & (img_rgb[:, :, 2] == 0)
    blue_mask = (img_rgb[:, :, 0] == 0) & (img_rgb[:, :, 1] == 0) & (img_rgb[:, :, 2] == 255)

    # Extract coordinates
    center_coords = np.column_stack(np.where(red_mask))
    soma_coords = np.column_stack(np.where(green_mask)).tolist()
    dendrite_coords = np.column_stack(np.where(yellow_mask)).tolist()
    bar_coords = np.column_stack(np.where(blue_mask))

    if center_coords.size > 0:
        center = center_coords[0].tolist()  # single point assumed
    else:
        center = None

    if bar_coords.size > 0:
        x_vals = bar_coords[:, 1]
        bar_length_px = x_vals.max() - x_vals.min()
        pixel_size_um = scale_bar_length_um / bar_length_px
    else:
        bar_length_px = 0
        pixel_size_um = 0

    
    # Scale coordinates to micrometers
    if center:
        center = [float(c) * pixel_size_um for c in center]
    soma_coords = [[float(x) * pixel_size_um, float(y) * pixel_size_um] for x, y in soma_coords]
    dendrite_coords = [[float(x) * pixel_size_um, float(y) * pixel_size_um] for x, y in dendrite_coords]

    # Build result dict with conversions
    result = {
        "scaled_center_micrometers": center,
        "scaled_soma_coordinates_micrometers": soma_coords,
        "scaled_dendrite_coordinates_micrometers": dendrite_coords,
        "original_bar_length_pixels": int(bar_length_px),
        "original_bar_length_micrometers": float(scale_bar_length_um),
        "original_pixel_size_micrometers": float(pixel_size_um),
        "scaled_pixel_size_micrometers": 1.0
    }

# Save JSON
    with open(output_json, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Saved results to {output_json}")

# Example usage:
extract_coordinates("/Users/kailicata/interneuron_confocal_image_pfc_001.png", 50.0, "cell_data_scaled.json")
