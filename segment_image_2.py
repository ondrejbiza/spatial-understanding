#!/usr/bin/env python3
"""
Python script to replicate segmentation mask functionality from the spatial-understanding app.

This script uses Google's Gemini AI to get segmentation masks for objects in images,
replicating the functionality from the React TypeScript app.
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types


def load_and_resize_image(image_path, max_size=640):
    """Load and resize image similar to the JavaScript app."""
    try:
        img = Image.open(image_path)
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Calculate scale factor
        scale = min(max_size / img.width, max_size / img.height)
        
        if scale < 1:
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        sys.exit(1)


def image_to_base64(image):
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode()


def get_segmentation_prompt(target_objects="all objects", language="English"):
    """Generate the segmentation prompt similar to the JavaScript app."""
    base_prompt = f"Give the segmentation masks for {target_objects}. Output a JSON list of segmentation masks where each entry contains the 2D bounding box in the key \"box_2d\", the segmentation mask in key \"mask\", and the text label in the key \"label\"."
    
    if language.lower() != "english":
        base_prompt += f" Use descriptive labels in {language}. Ensure labels are in {language}. DO NOT USE ENGLISH FOR LABELS."
    else:
        base_prompt += " Use descriptive labels."
    
    return base_prompt


def get_segmentation_masks(image_path, target_objects="all objects", language="English", temperature=0.4):
    """Get segmentation masks from Gemini API."""
    # Initialize Gemini
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set")
        sys.exit(1)
    
    # Create the client
    client = genai.Client(api_key=api_key)
    
    # Load and process image
    image = load_and_resize_image(image_path)
    image_b64 = image_to_base64(image)
    
    # Get the prompt
    prompt = get_segmentation_prompt(target_objects, language)
    print(f"Using prompt: {prompt}")
    
    try:
        # Create the content
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=base64.b64decode(image_b64),
                        mime_type='image/png'
                    )
                ]
            )
        ]
        
        # Configure generation settings
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )
        )
        
        # Generate content
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=generate_content_config,
        )
        
        response_text = response.candidates[0].content.parts[0].text
        
        # Parse JSON response (handle code blocks)
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        
        try:
            parsed_response = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Raw response: {response_text}")
            return None
        
        # Format the response similar to the JavaScript app
        formatted_masks = []
        for item in parsed_response:
            if 'box_2d' in item and 'label' in item and 'mask' in item:
                box_2d = item['box_2d']
                # Convert from [ymin, xmin, ymax, xmax] format (0-1000 scale) to normalized
                ymin, xmin, ymax, xmax = box_2d
                formatted_mask = {
                    'x': xmin / 1000,
                    'y': ymin / 1000, 
                    'width': (xmax - xmin) / 1000,
                    'height': (ymax - ymin) / 1000,
                    'label': item['label'],
                    'mask_data': item['mask']  # This would be the base64 image data
                }
                formatted_masks.append(formatted_mask)
        
        # Sort by area (largest to smallest) like the app
        formatted_masks.sort(key=lambda x: x['width'] * x['height'], reverse=True)
        
        return formatted_masks
        
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None


def visualize_masks(image_path, masks, output_path=None, show_masks=True):
    """Visualize the segmentation masks on the original image."""
    if not masks:
        print("No masks to visualize")
        return
        
    # Load original image
    original_image = Image.open(image_path)
    img_width, img_height = original_image.size
    
    # Create a copy for drawing
    viz_image = original_image.copy()
    draw = ImageDraw.Draw(viz_image)
    
    # Color palette (similar to the app)
    colors = [
        (230, 25, 75),    # Red
        (60, 137, 208),   # Blue  
        (60, 180, 75),    # Green
        (255, 225, 25),   # Yellow
        (145, 30, 180),   # Purple
        (66, 212, 244),   # Cyan
        (245, 130, 49),   # Orange
        (240, 50, 230),   # Magenta
        (191, 239, 69),   # Lime
        (70, 153, 144),   # Teal
    ]
    
    print(f"Visualizing {len(masks)} masks:")
    
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        
        # Calculate actual pixel coordinates
        x = int(mask['x'] * img_width)
        y = int(mask['y'] * img_height)
        w = int(mask['width'] * img_width)
        h = int(mask['height'] * img_height)

        # Draw segmentation mask if available and requested
        if show_masks and 'mask_data' in mask and mask['mask_data']:
            mask_data = mask['mask_data']

            import base64
            from io import BytesIO
            import numpy as np

            # Strip off the prefix
            header, encoded = mask_data.split(",", 1)

            # Decode base64 → bytes
            image_data = base64.b64decode(encoded)

            # Load with Pillow
            mask_image = Image.open(BytesIO(image_data))

            # Convert to NumPy array
            img_array = np.array(mask_image)

            print(img_array.shape)   # (height, width, channels)
            print(img_array.dtype)   # Usually uint8

            # import pdb ; pdb.set_trace()
            
            # # Check if mask_data is already a list/array of coordinates (polygon format)
            # if isinstance(mask_data, list):
            #     # Skip mask visualization for polygon format - just show bounding box
            #     print(f"  Info: Mask for '{mask['label']}' is in polygon format, showing bounding box only")
            #     continue
            
            # # Try to decode as base64 image
            # mask_b64 = str(mask_data).strip()
            
            # # Remove data URL prefix if present
            # if mask_b64.startswith('data:'):
            #     mask_b64 = mask_b64.split(',', 1)[-1]
            
            # # Add padding if needed for base64 decoding
            # missing_padding = len(mask_b64) % 4
            # if missing_padding:
            #     mask_b64 += '=' * (4 - missing_padding)
            
            # try:
            #     mask_image_data = base64.b64decode(mask_b64, validate=True)
            # except Exception as decode_error:
            #     print(f"  Info: Could not decode mask for '{mask['label']}' as base64 image: {decode_error}")
            #     print(f"  Info: Mask data type: {type(mask_data)}, length: {len(str(mask_data))}")
            #     continue
            
            # mask_image = Image.open(BytesIO(mask_image_data)).convert('RGBA')
            
            # Resize mask to match the bounding box size
            mask_resized = mask_image.resize((w, h), Image.LANCZOS)
            mask_resized = np.array(mask_resized)
            
            # Create colored overlay with transparency
            overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
            overlay_pixels = overlay.load()
            
            # Apply color to mask pixels only within the bounding box
            for py in range(h):
                for px in range(w):
                    # Check if this pixel is part of the mask (non-transparent)
                    if mask_resized[py, px] > 128:  # Alpha threshold
                        # Apply mask only within bounding box coordinates
                        overlay_pixels[x + px, y + py] = (*color, 100)  # Semi-transparent colored mask
            
            # Composite the mask overlay onto the image
            viz_image = Image.alpha_composite(viz_image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(viz_image)

        # Draw bounding box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
        
        # Draw label
        label = mask['label']
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        except:
            font = ImageFont.load_default()
        
        # Calculate text size and draw background
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.rectangle([x, y - text_height - 4, x + text_width + 8, y], fill=color)
        draw.text((x + 4, y - text_height - 2), label, fill='white', font=font)
        
        print(f"  {i+1}. {label} - Box: ({mask['x']:.3f}, {mask['y']:.3f}, {mask['width']:.3f}, {mask['height']:.3f})")
    
    # Save or show the result
    if output_path:
        viz_image.save(output_path)
        print(f"Visualization saved to: {output_path}")
    else:
        viz_image.show()


def main():
    parser = argparse.ArgumentParser(description="Get segmentation masks from images using Gemini AI")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("-o", "--objects", default="all objects", 
                       help="What objects to detect (default: 'all objects')")
    parser.add_argument("-l", "--language", default="English",
                       help="Language for labels (default: 'English')")
    parser.add_argument("-t", "--temperature", type=float, default=0.4,
                       help="Temperature for generation (default: 0.4)")
    parser.add_argument("--output", help="Output path for visualization image")
    parser.add_argument("--json-output", help="Output path for JSON results")
    parser.add_argument("--no-viz", action="store_true", help="Don't show visualization")
    parser.add_argument("--no-masks", action="store_true", help="Only show bounding boxes, not segmentation masks")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found")
        sys.exit(1)
    
    print(f"Processing image: {args.image_path}")
    print(f"Target objects: {args.objects}")
    print(f"Label language: {args.language}")
    
    # Get segmentation masks
    masks = get_segmentation_masks(
        args.image_path, 
        args.objects, 
        args.language, 
        args.temperature
    )
    
    if masks is None:
        print("Failed to get segmentation masks")
        sys.exit(1)
    
    print(f"\nFound {len(masks)} objects with segmentation masks")
    
    # Save JSON output if requested
    if args.json_output:
        with open(args.json_output, 'w', encoding='utf-8') as f:
            json.dump(masks, f, indent=2, ensure_ascii=False)
        print(f"JSON results saved to: {args.json_output}")
    
    # Show visualization unless disabled
    if not args.no_viz:
        visualize_masks(args.image_path, masks, args.output, show_masks=not args.no_masks)


if __name__ == "__main__":
    main()