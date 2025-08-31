#!/usr/bin/env python3
"""
Image segmentation script using Gemini 2.5 Flash
Loads an arbitrary image and runs it through Gemini with a segmentation prompt.
"""

import argparse
import base64
import mimetypes
import os

from google import genai
from google.genai import types


def load_and_encode_image(image_path):
    """Load an image file and encode it as base64."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Get the MIME type of the image
    mime_type, _ = mimetypes.guess_type(image_path)
    if not mime_type or not mime_type.startswith('image/'):
        raise ValueError(f"File does not appear to be an image: {image_path}")
    
    # Read and encode the image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    return encoded_image, mime_type


def segment_image(image_path, segmentation_prompt="Draw a red segmentation mask over the main object in the image."):
    """
    Run image segmentation using Gemini 2.5 Flash.
    
    Args:
        image_path (str): Path to the input image
        segmentation_prompt (str): Prompt describing what to segment
    
    Returns:
        The response from the model
    """
    try:
        # Load and encode the image
        encoded_image, mime_type = load_and_encode_image(image_path)
        
        # Get API key from environment
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GENAI_API_KEY environment variable not set. Please set it to your Google Generative AI API key.")
        
        # Create the client
        client = genai.Client(
            api_key=api_key,
        )
        
        # Configure the model
        model = "gemini-2.5-flash-image-preview"
        
        # Create the content
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=segmentation_prompt),
                    types.Part.from_bytes(
                        data=base64.b64decode(encoded_image),
                        mime_type=mime_type
                    )
                ]
            )
        ]
        
        # Configure generation settings
        generate_content_config = types.GenerateContentConfig(
            response_modalities=[
                "IMAGE",
                "TEXT",
            ],
        )
        
        # Generate content
        response_chunks = []
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            if (
                chunk.candidates is None
                or chunk.candidates[0].content is None
                or chunk.candidates[0].content.parts is None
            ):
                continue
            response_chunks.append(chunk)
        
        return response_chunks
        
    except Exception as e:
        print(f"Error during segmentation: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Segment objects in images using Gemini 2.5 Flash",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup:
  Set the GENAI_API_KEY environment variable to your Google Generative AI API key:
    export GENAI_API_KEY=your_api_key_here

Examples:
  python segment_image.py bob.png
  python segment_image.py dave.png --prompt "Draw a red segmentation mask over the person's face."
  python segment_image.py /path/to/image.jpg --prompt "Segment the background elements."
        """
    )
    
    parser.add_argument(
        'image_path',
        help='Path to the input image file'
    )
    
    parser.add_argument(
        '--prompt', '-p',
        default="Draw a red segmentation mask over the main object in the image.",
        help='Segmentation prompt (default: "Draw a red segmentation mask over the main object in the image.")'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print verbose output including the full prompt'
    )
    
    args = parser.parse_args()
    
    # Check if image file exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist.")
        return 1
    
    if args.verbose:
        print(f"Image path: {args.image_path}")
        print(f"Segmentation prompt: {args.prompt}")
        print("Processing...")
    
    try:
        # Run the segmentation
        response_chunks = segment_image(args.image_path, args.prompt)
        
        # Process the response
        if response_chunks:
            image_saved = False
            for chunk in response_chunks:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, 'text') and part.text:
                        print(part.text)
                    elif hasattr(part, 'inline_data') and part.inline_data.data:
                        # Save the image data
                        base_name = os.path.splitext(args.image_path)[0]
                        output_path = f"{base_name}_segmented.png"
                        
                        with open(output_path, 'wb') as f:
                            f.write(part.inline_data.data)
                        
                        print(f"Segmented image saved to: {output_path}")
                        image_saved = True
            
            if not image_saved:
                print("No image data received from the model")
        else:
            print("No response received from the model")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())