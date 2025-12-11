"""
Dataset Annotation Helper using Claude Vision
==============================================

Use Claude's vision capabilities to assist with dataset annotation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.claude_assistant import ClaudeAssistant
import argparse
from pathlib import Path


def annotate_images(image_dir: str, output_file: str = "annotations.txt"):
    """
    Process images and get annotation suggestions from Claude
    
    Args:
        image_dir: Directory containing images to annotate
        output_file: File to save annotations
    """
    assistant = ClaudeAssistant()
    
    # Find all images
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    images = [f for f in image_dir.iterdir() 
              if f.suffix.lower() in image_extensions]
    
    if not images:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(images)} images to annotate")
    print("="*60)
    
    annotations = []
    
    for i, image_path in enumerate(images, 1):
        print(f"\n[{i}/{len(images)}] Analyzing: {image_path.name}")
        print("-"*60)
        
        try:
            # Get Claude's analysis
            analysis = assistant.analyze_image_for_annotation(
                str(image_path),
                task_description="Focus on objects relevant to street scene segmentation"
            )
            
            print(analysis)
            
            # Save annotation
            annotations.append({
                'image': image_path.name,
                'annotation': analysis
            })
            
            # Save progressively
            with open(output_file, 'w') as f:
                f.write("Dataset Annotation Suggestions from Claude\n")
                f.write("="*60 + "\n\n")
                for ann in annotations:
                    f.write(f"Image: {ann['image']}\n")
                    f.write("-"*60 + "\n")
                    f.write(ann['annotation'])
                    f.write("\n\n" + "="*60 + "\n\n")
            
        except Exception as e:
            print(f"⚠ Error processing {image_path.name}: {e}")
            continue
    
    print(f"\n✓ Annotations saved to {output_file}")
    print(f"  Processed {len(annotations)}/{len(images)} images successfully")


def explain_single_prediction(image_path: str, prediction_classes: str):
    """
    Explain a segmentation result for a single image
    
    Args:
        image_path: Path to image
        prediction_classes: Comma-separated predicted classes
    """
    assistant = ClaudeAssistant()
    
    prediction_summary = f"Detected classes: {prediction_classes}"
    
    print(f"Analyzing segmentation result for: {image_path}")
    print("="*60)
    
    explanation = assistant.explain_segmentation_result(image_path, prediction_summary)
    
    print("\n🤖 Claude's Analysis:")
    print("-"*60)
    print(explanation)
    print("-"*60)
    
    # Save explanation
    output_file = f"{Path(image_path).stem}_explanation.txt"
    with open(output_file, 'w') as f:
        f.write(f"Image: {image_path}\n")
        f.write(f"Predictions: {prediction_classes}\n")
        f.write("="*60 + "\n\n")
        f.write(explanation)
    
    print(f"\n✓ Explanation saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Claude-powered dataset annotation helper')
    parser.add_argument('mode', choices=['annotate', 'explain'], 
                       help='Mode: annotate images or explain predictions')
    parser.add_argument('--image-dir', type=str, 
                       help='Directory containing images (for annotate mode)')
    parser.add_argument('--image', type=str,
                       help='Single image path (for explain mode)')
    parser.add_argument('--predictions', type=str,
                       help='Predicted classes (for explain mode)')
    parser.add_argument('--output', type=str, default='annotations.txt',
                       help='Output file name')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'annotate':
            if not args.image_dir:
                print("Error: --image-dir required for annotate mode")
                exit(1)
            annotate_images(args.image_dir, args.output)
        
        elif args.mode == 'explain':
            if not args.image or not args.predictions:
                print("Error: --image and --predictions required for explain mode")
                exit(1)
            explain_single_prediction(args.image, args.predictions)
    
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nSetup instructions:")
        print("1. Get API key from https://console.anthropic.com/")
        print("2. Set environment variable:")
        print("   export ANTHROPIC_API_KEY='your-api-key-here'")
        print("\nExample usage:")
        print("  # Annotate directory of images")
        print("  python examples/claude_annotation_helper.py annotate --image-dir data/raw/images/")
        print("\n  # Explain a prediction")
        print("  python examples/claude_annotation_helper.py explain --image test.jpg --predictions 'road,sidewalk,building,sky'")
