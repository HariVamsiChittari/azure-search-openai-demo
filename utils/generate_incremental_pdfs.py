import argparse
import os
import subprocess
import sys
from pathlib import Path


def generate_incremental_pdfs(
    input_path,
    output_dir,
    start_mb,
    increment_mb,
    count,
    validate=False,
    add_images=False
):
    """Generate multiple PDFs with incrementally increasing sizes.
    
    Args:
        input_path: Path to the source PDF file
        output_dir: Directory to store generated PDFs
        start_mb: Starting size in MB
        increment_mb: Size increment in MB for each file
        count: Number of files to generate
        validate: Whether to validate each generated PDF
        add_images: Whether to embed large images to guarantee target file size
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the inflate_pdf.py script path
    script_dir = Path(__file__).parent
    inflate_script = script_dir / "inflate_pdf.py"
    
    if not inflate_script.exists():
        raise FileNotFoundError(f"inflate_pdf.py not found at {inflate_script}")
    
    print(f"Generating {count} PDFs with sizes from {start_mb}MB to {start_mb + (count - 1) * increment_mb}MB")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Increment: {increment_mb}MB per file")
    print("-" * 80)
    
    # Generate each file
    for i in range(count):
        target_mb = start_mb + (i * increment_mb)
        
        # Create output filename
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{input_basename}_{target_mb}mb.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"\n[{i + 1}/{count}] Generating {output_filename} (target: {target_mb}MB)")
        
        # Build command to call inflate_pdf.py
        cmd = [
            sys.executable,
            str(inflate_script),
            "--input", input_path,
            "--output", output_path,
            "--target-mb", str(target_mb)
        ]
        
        if validate:
            cmd.append("--validate")
        
        if add_images:
            cmd.append("--add-images")
        
        # Execute the command
        try:
            result = subprocess.run(cmd, check=True, capture_output=False, text=True)
            print(f"✓ Successfully created {output_filename}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create {output_filename}: {e}")
            if i == 0:
                # If first file fails, abort the whole process
                raise
            else:
                # For subsequent files, just warn and continue
                print("Warning: Continuing with next file...")
                continue
    
    print("\n" + "=" * 80)
    print(f"✓ Generation complete! Created {count} PDF files in {output_dir}")
    
    # List all generated files with sizes
    print("\nGenerated files:")
    for i in range(count):
        target_mb = start_mb + (i * increment_mb)
        input_basename = os.path.splitext(os.path.basename(input_path))[0]
        output_filename = f"{input_basename}_{target_mb}mb.pdf"
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path):
            actual_size = os.path.getsize(output_path)
            actual_mb = actual_size / (1024 * 1024)
            print(f"  {output_filename}: {actual_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate multiple PDFs with incrementally increasing sizes"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to source PDF file"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="generated_pdfs",
        help="Directory to store generated PDFs (default: generated_pdfs)"
    )
    parser.add_argument(
        "--start-mb",
        type=float,
        default=100,
        help="Starting size in MB (default: 100)"
    )
    parser.add_argument(
        "--increment-mb",
        type=float,
        default=100,
        help="Size increment in MB for each file (default: 100)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of files to generate (default: 20)"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate each generated PDF"
    )
    parser.add_argument(
        "--add-images",
        action="store_true",
        help="Embed large images to guarantee target file size (slower)"
    )
    
    args = parser.parse_args()
    
    try:
        generate_incremental_pdfs(
            input_path=args.input,
            output_dir=args.output_dir,
            start_mb=args.start_mb,
            increment_mb=args.increment_mb,
            count=args.count,
            validate=args.validate,
            add_images=args.add_images,
        )
    except Exception as e:
        print(f"\n✗ Fatal error: {e}", file=sys.stderr)
        sys.exit(1)
