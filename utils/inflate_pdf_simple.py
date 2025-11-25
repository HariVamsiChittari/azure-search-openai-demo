import argparse
import os


def inflate_pdf_simple(input_path, output_path, target_mb):
    """Inflate PDF by appending padding bytes (simplest approach)."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    target_bytes = int(target_mb * 1024 * 1024)
    
    # Read original PDF
    with open(input_path, 'rb') as f:
        pdf_bytes = f.read()
    
    current_size = len(pdf_bytes)
    print(f"Input size: {current_size / (1024 * 1024):.2f} MB")
    
    if current_size >= target_bytes:
        print(f"Input already larger than target ({target_mb} MB)")
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)
        return
    
    # Calculate how many times to repeat
    repeat = target_bytes // current_size
    remainder = target_bytes % current_size
    
    print(f"Repeating PDF {repeat} times plus {remainder} bytes...")
    
    # Write repeated copies
    with open(output_path, 'wb') as f:
        for _ in range(repeat):
            f.write(pdf_bytes)
        if remainder > 0:
            f.write(pdf_bytes[:remainder])
    
    final_size = os.path.getsize(output_path)
    print(f"Done. Output: {output_path}")
    print(f"Output size: {final_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--target-mb", type=float, required=True)
    args = parser.parse_args()
    
    inflate_pdf_simple(args.input, args.output, args.target_mb)
