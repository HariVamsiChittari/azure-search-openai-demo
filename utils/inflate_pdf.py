import argparse
import os
from math import ceil

try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False


def validate_pdf(pdf_path):
    """Validate that the PDF can be read without issues."""
    if not PYPDF_AVAILABLE:
        print("Warning: pypdf not installed. Skipping validation.")
        print("Install with: pip install pypdf")
        return False
    
    print(f"\nValidating PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        num_pages = len(reader.pages)
        print(f"✓ PDF is readable")
        print(f"✓ Contains {num_pages} page(s)")
        
        # Calculate page positions to sample
        pages_to_check = []
        if num_pages > 0:
            pages_to_check.append(("First", 0))
        if num_pages > 1:
            pages_to_check.append(("Last", num_pages - 1))
        if num_pages > 2:
            page_25 = int(num_pages * 0.25)
            pages_to_check.append(("25th percentile", page_25))
        if num_pages > 3:
            page_75 = int(num_pages * 0.75)
            pages_to_check.append(("75th percentile", page_75))
        
        # Read and display text from sample pages
        print(f"\nReading sample pages:")
        for page_label, page_idx in pages_to_check:
            try:
                page = reader.pages[page_idx]
                text = page.extract_text()
                print(f"\n{page_label} page (index {page_idx}):")
                print(f"  Characters: {len(text)}")
                # Print first 200 characters of text content
                preview = text[:200].strip().replace('\n', ' ')
                if preview:
                    print(f"  Preview: {preview}...")
                else:
                    print(f"  Preview: [No extractable text]")
            except Exception as e:
                print(f"  ✗ Error reading page {page_idx}: {e}")
                return False
        
        # Check metadata
        if reader.metadata:
            print(f"\n✓ Metadata present: {len(reader.metadata)} field(s)")
        
        print("\n✓ Validation successful!")
        return True
        
    except Exception as e:
        print(f"✗ Validation failed: {e}")
        print("Warning: The generated PDF may not be readable by all PDF readers.")
        return False


def inflate_pdf(input_path, output_path, repeat=None, target_mb=None, target_gb=None, validate=False):
    """Inflate PDF by duplicating pages, creating a valid multi-page PDF."""
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf is required for page duplication. Install with: pip install pypdf")
    
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    src_size = os.path.getsize(input_path)

    if src_size == 0:
        raise ValueError("Input PDF is empty (0 bytes). Cannot use it to inflate.")

    # Determine repeat count
    if repeat is None:
        if target_gb is not None:
            target_bytes = int(target_gb * 1024 * 1024 * 1024)
        elif target_mb is not None:
            target_bytes = int(target_mb * 1024 * 1024)
        else:
            raise ValueError("You must specify either --repeat, --target-mb, or --target-gb")

        repeat = max(1, ceil(target_bytes / src_size))

    print(f"Input size: {src_size / (1024 * 1024):.2f} MB")
    print(f"Duplicating pages {repeat} time(s)...")

    # Read the input PDF
    reader = PdfReader(input_path)
    num_pages = len(reader.pages)
    print(f"Input PDF has {num_pages} page(s)")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create writer and duplicate pages
    writer = PdfWriter()
    total_pages_to_write = num_pages * repeat
    
    print(f"Creating PDF with {total_pages_to_write} total pages...")
    
    for i in range(repeat):
        for page_num in range(num_pages):
            writer.add_page(reader.pages[page_num])
            current_page = i * num_pages + page_num + 1
            if current_page % 100 == 0 or current_page == total_pages_to_write:
                print(f"  Added {current_page}/{total_pages_to_write} pages", end="\r")

    print()
    print("Writing output file...")
    
    # Write the output PDF
    with open(output_path, "wb") as f:
        writer.write(f)

    final_size = os.path.getsize(output_path)
    print(f"Done. Output: {output_path}")
    print(f"Output size: {final_size / (1024 * 1024):.2f} MB ({final_size / (1024 * 1024 * 1024):.2f} GB)")
    print(f"Total pages: {total_pages_to_write}")

    if target_mb or target_gb:
        target_bytes = int(target_gb * 1024 * 1024 * 1024) if target_gb else int(target_mb * 1024 * 1024)
        if final_size < target_bytes * 0.95:
            print(
                f"Warning: Output size ({final_size / (1024 * 1024):.2f} MB) "
                f"is significantly less than target ({target_bytes / (1024 * 1024):.2f} MB)"
            )
    
    # Validate the output PDF if requested
    if validate:
        validate_pdf(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inflate a PDF by duplicating its pages to reach a target size"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input PDF")
    parser.add_argument("--output", "-o", default="output.pdf", help="Path to output PDF")
    parser.add_argument("--repeat", type=int, help="Number of times to duplicate all pages")
    parser.add_argument("--target-mb", type=float, help="Target minimum size in MB")
    parser.add_argument("--target-gb", type=float, help="Target minimum size in GB")
    parser.add_argument("--validate", "-v", action="store_true", help="Validate the output PDF after generation")

    args = parser.parse_args()

    inflate_pdf(
        input_path=args.input,
        output_path=args.output,
        repeat=args.repeat,
        target_mb=args.target_mb,
        target_gb=args.target_gb,
        validate=args.validate,
    )
