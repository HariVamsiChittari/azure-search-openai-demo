import argparse
import os
from math import ceil
from io import BytesIO

# Check for pypdf separately from reportlab
try:
    from pypdf import PdfReader, PdfWriter
    PYPDF_AVAILABLE = True
except ImportError:
    PYPDF_AVAILABLE = False

# Check for reportlab separately (optional, only needed for unique content)
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


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


def add_page_number_overlay(page, page_number, add_large_image=False):
    """Add substantial unique content overlay to prevent PDF compression and increase file size."""
    # Create a PDF with unique content
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=letter)
    
    if add_large_image:
        # Add a large invisible image to significantly increase file size
        # Create a unique bitmap pattern for this page
        from PIL import Image
        from reportlab.lib.utils import ImageReader
        import random
        
        # Create a 1000x1000 image with unique random noise based on page number
        random.seed(page_number)
        img_size = (1000, 1000)
        img_data = bytes([random.randint(250, 255) for _ in range(img_size[0] * img_size[1] * 3)])
        img = Image.frombytes('RGB', img_size, img_data)
        
        # Wrap in ImageReader for reportlab
        img_reader = ImageReader(img)
        
        # Draw image at very low opacity (almost invisible)
        can.setFillAlpha(0.01)
        can.drawImage(img_reader, 0, 0, width=612, height=792, mask='auto')
        can.setFillAlpha(1.0)
    
    # Add page number in very small text at bottom right
    can.setFont("Helvetica", 1)
    can.setFillColorRGB(0.99, 0.99, 0.99)
    can.drawString(580, 5, f"pg{page_number}")
    
    # Add grid of unique text
    can.setFont("Helvetica", 0.1)
    can.setFillColorRGB(0.999, 0.999, 0.999)
    unique_text = f"PAGE_{page_number}_" * 50
    
    for y in range(10, 800, 20):
        for x in range(10, 600, 100):
            offset = (y * 30 + x) % 50
            can.drawString(x, y, unique_text[offset:offset+20])
    
    can.save()
    packet.seek(0)
    
    # Read the overlay PDF
    overlay_pdf = PdfReader(packet)
    overlay_page = overlay_pdf.pages[0]
    
    # Merge the overlay onto the original page
    page.merge_page(overlay_page)
    
    return page


def inflate_pdf(input_path, output_path, repeat=None, target_mb=None, target_gb=None, validate=False, add_unique_content=True, add_images=False):
    """Inflate PDF by duplicating pages, creating a valid multi-page PDF."""
    if not PYPDF_AVAILABLE:
        raise ImportError("pypdf is required for page duplication. Install with: pip install pypdf")
    
    if add_unique_content and not REPORTLAB_AVAILABLE:
        print("Warning: reportlab not available. Pages will be duplicated without unique content.")
        print("This may result in smaller file size due to PDF compression deduplicating identical pages.")
        print("Install with: pip install reportlab")
        add_unique_content = False
    
    if add_images:
        try:
            from PIL import Image
            print("Using image embedding to reach target file size...")
        except ImportError:
            print("Warning: PIL/Pillow not available for image embedding.")
            print("Install with: pip install Pillow")
            add_images = False
    
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
    if add_unique_content:
        print("Adding unique content to each page to prevent compression...")
    
    for i in range(repeat):
        for page_num in range(num_pages):
            # Get the original page
            page = reader.pages[page_num]
            
            # Add unique content to prevent PDF compression from deduplicating pages
            if add_unique_content:
                page = add_page_number_overlay(page, i * num_pages + page_num + 1, add_large_image=add_images)
            
            writer.add_page(page)
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
            if not add_unique_content:
                print("Tip: Install reportlab and use unique content to prevent compression:")
                print("  pip install reportlab")
    
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
    parser.add_argument("--no-unique-content", action="store_true", help="Don't add unique content to pages")
    parser.add_argument("--add-images", action="store_true", help="Embed large images to reach target size (slower but guaranteed size)")

    args = parser.parse_args()

    inflate_pdf(
        input_path=args.input,
        output_path=args.output,
        repeat=args.repeat,
        target_mb=args.target_mb,
        target_gb=args.target_gb,
        validate=args.validate,
        add_unique_content=not args.no_unique_content,
        add_images=args.add_images,
    )
