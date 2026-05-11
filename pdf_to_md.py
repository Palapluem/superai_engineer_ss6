import fitz  # PyMuPDF
import os

pdf_path = r"C:\Users\CPE KMUTT\Documents\GitHub\superai_engineer_ss6\Level 2\เอกสารการเรียนการสอน\Week 3\11052026\ช่วงเช่้า\Chapter 10 Time Series Analysis.pdf"
output_folder = os.path.dirname(pdf_path)
md_file_path = os.path.join(output_folder, "Chapter_10_Time_Series_Analysis.md")
image_folder = os.path.join(output_folder, "images")

if not os.path.exists(image_folder):
    os.makedirs(image_folder)

try:
    doc = fitz.open(pdf_path)
    with open(md_file_path, "w", encoding="utf-8") as md_file:
        md_file.write("# Chapter 10 Time Series Analysis\n\n")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            
            md_file.write(f"## Slide {page_num + 1}\n\n")
            if text.strip():
                md_file.write(text + "\n\n")
            
            # Extract images
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                image_filename = f"slide_{page_num+1}_img_{img_index+1}.{ext}"
                image_filepath = os.path.join(image_folder, image_filename)
                
                with open(image_filepath, "wb") as img_file:
                    img_file.write(image_bytes)
                    
                md_file.write(f"![Image {img_index+1}](images/{image_filename})\n\n")

            md_file.write("---\n\n")

    print(f"Success! Exported Markdown to {md_file_path}")
    print(f"Exported images to {image_folder}")
except Exception as e:
    print(f"Error: {e}")
