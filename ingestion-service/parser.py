import os
import fitz
import pdfplumber
from docx import Document
from typing import List, Dict, Any
from shared.models import DocumentType


def get_document_type(filename: str) -> DocumentType:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return DocumentType.PDF
    elif ext in ("docx", "doc"):
        return DocumentType.DOCX
    elif ext == "txt":
        return DocumentType.TXT
    elif ext in ("png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"):
        return DocumentType.IMAGE
    raise ValueError(f"Unsupported file type: {ext}")


def parse_pdf(file_path: str, tenant_id: str) -> List[Dict[str, Any]]:
    pages = []
    temp_dir = f"/tmp/{tenant_id}/images"
    os.makedirs(temp_dir, exist_ok=True)

    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages.append({"page_number": i + 1, "text": text, "images": []})

    doc = fitz.open(file_path)
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_idx, img in enumerate(page.get_images(full=True), start=1):
            xref = img[0]
            base_image = doc.extract_image(xref)
            ext = base_image["ext"]
            img_path = os.path.join(
                temp_dir, f"{os.path.basename(file_path)}_p{page_num + 1}_img{img_idx}.{ext}"
            )
            with open(img_path, "wb") as f:
                f.write(base_image["image"])
            if page_num < len(pages):
                pages[page_num]["images"].append({"path": img_path, "index": img_idx})
    doc.close()
    return pages


def parse_docx(file_path: str, tenant_id: str) -> List[Dict[str, Any]]:
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)

    temp_dir = f"/tmp/{tenant_id}/images"
    os.makedirs(temp_dir, exist_ok=True)
    images = []
    try:
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                image = rel.target_part
                ext = image.content_type.split("/")[-1]
                img_path = os.path.join(
                    temp_dir, f"{os.path.basename(file_path)}_img{len(images)}.{ext}"
                )
                with open(img_path, "wb") as f:
                    f.write(image.blob)
                images.append({"path": img_path, "index": len(images)})
    except Exception:
        pass

    return [{"page_number": 1, "text": full_text, "images": images}]


def parse_txt(file_path: str, tenant_id: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [{"page_number": 1, "text": text, "images": []}]


def parse_image(file_path: str, tenant_id: str) -> List[Dict[str, Any]]:
    return [{"page_number": 1, "text": "", "images": [{"path": file_path, "index": 1}]}]