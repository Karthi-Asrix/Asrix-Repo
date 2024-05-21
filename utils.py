import fitz
import io
import os
import easyocr
import requests
import re
import tabula
from subprocess import CalledProcessError
from PIL import Image
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredFileLoader,
    SeleniumURLLoader
)


def append_doc_log(doc_name, doc_log):
    f = open(doc_log, "a+")  # append new document name at doc_log directory
    f.write(f'{doc_name}\n')
    f.close()


def extract_pdf(doc_path, temp_dir):

    # Clearing the text file after each use
    output_file = f"{temp_dir}/output.txt"
    f = open(output_file, 'w')
    f.close()

    output_dir = f"{temp_dir}/pdf_image/"
    os.makedirs(output_dir)

    # Desired output image format
    output_format = "png"

    # Minimum width and height for extracted images
    min_width = 10
    min_height = 10

    # Setting the language of the reader for text extraction from image
    reader = easyocr.Reader(['en'])

    # Load the PDF document
    loader_for_text = PyPDFLoader(doc_path)
    loader_for_image = fitz.open(doc_path)

    # Extract text
    pdf_text = loader_for_text.load()

    # looping through PDF pages
    for page_index in range(len(loader_for_image)):

        print("Extrating page " + str(page_index))

        # Writing text into text file
        text_file = open(output_file, "a+")
        text_file.write(pdf_text[page_index].page_content)
        text_file.close()

        # Get the page itself
        page = loader_for_image[page_index]

        # Get image list
        image_list = page.get_images(full=True)

        # Iterate over the images on the page
        for image_index, img in enumerate(image_list, start=1):
            # Get the XREF of the image
            xref = img[0]
            # Extract the image bytes
            base_image = loader_for_image.extract_image(xref)
            image_bytes = base_image["image"]
            # Load it to PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Check if the image meets the minimum dimensions and save it
            if image.width >= min_width and image.height >= min_height:
                image_path = os.path.join(output_dir, f"image{page_index + 1}_{image_index}.{output_format}")
                image.save(
                    open(image_path, "wb"),
                    format=output_format.upper())

                # read the image
                result = reader.readtext(image_path, detail=0, paragraph=True)

                # write image text into the text file
                text_file = open(output_file, "a+")
                for text in result:
                    text_file.write(text + "\n")
                text_file.close()

        # Deleting image after extracting text
        for image_index, img in enumerate(image_list, start=1):
            img_filename = os.path.join(output_dir, f"image{page_index + 1}_{image_index}.{output_format}")
            os.remove(img_filename)

        while True:  # Extracting proper table format into output_file
            try:
                table_data = tabula.read_pdf(doc_path, pages=page_index+1)
                if len(table_data) != 0:
                    text_file = open(output_file, "a+")
                    for i in range(len(table_data)):
                        text_file.write(str(table_data[i]))
                    text_file.close()
                    print(table_data)
                break
            except CalledProcessError:
                print(f"Skipping unsupported table found in page {page_index}")
                break

    loader = UnstructuredFileLoader(output_file)
    pages = loader.load_and_split()
    os.remove(output_file)
    os.rmdir(output_dir)

    return pages


def doc_load(doc_url, doc_log, temp_dir):
    match = re.search(r'\/([^\/]+)\.(pdf|txt|docx?)$', doc_url)
    doc_name = match.group(1) if match else None

    if doc_name:
        response = requests.get(doc_url)
        if response.status_code == 200:
            doc_path = os.path.join(temp_dir, os.path.basename(doc_url))
            with open(doc_path, 'wb') as f:
                f.write(response.content)
                print("File successfully downloaded.")
        else:
            print("Unsuccessful file download.")
            doc_path = None

        file_pdf = re.search(r"pdf$", doc_path)

        if file_pdf:  # Checks if file type is pdf, else .doc/.docx/.txt
            # pages = extract_pdf(doc_path, temp_dir)
            loader = PyPDFLoader(doc_path)
        else:
            loader = UnstructuredFileLoader(doc_path)

        pages = loader.load_and_split()
        os.remove(doc_path)

    else:
        loader = SeleniumURLLoader([doc_url],
                                   binary_location="/snap/bin/chromium")
        pages = loader.load()
        doc_name = doc_url.replace("/", "_")

    print(doc_name)
    append_doc_log(doc_name, doc_log)

    return pages
