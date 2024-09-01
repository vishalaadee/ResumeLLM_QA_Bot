from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import os
import PyPDF2
from io import BytesIO


load_dotenv()
client_id = os.environ['AZURE_CLIENT_ID']
tenant_id = os.environ['AZURE_TENANT_ID']
client_secret = os.environ['AZURE_CLIENT_SECRET']
account_url = os.environ["AZURE_STORAGE_URL"]


credentials = ClientSecretCredential(
    client_id=client_id,
    client_secret=client_secret,
    tenant_id=tenant_id
)

def get_blob_data():
    container_name = 'nlp'
    blob_name = 'CV_Vishal_Mishra.pdf'
    
    # Set client to access Azure storage container
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credentials)
    # Get the container client
    container_client = blob_service_client.get_container_client(container=container_name)
    # Get the blob client
    blob_client = container_client.get_blob_client(blob=blob_name)
    
    # Download blob data as bytes
    pdf_data = blob_client.download_blob().readall()

    # Read PDF data using PyPDF2
    pdf_text = ''
    with BytesIO(pdf_data) as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            pdf_text += page.extract_text()

    
    output_file_path = 'extracted_context.txt'
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(pdf_text)

    print(f"Extracted PDF text has been written to {output_file_path}")

if __name__ == "__main__":
    get_blob_data()
