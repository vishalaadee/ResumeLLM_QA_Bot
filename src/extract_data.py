import re
import os
import spacy
from spacy.matcher import Matcher
from io import BytesIO
from azure.identity import ClientSecretCredential
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv
import PyPDF2

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

def list_files_in_container(container_name):
    """
    Lists all PDF files available in the specified Azure Blob Storage container.

    Parameters:
    - container_name (str): Name of the Azure Blob Storage container.

    Returns:
    - list: List of file names available in the container.
    """
    try:
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credentials)
        container_client = blob_service_client.get_container_client(container=container_name)

        blobs = container_client.list_blobs()
        files = [blob.name for blob in blobs if blob.name.endswith('.pdf')]  

        return files

    except Exception as e:
        print(f"Error listing files: {e}")
        return []  

def get_blob_data(blob_name, container_name='nlp'):
    """
    Retrieves PDF data from Azure Blob Storage and extracts text from it.

    Parameters:
    - blob_name (str): The name of the PDF file in Azure Blob Storage.
    - container_name (str): The container in Azure Blob Storage where the file resides.

    Returns:
    - str: Extracted text from the PDF.
    """
    try:
        
        blob_service_client = BlobServiceClient(account_url=account_url, credential=credentials)
        container_client = blob_service_client.get_container_client(container=container_name)
        blob_client = container_client.get_blob_client(blob=blob_name)
    
        
        pdf_data = blob_client.download_blob().readall()

        
        with BytesIO(pdf_data) as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            pdf_text = ''
            for page in reader.pages:
                pdf_text += page.extract_text()

        return pdf_text

    except Exception as e:
        print(f"Error downloading or reading file: {e}")
        return ""  

def preprocess_text(text):
    """
    Preprocesses the extracted text by removing unwanted characters and normalizing.

    Parameters:
    - text (str): Raw text extracted from the resume.

    Returns:
    - str: Preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)  
    return text

def extract_email(text):
    """
    Extracts the first occurring email address from the provided text.

    Parameters:
    - text (str): The text from which to extract an email address.

    Returns:
    - str or None: The extracted email address if found, otherwise None.
    """
    pattern = r"[\w\.-]+@[\w\.-]+"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None

def extract_phone_number(text):
    """
    Extracts the first occurring phone number from the provided text.

    Parameters:
    - text (str): The text from which to extract a phone number.

    Returns:
    - str or None: The extracted phone number if found, otherwise None.
    """
    pattern = r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None

def extract_name(text):
    """
    Extracts the name from the provided text using SpaCy's Named Entity Recognition.

    Parameters:
    - text (str): The text from which to extract a name.

    Returns:
    - str or None: The extracted name if found, otherwise None.
    """
    nlp = spacy.load("en_core_web_trf")  # Load SpaCy model
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    
    if names:
        
        return names[0]
    return None

def extract_portfolio_linkedin(text):
    """
    Extracts the first occurring portfolio or LinkedIn URL from the provided text.

    Parameters:
    - text (str): The text from which to extract a URL.

    Returns:
    - str or None: The extracted URL if found, otherwise None.
    """
    pattern = r"(https?://[^\s]+)|(linkedin\.com/[^\s]+)"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None

def get_nlp_doc(text):
    """
    Processes the provided text with SpaCy to create a SpaCy Document object.

    Parameters:
    - text (str): The text to process.

    Returns:
    - Doc: A SpaCy Document object.
    """
    nlp = spacy.load("en_core_web_trf")  # Transformer-based model
    return nlp(text)

def extract_categories(text):
    """
    Extracts and categorizes different sections of the resume based on predefined keywords using SpaCy's Matcher.

    Parameters:
    - text (str): The resume text.

    Returns:
    - dict: A dictionary with keys as categories and values as the extracted text for each category.
    """
    nlp = spacy.load("en_core_web_trf")  # Transformer-based model
    matcher = Matcher(nlp.vocab)

    patterns = {
        "Education": [[{"LOWER": "education"}]],
        "Experience": [[{"LOWER": "experience"}]],
        "SKILLS, INTERESTS AND EXTRACURRICULAR ACTIVITIES": [
            [{"LOWER": "skills"}],
            [{"LOWER": "interests"}],
            [{"LOWER": "extracurricular"}, {"LOWER": "activities"}],
        ],
        "Key Achievements": [[{"LOWER": "key"}, {"LOWER": "achievements"}]],
        "Personal Statement": [[{"LOWER": "personal"}, {"LOWER": "statement"}]],
    }

    for category, pattern in patterns.items():
        matcher.add(category, pattern)

    doc = get_nlp_doc(text)
    matches = matcher(doc)
    categories = {category: "" for category in patterns.keys()}
    start_pos = {category: None for category in patterns.keys()}

    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  
        start_pos[rule_id] = start

    for category, pos in start_pos.items():
        if pos is not None:
            end_pos = len(doc)  
            for other_cat, other_pos in start_pos.items():
                if other_pos is not None and other_pos > pos:
                    end_pos = min(end_pos, other_pos)
            span = doc[pos:end_pos]  
            categories[category] = span.text.strip().replace(
                category.upper() + "\n\n", ""
            )

    return categories

def extract_experiences(doc):
    """
    Extracts detailed work experiences from the 'Experience' section of a resume.

    Parameters:
    - doc (Doc): A SpaCy Document object containing the 'Experience' section text.

    Returns:
    - list: A list of dictionaries, each representing an extracted work experience.
    """
    nlp = spacy.load("en_core_web_trf") 
    doc = nlp(doc["Experience"])

    experiences = []
    index = -1
    for sent in doc.sents:
        labels = [ent.label_ for ent in sent.ents]

        if "DATE" in labels and "ORG" in labels and "GPE" in labels:
            experiences.append(
                {
                    "date": "",
                    "place": "",
                    "org-name": "",
                    "org-description": "",
                    "role": "",
                    "description": "",
                }
            )

            index += 1

            date = [ent.text for ent in sent.ents if ent.label_ == "DATE"]
            experiences[index]["date"] = date[0] if date else "N/A"

            org = [
                ent.text
                for ent in sent.ents
                if ent.label_ == "ORG" and "experience" not in ent.text.lower()
            ]
            experiences[index]["org-name"] = org[0] if org else "N/A"

            place = [ent.text for ent in sent.ents if ent.label_ == "GPE"]
            experiences[index]["place"] = place[0] if place else "N/A"

            sent_split = sent.text.split("\n")
            sent_split = [s for s in sent_split if s.strip() and "experience" not in s.lower()]

            if len(sent_split) > 1:
                role = sent_split[0].split(",")[3] if len(sent_split[0].split(",")) > 3 else "N/A"
                experiences[index]["role"] = role.replace(date[0], "").strip()

                org_description = sent_split[1] if len(sent_split) > 1 else "N/A"
                experiences[index]["org-description"] = org_description
            else:
                experiences[index]["role"] = "N/A"
                experiences[index]["org-description"] = "N/A"

        else:
            if index >= 0:
                experiences[index]["description"] += sent.text.replace("\n", " ") + " "
            else:
                print("No date, org, or place found in the sentence")

    return experiences

def extract_education(doc):
    """
    Extracts detailed educational information from the 'Education' section of a resume.

    Parameters:
    - doc (Doc): A SpaCy Document object containing the 'Education' section text.

    Returns:
    - list: A list of dictionaries, each representing extracted educational details.
    """
    nlp = spacy.load("en_core_web_trf")
    doc = nlp(doc["Education"])

    education = []
    index = -1
    for sent in doc.sents:
        labels = [ent.label_ for ent in sent.ents]

        if "DATE" in labels and "ORG" in labels and "GPE" in labels:
            education.append(
                {
                    "date": "",
                    "place": "",
                    "institution": "",
                    "formation-name": "",
                    "description": "",
                }
            )

            index += 1

            date = [ent.text for ent in sent.ents if ent.label_ == "DATE"]
            education[index]["date"] = date[0] if date else "N/A"

            institution = [ent.text for ent in sent.ents if ent.label_ == "ORG"]
            education[index]["institution"] = institution[0] if institution else "N/A"

            place = [ent.text for ent in sent.ents if ent.label_ == "GPE"]
            education[index]["place"] = place[0] if place else "N/A"

            sent_split = sent.text.split("\n")
            sent_split = [s for s in sent_split if s.strip() and "education" not in s.lower()]

            if len(sent_split) > 1:
                formation_name = sent_split[1] if len(sent_split) > 1 else "N/A"
                education[index]["formation-name"] = formation_name

                description = sent_split[2] if len(sent_split) > 2 else "N/A"
                education[index]["description"] = "Modules: " + description + " "
            else:
                education[index]["formation-name"] = "N/A"
                education[index]["description"] = "N/A"

        else:
            if index >= 0:
                education[index]["description"] += sent.text.replace("\n", " ") + " "
            else:
                print("No date, org, or place found in the sentence")

    return education

def extract_resume_data(text):
    """
    Integrates all extraction functions to structure a resume's unstructured text into categorized data.

    Parameters:
    - text (str): The full text of the resume.

    Returns:
    - dict: A dictionary with structured resume data, categorized into education, experience, etc.
    """
    email = extract_email(text)
    phone_number = extract_phone_number(text)
    name = extract_name(text)
    portfolio_linkedin = extract_portfolio_linkedin(text)
    categories = extract_categories(text)

    data = {
        "education": extract_education(categories),
        "experience": extract_experiences(categories),
        "skills_interests_and_extracurricular_activities": categories.get('SKILLS, INTERESTS AND EXTRACURRICULAR ACTIVITIES', "N/A"),
        "key_achievements": categories.get('Key Achievements', "N/A"),
        "personal_statement": categories.get('Personal Statement', "N/A"),
        "contact_information": {
            "Name": name,
            "Email": email,
            "Phone Number": phone_number,
            "Portfolio/LinkedIn": portfolio_linkedin
        }
    }

    return data


if __name__ == "__main__":
    try:
        pdf_text = get_blob_data()  # Retrieve PDF text from Azure Blob Storage
        text = preprocess_text(pdf_text)  # Preprocess the text
        resume_data = extract_resume_data(text)  # Extract structured resume data

        print(resume_data)
    except Exception as e:
        print(f"An error occurred: {e}")
