from fastapi import FastAPI, HTTPException, Depends, APIRouter, Query
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import msal
import requests
from typing import List
import base64
import io
import mimetypes
from dotenv import load_dotenv
import os


load_dotenv()
router = APIRouter()

# Configuration
CLIENT_ID = os.getenv("MSAL_CLIENT_ID")
CLIENT_SECRET = os.getenv("MSAL_CLIENT_SECRET")
AUTHORITY = "https://login.microsoftonline.com/common"
SCOPE = ["https://graph.microsoft.com/Mail.Read", "https://graph.microsoft.com/Mail.Send", "https://graph.microsoft.com/Mail.ReadWrite"]

# Initialize MSAL application
msal_app = msal.ConfidentialClientApplication(
    CLIENT_ID, authority=AUTHORITY,
    client_credential=CLIENT_SECRET,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class EmailSchema(BaseModel):
    recipient: str
    subject: str
    body: str

class EmailAttachment(BaseModel):
    filename: str
    download_url: str
    mime_type: str

class EmailInfo(BaseModel):
    id: str
    subject: str
    attachments: List[EmailAttachment]

def send_attachment_to_api(file_data, filename):
    api_url = "https://api.dragify.ai/general/uploader-secret"
    
    headers = {
        "x-api-key": "uLdiVUo67043G997lIua"
    }
    
    # Create a file-like object from the binary data
    file_obj = io.BytesIO(file_data)
    
    # Guess the MIME type based on the filename
    mime_type, _ = mimetypes.guess_type(filename)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default if type can't be guessed
    
    files = {
        'file': (filename, file_obj, mime_type)
    }
    
    try:
        response = requests.post(api_url, headers=headers, files=files)
        response.raise_for_status()
        
        result = response.json()
        return result.get('url', '')
    except requests.RequestException as e:
        print(f"Error uploading file: {e}")
        return None

@router.get("/login")
async def login():
    auth_url = msal_app.get_authorization_request_url(
        SCOPE,
        redirect_uri="http://localhost:8000/getAToken"
    )
    return {"login_url": auth_url}

@router.get("/getAToken")
async def get_token(code: str):
    result = msal_app.acquire_token_by_authorization_code(
        code,
        scopes=SCOPE,
        redirect_uri="http://localhost:8000/getAToken"
    )
    if "access_token" in result:
        return {"access_token": result["access_token"]}
    else:
        raise HTTPException(status_code=400, detail="Failed to acquire token")

@router.post("/send-email")
async def send_email(email: EmailSchema, token: str = Depends(oauth2_scheme)):
    graph_endpoint = 'https://graph.microsoft.com/v1.0/me/sendMail'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
    email_data = {
        "message": {
            "subject": email.subject,
            "body": {
                "contentType": "Text",
                "content": email.body
            },
            "toRecipients": [
                {
                    "emailAddress": {
                        "address": email.recipient
                    }
                }
            ]
        },
        "saveToSentItems": "true"
    }

    response = requests.post(graph_endpoint, headers=headers, json=email_data)

    if response.status_code == 202:
        return {"message": "Email sent successfully!"}
    else:
        raise HTTPException(status_code=response.status_code, detail=response.json())

import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@router.get("/watch_emails", response_model=List[EmailInfo])
async def watch_emails(
    token: str = Depends(oauth2_scheme),
    keywords: str = Query(..., description="Comma-separated list of keywords to search for"),
    max_results: int = Query(10)
):
    graph_endpoint = f'https://graph.microsoft.com/v1.0/me/messages?$top={max_results}&$orderby=receivedDateTime desc'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    logger.info(f"Fetching emails from: {graph_endpoint}")
    response = requests.get(graph_endpoint, headers=headers)

    if response.status_code != 200:
        logger.error(f"Error response from Graph API: {response.status_code} - {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.json())

    emails = response.json().get('value', [])
    logger.info(f"Retrieved {len(emails)} emails")

    keyword_list = [k.strip().lower() for k in keywords.split(',')]
    logger.info(f"Searching for keywords: {keyword_list}")

    matching_emails = []

    for email in emails:
        subject = email['subject'].lower()
        logger.info(f"Checking email: {subject}")
        logger.info(f"Has attachments: {email.get('hasAttachments', False)}")

        subject_match = any(keyword in subject for keyword in keyword_list)
        
        if email.get('hasAttachments', False):
            if 'attachments' not in email:
                logger.info(f"Fetching attachments for email: {subject}")
                attachment_response = requests.get(f"https://graph.microsoft.com/v1.0/me/messages/{email['id']}/attachments", headers=headers)
                if attachment_response.status_code == 200:
                    email['attachments'] = attachment_response.json().get('value', [])
                    logger.info(f"Fetched attachments separately: {len(email['attachments'])}")
                else:
                    logger.error(f"Failed to fetch attachments: {attachment_response.status_code} - {attachment_response.text}")
                    continue

            matching_attachments = []
            for attachment in email.get('attachments', []):
                attachment_name = attachment.get('name', '').lower()
                logger.info(f"Checking attachment: {attachment_name}")
                
                if subject_match or any(keyword in attachment_name for keyword in keyword_list):
                    logger.info(f"Matching attachment found: {attachment_name}")
                    if attachment['@odata.type'] == '#microsoft.graph.fileAttachment':
                        try:
                            file_data = base64.b64decode(attachment['contentBytes'])
                            file_url = send_attachment_to_api(file_data, attachment['name'])
                            
                            if file_url:
                                logger.info(f"Attachment uploaded successfully: {file_url}")
                                matching_attachments.append(EmailAttachment(
                                    filename=attachment['name'],
                                    download_url=file_url,
                                    mime_type=attachment['contentType']
                                ))
                            else:
                                logger.warning(f"Failed to upload attachment: {attachment['name']}")
                        except Exception as e:
                            logger.error(f"Error processing attachment {attachment['name']}: {str(e)}")
                    else:
                        logger.warning(f"Skipping non-file attachment: {attachment_name}")

            if matching_attachments:
                matching_emails.append(EmailInfo(
                    id=email['id'],
                    subject=email['subject'],
                    attachments=matching_attachments
                ))
                logger.info(f"Added email to matching list: {subject}")
            else:
                logger.info(f"No matching attachments found for email: {subject}")
        elif subject_match:
            logger.info(f"Email subject matches but has no attachments: {subject}")

    logger.info(f"Returning {len(matching_emails)} matching emails")
    return matching_emails

@router.get("/fetch_new_emails_with_attachments", response_model=List[EmailInfo])
async def fetch_new_emails_with_attachments(
    token: str = Depends(oauth2_scheme),
    keywords: str = Query(..., description="Comma-separated list of keywords to search for"),
    max_results: int = Query(10, alias="max_limit")  # Changed alias to match the query parameter
):
    graph_endpoint = f'https://graph.microsoft.com/v1.0/me/messages?$top={max_results}&$orderby=receivedDateTime desc&$filter=isRead eq false'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    logger.info(f"Fetching new unread emails from: {graph_endpoint}")
    response = requests.get(graph_endpoint, headers=headers)

    if response.status_code != 200:
        logger.error(f"Error response from Graph API: {response.status_code} - {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.json())

    emails = response.json().get('value', [])
    logger.info(f"Retrieved {len(emails)} unread emails")

    keyword_list = [k.strip().lower() for k in keywords.split(',')]
    logger.info(f"Searching for keywords: {keyword_list}")

    new_matching_emails = []

    for email in emails:
        subject = email['subject'].lower()
        logger.info(f"Checking email: {subject}")
        logger.info(f"Has attachments: {email.get('hasAttachments', False)}")

        subject_match = any(keyword in subject for keyword in keyword_list)
        
        if email.get('hasAttachments', False):
            if 'attachments' not in email:
                logger.info(f"Fetching attachments for email: {subject}")
                attachment_response = requests.get(f"https://graph.microsoft.com/v1.0/me/messages/{email['id']}/attachments", headers=headers)
                if attachment_response.status_code == 200:
                    email['attachments'] = attachment_response.json().get('value', [])
                    logger.info(f"Fetched attachments separately: {len(email['attachments'])}")
                else:
                    logger.error(f"Failed to fetch attachments: {attachment_response.status_code} - {attachment_response.text}")
                    continue

            matching_attachments = []
            for attachment in email.get('attachments', []):
                attachment_name = attachment.get('name', '').lower()
                logger.info(f"Checking attachment: {attachment_name}")
                
                if subject_match or any(keyword in attachment_name for keyword in keyword_list):
                    logger.info(f"Matching attachment found: {attachment_name}")
                    if attachment['@odata.type'] == '#microsoft.graph.fileAttachment':
                        try:
                            file_data = base64.b64decode(attachment['contentBytes'])
                            file_url = send_attachment_to_api(file_data, attachment['name'])
                            
                            if file_url:
                                logger.info(f"Attachment uploaded successfully: {file_url}")
                                matching_attachments.append(EmailAttachment(
                                    filename=attachment['name'],
                                    download_url=file_url,
                                    mime_type=attachment['contentType']
                                ))
                            else:
                                logger.warning(f"Failed to upload attachment: {attachment['name']}")
                        except Exception as e:
                            logger.error(f"Error processing attachment {attachment['name']}: {str(e)}")
                    else:
                        logger.warning(f"Skipping non-file attachment: {attachment_name}")

            if matching_attachments:
                new_matching_emails.append(EmailInfo(
                    id=email['id'],
                    subject=email['subject'],
                    attachments=matching_attachments
                ))
                logger.info(f"Added email to matching list: {subject}")

                # Mark the email as read
                update_endpoint = f'https://graph.microsoft.com/v1.0/me/messages/{email["id"]}'
                update_data = {
                    "isRead": True
                }
                update_response = requests.patch(update_endpoint, headers=headers, json=update_data)
                if update_response.status_code == 200:
                    logger.info(f"Marked email as read: {subject}")
                else:
                    logger.error(f"Failed to mark email as read: {update_response.status_code} - {update_response.text}")
            else:
                logger.info(f"No matching attachments found for email: {subject}")
        elif subject_match:
            logger.info(f"Email subject matches but has no attachments: {subject}")

    logger.info(f"Returning {len(new_matching_emails)} new matching emails")
    return new_matching_emails

@router.get("/search_email_content")
async def search_email_content(
    token: str = Depends(oauth2_scheme),
    search_query: str = Query(..., description="Search query to find in email content"),
    max_results: int = Query(10)
):
    graph_endpoint = f'https://graph.microsoft.com/v1.0/me/messages?$top={max_results}&$search="{search_query}"'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    logger.info(f"Searching emails with query: {search_query}")
    response = requests.get(graph_endpoint, headers=headers)

    if response.status_code != 200:
        logger.error(f"Error response from Graph API: {response.status_code} - {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.json())

    emails = response.json().get('value', [])
    logger.info(f"Retrieved {len(emails)} matching emails")

    email_contents = []
    for email in emails:
        email_content = {
            "id": email['id'],
            "subject": email['subject'],
            "sender": email.get('from', {}).get('emailAddress', {}).get('address', ''),
            "received_time": email.get('receivedDateTime', ''),
            "body": email.get('body', {}).get('content', ''),
            "is_read": email.get('isRead', False)
        }
        email_contents.append(email_content)

    return email_contents

@router.get("/search_email_titles")
async def search_email_titles(
    token: str = Depends(oauth2_scheme),
    search_query: str = Query(..., description="Search query to find in email titles"),
    max_results: int = Query(10)
):
    graph_endpoint = f'https://graph.microsoft.com/v1.0/me/messages?$top={max_results}&$search="subject:{search_query}"&$select=id,subject,receivedDateTime,from'
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    logger.info(f"Searching email titles with query: {search_query}")
    response = requests.get(graph_endpoint, headers=headers)

    if response.status_code != 200:
        logger.error(f"Error response from Graph API: {response.status_code} - {response.text}")
        raise HTTPException(status_code=response.status_code, detail=response.json())

    emails = response.json().get('value', [])
    logger.info(f"Retrieved {len(emails)} matching email titles")

    email_titles = []
    for email in emails:
        title_info = {
            "id": email['id'],
            "subject": email['subject'],
            "sender": email.get('from', {}).get('emailAddress', {}).get('address', ''),
            "received_time": email.get('receivedDateTime', '')
        }
        email_titles.append(title_info)

    return email_titles