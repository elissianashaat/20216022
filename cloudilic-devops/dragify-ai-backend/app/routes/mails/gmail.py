import base64
from email.mime.text import MIMEText
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request as GoogleRequest
from googleapiclient.discovery import build
import requests
import io
import mimetypes

SCOPES = [
    'https://www.googleapis.com/auth/gmail.compose',
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]

router = APIRouter()
route_metadata = {
    "prefix": "/api/mails/gmail",
    "tags": ["gmail"]
}

# Pydantic models
class EmailContent(BaseModel):
    id: str
    subject: str
    sender: str
    received_time: str
    body: str
    is_read: bool

class EmailAttachment(BaseModel):
    filename: str
    download_url: str
    mime_type: str

class EmailInfo(BaseModel):
    id: str
    subject: str
    attachments: List[EmailAttachment]
    
class EmailMessage(BaseModel):
    to: str
    subject: str
    body: str

class DraftMessage(BaseModel):
    to: str
    subject: str
    body: str

class DraftInfo(BaseModel):
    id: str

class DraftListResponse(BaseModel):
    drafts: List[DraftInfo]

def authenticate(token: str):
    creds = Credentials(token=token, scopes=SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(GoogleRequest())
    return creds

def create_message(to, subject, body):
    message = MIMEText(body)
    message['to'] = to
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

@router.post("/send_email")
async def send_email(email: EmailMessage, token: str = Query(...)):
    creds = authenticate(token)
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        message = create_message(email.to, email.subject, email.body)
        sent_message = service.users().messages().send(userId='me', body=message).execute()
        return {"message": "Email sent successfully", "id": sent_message['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")

@router.post("/create_draft")
async def create_draft(draft: DraftMessage, token: str = Query(...)):
    creds = authenticate(token)
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        message = create_message(draft.to, draft.subject, draft.body)
        draft = service.users().drafts().create(userId='me', body={'message': message}).execute()
        return {"message": "Draft created successfully", "id": draft['id']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating draft: {str(e)}")

@router.get("/list_drafts", response_model=DraftListResponse)
async def list_drafts(token: str = Query(...)):
    creds = authenticate(token)
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        results = service.users().drafts().list(userId='me').execute()
        drafts = results.get('drafts', [])
        return DraftListResponse(drafts=[DraftInfo(id=draft['id']) for draft in drafts])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing drafts: {str(e)}")


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
        mime_type = 'application/pdf'  # Default to PDF if type can't be guessed
    
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

@router.get("/watch_emails", response_model=List[EmailInfo])
async def watch_emails(
    token: str = Query(...),
    keywords: str = Query(..., description="Comma-separated list of keywords to search for in attachment names"),
    max_attachments: int = Query(10, description="Maximum number of attachments to return")
):
    creds = authenticate(token)
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        keyword_list = [k.strip().lower() for k in keywords.split(',')]

        query = "has:attachment"
        
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])

        matching_emails = []
        total_attachments = 0

        for message in messages:
            if total_attachments >= max_attachments:
                break

            msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
            
            if 'payload' in msg and 'headers' in msg['payload']:
                subject = next((header['value'] for header in msg['payload']['headers'] if header['name'].lower() == 'subject'), '')
                
                attachments = []
                
                parts = [msg['payload']]
                while parts:
                    part = parts.pop(0)
                    if 'parts' in part:
                        parts.extend(part['parts'])
                    if 'filename' in part and part['filename']:
                        filename = part['filename'].lower()
                        if any(keyword in filename for keyword in keyword_list):
                            if 'body' in part and 'attachmentId' in part['body']:
                                attachment = service.users().messages().attachments().get(
                                    userId='me', messageId=message['id'], id=part['body']['attachmentId']
                                ).execute()

                                file_data = base64.urlsafe_b64decode(attachment['data'])
                                
                                mime_type, _ = mimetypes.guess_type(filename)
                                if mime_type is None:
                                    mime_type = 'application/octet-stream'
                                
                                file_url = send_attachment_to_api(file_data, filename)
                                
                                if file_url:
                                    attachments.append(EmailAttachment(
                                        filename=part['filename'],
                                        download_url=file_url,
                                        mime_type=mime_type
                                    ))
                                    total_attachments += 1
                                    if total_attachments >= max_attachments:
                                        break
                
                if attachments:
                    matching_emails.append(EmailInfo(id=message['id'], subject=subject, attachments=attachments))

        return matching_emails

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error watching emails: {str(e)}")

@router.get("/fetch_new_emails_with_attachments", response_model=List[EmailInfo])
async def fetch_new_emails_with_attachments(
    token: str = Query(...),
    keywords: str = Query(..., description="Comma-separated list of keywords to search for in attachment names"),
    max_attachments: int = Query(10, description="Maximum number of attachments to return")
):
    creds = authenticate(token)
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        keyword_list = [k.strip().lower() for k in keywords.split(',')]

        query = "is:unread has:attachment"
        
        results = service.users().messages().list(userId='me', q=query).execute()
        messages = results.get('messages', [])

        new_emails = []
        total_attachments = 0

        for message in messages:
            if total_attachments >= max_attachments:
                break

            msg = service.users().messages().get(userId='me', id=message['id'], format='full').execute()
            
            if 'payload' in msg and 'headers' in msg['payload']:
                subject = next((header['value'] for header in msg['payload']['headers'] if header['name'].lower() == 'subject'), '')
                
                attachments = []
                
                parts = [msg['payload']]
                while parts:
                    part = parts.pop(0)
                    if 'parts' in part:
                        parts.extend(part['parts'])
                    if 'filename' in part and part['filename']:
                        filename = part['filename'].lower()
                        if any(keyword in filename for keyword in keyword_list):
                            if 'body' in part and 'attachmentId' in part['body']:
                                attachment = service.users().messages().attachments().get(
                                    userId='me', messageId=message['id'], id=part['body']['attachmentId']
                                ).execute()

                                file_data = base64.urlsafe_b64decode(attachment['data'])
                                
                                mime_type, _ = mimetypes.guess_type(filename)
                                if mime_type is None:
                                    mime_type = 'application/octet-stream'
                                
                                file_url = send_attachment_to_api(file_data, filename)
                                
                                if file_url:
                                    attachments.append(EmailAttachment(
                                        filename=part['filename'],
                                        download_url=file_url,
                                        mime_type=mime_type
                                    ))
                                    total_attachments += 1
                                    if total_attachments >= max_attachments:
                                        break
                
                if attachments:
                    new_emails.append(EmailInfo(id=message['id'], subject=subject, attachments=attachments))
                    
                    # Mark the email as read
                    service.users().messages().modify(
                        userId='me',
                        id=message['id'],
                        body={'removeLabelIds': ['UNREAD']}
                    ).execute()

        return new_emails

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching new emails: {str(e)}")
    
@router.get("/search_email_content", response_model=List[EmailContent])
async def search_email_content(
    token: str = Query(...),
    search_query: str = Query(..., description="Search query to find in email content"),
    max_results: int = Query(10)
):
    creds = authenticate(token)
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        results = service.users().messages().list(
            userId='me',
            q=search_query,
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        email_contents = []

        for message in messages:
            msg = service.users().messages().get(
                userId='me',
                id=message['id'],
                format='full'
            ).execute()
            
            if 'payload' in msg and 'headers' in msg['payload']:
                headers = msg['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
                sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
                date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
                
                # Get email body
                body = ''
                if 'parts' in msg['payload']:
                    for part in msg['payload']['parts']:
                        if part['mimeType'] == 'text/plain':
                            if 'data' in part['body']:
                                body = base64.urlsafe_b64decode(part['body']['data']).decode()
                elif 'body' in msg['payload'] and 'data' in msg['payload']['body']:
                    body = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode()
                
                # Check if email is read
                is_read = 'UNREAD' not in msg['labelIds'] if 'labelIds' in msg else True
                
                email_contents.append(EmailContent(
                    id=message['id'],
                    subject=subject,
                    sender=sender,
                    received_time=date,
                    body=body,
                    is_read=is_read
                ))
                
        return email_contents

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching emails: {str(e)}")

class EmailTitle(BaseModel):
    id: str
    subject: str
    sender: str
    received_time: str

@router.get("/search_email_titles", response_model=List[EmailTitle])
async def search_email_titles(
    token: str = Query(...),
    search_query: str = Query(..., description="Search query to find in email titles"),
    max_results: int = Query(10)
):
    creds = authenticate(token)
    service = build('gmail', 'v1', credentials=creds)
    
    try:
        # Add subject: prefix to search only in subjects
        query = f"subject:{search_query}"
        results = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()
        
        messages = results.get('messages', [])
        email_titles = []

        for message in messages:
            msg = service.users().messages().get(
                userId='me',
                id=message['id'],
                format='metadata',
                metadataHeaders=['subject', 'from', 'date']
            ).execute()
            
            if 'payload' in msg and 'headers' in msg['payload']:
                headers = msg['payload']['headers']
                subject = next((h['value'] for h in headers if h['name'].lower() == 'subject'), '')
                sender = next((h['value'] for h in headers if h['name'].lower() == 'from'), '')
                date = next((h['value'] for h in headers if h['name'].lower() == 'date'), '')
                
                email_titles.append(EmailTitle(
                    id=message['id'],
                    subject=subject,
                    sender=sender,
                    received_time=date
                ))
                
        return email_titles

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching email titles: {str(e)}")