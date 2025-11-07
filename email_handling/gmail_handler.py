import base64
import json
from pathlib import Path
from typing import Optional

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from email_handling.email_objects import EmailMessage, EmailThread


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_credentials():
    creds = None
    token_path = Path("config/gcloud_oauth_token.json")
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if creds is None or not creds.valid:
        if creds is not None and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "config/gcloud_oauth_credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())
    return creds


class GmailHandler:
    """
    Manages a persistent database of Gmail messages and threads.
    Fetches new emails from Gmail API and stores them locally.
    """
    
    def __init__(self, db_path: str = "config/gmail_db.json"):
        """
        Initialize the GmailHandler with a database file path.
        
        Args:
            db_path: Path to the JSON database file
        """
        self.db_path = Path(db_path)
        self.service = None
        self.threads = {}  # thread_id -> EmailThread
        self.messages = {}  # message_id -> EmailMessage
        self._load_database()
    
    def _load_database(self):
        """Load the email database from disk if it exists."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                
            # Reconstruct EmailMessage objects
            for msg_id, msg_data in data.get('messages', {}).items():
                self.messages[msg_id] = EmailMessage(
                    subject=msg_data['subject'],
                    sender=msg_data['sender'],
                    recipients=msg_data['recipients'],
                    body=msg_data['body'],
                    labels=msg_data.get('labels', []),
                    message_id=msg_data['message_id']
                )
            
            # Reconstruct EmailThread objects
            for thread_id, thread_data in data.get('threads', {}).items():
                messages = [self.messages[msg_id] for msg_id in thread_data['message_ids'] 
                           if msg_id in self.messages]
                self.threads[thread_id] = EmailThread(
                    thread_id=thread_id,
                    messages=messages
                )
    
    def _save_database(self):
        """Save the email database to disk."""
        # Ensure the directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Serialize messages
        messages_data = {}
        for msg_id, msg in self.messages.items():
            messages_data[msg_id] = {
                'subject': msg.subject,
                'sender': msg.sender,
                'recipients': msg.recipients,
                'body': msg.body,
                'labels': msg.labels,
                'message_id': msg.message_id
            }
        
        # Serialize threads
        threads_data = {}
        for thread_id, thread in self.threads.items():
            threads_data[thread_id] = {
                'thread_id': thread_id,
                'message_ids': [msg.message_id for msg in thread.messages]
            }
        
        data = {
            'messages': messages_data,
            'threads': threads_data
        }
        
        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _initialize_service(self):
        """Initialize the Gmail API service if not already done."""
        if self.service is None:
            creds = get_credentials()
            self.service = build('gmail', 'v1', credentials=creds)
    
    def _parse_message(self, raw_message: dict) -> EmailMessage:
        """
        Parse a raw Gmail API message into an EmailMessage object.
        
        Args:
            raw_message: Raw message dict from Gmail API
            
        Returns:
            EmailMessage object
        """
        headers = raw_message['payload']['headers']
        header_dict = {h['name'].lower(): h['value'] for h in headers}
        
        subject = header_dict.get('subject', '(No Subject)')
        sender = header_dict.get('from', '')
        to = header_dict.get('to', '')
        recipients = [r.strip() for r in to.split(',') if r.strip()]
        
        # Extract body
        body = ""
        if 'parts' in raw_message['payload']:
            for part in raw_message['payload']['parts']:
                if part['mimeType'] == 'text/plain':
                    if 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
        elif 'body' in raw_message['payload'] and 'data' in raw_message['payload']['body']:
            body = base64.urlsafe_b64decode(raw_message['payload']['body']['data']).decode('utf-8')
        
        labels = raw_message.get('labelIds', [])
        message_id = raw_message['id']
        
        return EmailMessage(
            subject=subject,
            sender=sender,
            recipients=recipients,
            body=body,
            labels=labels,
            message_id=message_id
        )
    
    def update_emails(self, max_results: int = 100):
        """
        Fetch new emails from Gmail and update the local database.
        
        Args:
            max_results: Maximum number of threads to fetch
        """
        self._initialize_service()
        
        try:
            # Fetch threads
            threads_response = self.service.users().threads().list(
                userId='me',
                maxResults=max_results
            ).execute()
            
            threads = threads_response.get('threads', [])
            
            for thread_info in threads:
                thread_id = thread_info['id']
                
                # Skip if we already have this thread
                if thread_id in self.threads:
                    continue
                
                # Fetch full thread details
                thread_data = self.service.users().threads().get(
                    userId='me',
                    id=thread_id
                ).execute()
                
                thread_messages = []
                for raw_message in thread_data.get('messages', []):
                    message_id = raw_message['id']
                    
                    # Skip if we already have this message
                    if message_id in self.messages:
                        thread_messages.append(self.messages[message_id])
                        continue
                    
                    # Parse and store the message
                    email_message = self._parse_message(raw_message)
                    self.messages[message_id] = email_message
                    thread_messages.append(email_message)
                
                # Create and store the thread
                email_thread = EmailThread(
                    thread_id=thread_id,
                    messages=thread_messages
                )
                self.threads[thread_id] = email_thread
            
            # Save the updated database
            self._save_database()
            
            print(f"Updated database with {len(threads)} threads")
            
        except HttpError as error:
            print(f"An error occurred: {error}")
    
    def get_thread(self, thread_id: str) -> Optional[EmailThread]:
        """
        Get a specific email thread by ID.
        
        Args:
            thread_id: The thread ID to retrieve
            
        Returns:
            EmailThread if found, None otherwise
        """
        return self.threads.get(thread_id)
    
    def get_all_threads(self) -> list[EmailThread]:
        """
        Get all email threads in the database.
        
        Returns:
            List of all EmailThread objects
        """
        return list(self.threads.values())
    
    def get_message(self, message_id: str) -> Optional[EmailMessage]:
        """
        Get a specific email message by ID.
        
        Args:
            message_id: The message ID to retrieve
            
        Returns:
            EmailMessage if found, None otherwise
        """
        return self.messages.get(message_id)

