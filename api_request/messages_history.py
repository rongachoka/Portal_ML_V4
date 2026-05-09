import csv
import os
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

load_dotenv()


def get_existing_message_ids(filepath: Path) -> set:
    """Reads the existing CSV and returns a set of all Message IDs."""
    existing_ids = set()
    if filepath.exists():
        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                msg_id = row.get("Message ID")
                if msg_id:
                    existing_ids.add(msg_id)
    return existing_ids


def extract_message_timestamp(status_array: list) -> str:
    """
    Extracts the most relevant timestamp from the status array
    and formats it. Prefers 'sent' or 'delivered'.
    """
    if not status_array:
        return ""

    target_ts = None
    for status in status_array:
        val = status.get("value")
        ts = status.get("timestamp")
        if val in ("sent", "delivered"):
            target_ts = ts
            break

    # Fallback to the first available timestamp if 'sent'/'delivered' not found
    if not target_ts and status_array:
        target_ts = status_array[0].get("timestamp")

    if target_ts:
        try:
            # Safely cast to float to handle unexpected string types from the API
            dt_obj = datetime.fromtimestamp(float(target_ts) / 1000.0)
            return dt_obj.strftime("%d/%m/%Y %H:%M")
        except (ValueError, TypeError):
            # If the API returns complete garbage (e.g. "unknown"), fail gracefully
            return ""
            
    return ""

def extract_message_content(message_obj: dict) -> str:
    """Safely extracts text content based on the message type."""
    msg_type = message_obj.get("type")
    
    if msg_type == "text":
        return message_obj.get("text", "")
    
    elif msg_type == "quick_reply":
        title = message_obj.get("title", "")
        return title
        
    elif msg_type == "whatsapp_template":
        try:
            components = message_obj["template"]["components"]
            for comp in components:
                if comp.get("type") == "body":
                    return comp.get("text", "")
        except KeyError:
            return "[Template data missing]"
            
    return f"[{msg_type}]"


def fetch_new_messages_for_contact(
    api_token: str, contact_id: str, existing_ids: set
) -> list:
    """
    Fetches message history for a specific contact.
    Stops paginating if it hits a message ID we already have.
    """
    url = f"https://api.respond.io/v2/contact/id:{contact_id}/message/list?limit=100"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json"
    }
    
    new_messages = []
    
    while url:
        try:
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 404:
                break
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if not items:
                break
                
            for item in items:
                msg_id = str(item.get("messageId"))
                
                if msg_id in existing_ids:
                    return new_messages
                
                sender = item.get("sender", {})
                sender_source = sender.get("source", "")
                
                sender_id = ""
                if sender_source == "user":
                    sender_id = sender.get("userId", "")
                elif sender_source == "workflow":
                    sender_id = sender.get("workflowId", "")
                elif sender_source == "broadcast":
                    sender_id = sender.get("broadcastHistoryId", "")

                # Safely extract userId regardless of sender source
                # If it's a workflow or contact, this will gracefully be empty/None
                raw_user_id = sender.get("userId", "") if sender else ""

                row = {
                    "Date & Time": extract_message_timestamp(item.get("status", [])),
                    "Sender ID": sender_id,
                    "Sender Type": sender_source,
                    "User ID": raw_user_id,  # <--- NEW DEDICATED COLUMN
                    "Contact ID": item.get("contactId", ""),
                    "Message ID": msg_id,
                    "Content Type": item.get("message", {}).get("type", ""),
                    "Message Type": item.get("traffic", ""),
                    "Content": extract_message_content(item.get("message", {})),
                    "Channel ID": item.get("channelId", ""),
                    "Type": "",      
                    "Sub Type": ""   
                }
                new_messages.append(row)
                
            next_url = data.get("pagination", {}).get("next")
            if next_url:
                url = next_url
                time.sleep(0.5)
            else:
                url = None
                
        except requests.exceptions.RequestException as error:
            print(f"API failed for contact {contact_id}: {error}")
            break
            
    return new_messages


def append_messages_to_csv(messages: list, filepath: Path) -> None:
    """Appends new message dictionaries to the CSV."""
    # Added "User ID" to the master header list
    headers = [
        "Date & Time", "Sender ID", "Sender Type", "User ID", "Contact ID", 
        "Message ID", "Content Type", "Message Type", "Content", 
        "Channel ID", "Type", "Sub Type"
    ]
    
    file_exists = filepath.exists()
    
    with open(str(filepath), mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
            
        for msg in messages:
            writer.writerow(msg)


def main():
    TOKEN = os.environ.get("RESPOND_IO_TOKEN")
    if not TOKEN:
        print("Error: RESPOND_IO_TOKEN missing.")
        return

    save_directory = PROCESSED_DATA_DIR / "respond_io"
    contacts_csv = save_directory / "contacts_history.csv"
    messages_csv = save_directory / "messages_history.csv"
    
    if not contacts_csv.exists():
        print("Error: contacts_history.csv not found. Please run the contacts script first.")
        return

    print("Loading existing message IDs...")
    existing_ids = get_existing_message_ids(messages_csv)
    print(f"Loaded {len(existing_ids)} existing messages.")

    contact_ids = []
    with open(contacts_csv, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("ContactID"):
                contact_ids.append(row["ContactID"])
                
    total_contacts = len(contact_ids)
    print(f"Found {total_contacts} contacts to process.")

    total_new_messages = 0
    for i, contact_id in enumerate(contact_ids, 1):
        if i % 100 == 0:
            print(f"Processing contact {i}/{total_contacts}...")
            
        new_msgs = fetch_new_messages_for_contact(TOKEN, contact_id, existing_ids)
        
        if new_msgs:
            append_messages_to_csv(new_msgs, messages_csv)
            for msg in new_msgs:
                existing_ids.add(msg["Message ID"])
            total_new_messages += len(new_msgs)
            
        time.sleep(0.5)

    print(f"Finished processing. Added {total_new_messages} new messages.")


if __name__ == "__main__":
    main()