import csv
import os
import time
from datetime import datetime

import requests
from dotenv import load_dotenv

# Import your project's custom data directory path
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR

# Load variables from the .env file
load_dotenv()


def fetch_all_contacts(api_token: str) -> list:
    """
    Fetch all contacts from the respond.io API using POST.
    Handles pagination, formats timestamps, and logs progress.
    """
    # Added ?limit=100 to pull data 10x faster
    url = "https://api.respond.io/v2/contact/list?limit=100"
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Accept": "application/json",
        "Content-Type": "application/json"
    }
    
    payload = {
        "search": "",
        "filter": {
            "$and": []
        },
        "timezone": "Africa/Nairobi"
    }
    
    all_contacts = []
    page_count = 1
    
    while url:
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=15
            )
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            
            # SAFEGUARD: If the page is empty, break the loop immediately
            if not items:
                break
                
            for item in items:
                if item.get("created_at"):
                    dt_obj = datetime.fromtimestamp(item["created_at"])
                    item["formatted_date"] = dt_obj.strftime("%d/%m/%Y %H:%M")
                else:
                    item["formatted_date"] = ""
                    
                all_contacts.append(item)
            
            # VISIBILITY: Print a status update to the terminal
            print(f"Page {page_count} complete. Total contacts fetched so far: {len(all_contacts)}")
            page_count += 1
                
            next_url = data.get("pagination", {}).get("next")
            if next_url:
                url = next_url
                time.sleep(0.5)
            else:
                url = None
                
        except requests.exceptions.RequestException as error:
            print(f"API request failed on URL {url}: {error}")
            if response is not None and response.status_code == 400:
                print(f"Server response: {response.text}")
            break
            
    return all_contacts


def export_contacts_to_csv(contacts_data: list, filename: str) -> None:
    """
    Flattens the JSON contact data and writes it to a CSV file
    matching the manual export structure.
    """
    headers = [
        "ContactID", "FirstName", "LastName", "PhoneNumber", "Email", 
        "Country", "Language", "Tags", "Status", "Lifecycle", 
        "Assignee", "LastInteractionTime", "DateTimeCreated", 
        "Channels", "branch_contact_number"
    ]

    # Use str(filename) to ensure compatibility if a Path object is passed
    with open(str(filename), mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()

        for contact in contacts_data:
            branch_number = ""
            for field in contact.get("custom_fields", []):
                if field.get("name") == "branch_contact_number":
                    branch_number = field.get("value") or ""
                    break
            
            assignee_email = ""
            if contact.get("assignee"):
                assignee_email = contact["assignee"].get("email") or ""

            tags_list = contact.get("tags", [])
            tags_str = ",".join(tags_list) if tags_list else ""

            row = {
                "ContactID": contact.get("id", ""),
                "FirstName": contact.get("firstName", ""),
                "LastName": contact.get("lastName", ""),
                "PhoneNumber": contact.get("phone", ""),
                "Email": contact.get("email", ""),
                "Country": contact.get("countryCode", ""),
                "Language": contact.get("language", ""),
                "Tags": tags_str,
                "Status": contact.get("status", ""),
                "Lifecycle": contact.get("lifecycle", ""),
                "Assignee": assignee_email,
                "LastInteractionTime": "",  
                "DateTimeCreated": contact.get("formatted_date", ""),
                "Channels": "",  
                "branch_contact_number": branch_number
            }
            writer.writerow(row)


if __name__ == "__main__":
    TOKEN = os.environ.get("RESPOND_IO_TOKEN")
    
    if not TOKEN:
        print("Error: RESPOND_IO_TOKEN not found in .env file.")
    else:
        print("Fetching contacts from respond.io...")
        contacts = fetch_all_contacts(TOKEN)
        
        if contacts:
            # Construct the target directory using the imported config
            save_directory = PROCESSED_DATA_DIR / "respond_io"
            
            # Ensure the directory exists (mkdir is the pathlib equivalent of os.makedirs)
            save_directory.mkdir(parents=True, exist_ok=True)
            
            # Combine the directory with the desired file name
            csv_file_path = save_directory / "contacts_history.csv"
            
            export_contacts_to_csv(contacts, csv_file_path)
            print(f"Success! {len(contacts)} contacts exported to: {csv_file_path}")