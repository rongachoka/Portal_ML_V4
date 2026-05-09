from dotenv import load_dotenv
import os
from pathlib import Path
import msal
import requests

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

TENANT_ID = os.getenv("MS_TENANT_ID")
CLIENT_ID = os.getenv("MS_CLIENT_ID")
CLIENT_SECRET = os.getenv("MS_CLIENT_SECRET")

AUTHORITY = f"https://login.microsoftonline.com/{TENANT_ID}"
SCOPE = ["https://graph.microsoft.com/.default"]  # Use the default scope for application permissions

def get_access_token():
    if not TENANT_ID or not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("Microsoft API credentials are not set in environment variables.")
    
    app = msal.ConfidentialClientApplication(
        CLIENT_ID,
        authority=AUTHORITY,
        client_credential=CLIENT_SECRET,
    )

    result = app.acquire_token_for_client(scopes=SCOPE)
    
    if "access_token" not in result:
        raise RuntimeError(
            f"Could not get access token. Error: {result.get('error')}",
            f"Description: {result.get('error_description')}"
        )
    
    return result["access_token"]


def graph_get(url: str, params: dict | None = None) -> dict:
    token = get_access_token()
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    response = requests.get(url, headers=headers, params=params, timeout=60)

    if not response.ok:
        raise RuntimeError(
            f"Graph request failed\n"
            f"URL: {response.url}\n"
            f"Status: {response.status_code}\n"
            f"Body: {response.text}"
        )

    return response.json()


if __name__ == "__main__":
    token = get_access_token()
    print("Acess token acquired successfully.")
    print(token[:40] + "...")
