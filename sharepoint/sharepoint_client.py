from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote
from email.utils import parsedate_to_datetime
import requests
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from Portal_ML_V4.sharepoint.sharepoint_auth import get_access_token


GRAPH_BASE = "https://graph.microsoft.com/v1.0"


class SharePointClient:
    def __init__(self, drive_id: str) -> None:
        self.drive_id = drive_id

    def _headers(self) -> dict[str, str]:
        token = get_access_token()
        return {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

    def _get_json(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.get(url, headers=self._headers(), params=params, timeout=120)
        if not response.ok:
            raise RuntimeError(
                f"Graph GET failed\nURL: {response.url}\nStatus: {response.status_code}\nBody: {response.text}"
            )
        return response.json()


    def _get_stream(self, url: str):
        session = requests.Session()
        
        # Retry up to 4 times with exponential backoff: 2s, 4s, 8s, 16s
        retry = Retry(
            total=4,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        
        response = session.get(
            url,
            headers=self._headers(),
            stream=True,
            timeout=300
        )
        response.raise_for_status()
        return response
    
    # def _get_stream(self, url: str) -> requests.Response:
    #     response = requests.get(url, headers=self._headers(), stream=True, timeout=300)
    #     if not response.ok:
    #         raise RuntimeError(
    #             f"Graph download failed\nURL: {response.url}\nStatus: {response.status_code}\nBody: {response.text}"
    #         )
    #     return response
    
    def list_root_children(self) -> list[dict[str, Any]]:
        url = f"{GRAPH_BASE}/drives/{self.drive_id}/root/children"

        items : list[dict[str, Any]] = []
        while url:
            payload = self._get_json(url)
            items.extend(payload.get("value", []))
            url = payload.get("@odata.nextLink")
        return items

    def list_children_by_path(self, folder_path: str) -> list[dict[str, Any]]:
        """
        Example folder_path:
        Finance Reports/Cashier Reports
        Sales&Order Reports/Galleria/Sales Reports
        """
        encoded_path = quote(folder_path.strip("/"))
        url = f"{GRAPH_BASE}/drives/{self.drive_id}/root:/{encoded_path}:/children"

        items: list[dict[str, Any]] = []
        while url:
            payload = self._get_json(url)
            items.extend(payload.get("value", []))
            url = payload.get("@odata.nextLink")
        return items

    def list_children_by_item_id(self, item_id: str) -> list[dict[str, Any]]:
        url = f"{GRAPH_BASE}/drives/{self.drive_id}/items/{item_id}/children"

        items: list[dict[str, Any]] = []
        while url:
            payload = self._get_json(url)
            items.extend(payload.get("value", []))
            url = payload.get("@odata.nextLink")
        return items

    def download_file_by_item_id(self, item_id: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{GRAPH_BASE}/drives/{self.drive_id}/items/{item_id}/content"
        response = self._get_stream(url)

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    f.write(chunk)

    @staticmethod
    def is_folder(item: dict[str, Any]) -> bool:
        return "folder" in item

    @staticmethod
    def is_file(item: dict[str, Any]) -> bool:
        return "file" in item

    @staticmethod
    def get_name(item: dict[str, Any]) -> str:
        return item.get("name", "")

    @staticmethod
    def get_item_id(item: dict[str, Any]) -> str:
        return item["id"]

    @staticmethod
    def get_last_modified(item: dict[str, Any]):
        # Example: 2026-03-16T09:15:44Z
        value = item.get("lastModifiedDateTime")
        if not value:
            return None
        return parsedate_to_datetime(
            parsedate_to_datetime(value).strftime("%a, %d %b %Y %H:%M:%S GMT")
        )

    @staticmethod
    def get_size(item: dict[str, Any]) -> int:
        return int(item.get("size", 0))