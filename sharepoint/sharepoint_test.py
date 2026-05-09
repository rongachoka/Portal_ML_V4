# from Portal_ML_V4.sharepoint.sharepoint_auth import graph_get

# HOSTNAME = "portalpharmacy.sharepoint.com"
# SITE_PATH = "/sites/PortalPharmacyLimited"

# url = f"https://graph.microsoft.com/v1.0/sites/{HOSTNAME}:{SITE_PATH}"
# site = graph_get(url)

# print(site)
# print("Site ID: ", site["id"])
# print("Site Name: ", site["name"])

from Portal_ML_V4.sharepoint.sharepoint_client import SharePointClient

DRIVE_ID = "b!whL3rPzNh0-7qRe5yrHoftRvAUJj1gFFvAMiiq_bJDX64liSkv0CSZtdTu6bqccj"

client = SharePointClient(drive_id=DRIVE_ID)

items = client.list_children_by_item_id("root")

for item in items:
    kind = "Folder" if client.is_folder(item) else "File"
    print(f"{kind}: {item["name"]}")