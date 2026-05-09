from Portal_ML_V4.sharepoint.sharepoint_auth import graph_get

## ----------------Getting Site ID
# url = "https://graph.microsoft.com/v1.0/sites?search=PortalPharmacy"

# result = graph_get(url)

# for site in result["value"]:
#     print("Name: ", site["name"])
#     print("Site ID: ", site["id"])
#     print("Web URL: ", site["webUrl"])
#     print("-" * 40)


## ----------------List Document Libraries (Drives) in the Site
SITE_ID = "portalpharmacy.sharepoint.com,acf712c2-cdfc-4f87-bba9-17b9cab1e87e,42016fd4-d663-4501-bc03-228aafdb2435"

url = f"https://graph.microsoft.com/v1.0/sites/{SITE_ID}/drives"

drives = graph_get(url)

for d in drives["value"]:
    print("Drive Name: ", d["name"])
    print("Drive ID: ", d["id"])
    print()