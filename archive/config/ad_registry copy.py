"""
ad_registry.py
==============
Source of truth for every Meta ad Portal has run.

Keyed by ad name (string) — identical across both systems:
  - Meta Ads Manager exports ('Ad name' column)
  - AD_NAME_MAP values (Respond.io contact export 'Ad ID' → name)
  Both use the same string, so the join is exact with no alias resolution needed.

Each entry holds:
    name             : mirrors the key — kept for convenience when iterating
    campaign_name    : the parent campaign in Meta Ads Manager
                     used for campaign-level rollup in the combined report
    start_date       : "YYYY-MM-DD" — when the ad started running (from Meta)
    end_date         : "YYYY-MM-DD" or None (None = still active)
    featured_brands  : brands shown in this ad, exactly as in KB 'Brand' column
                     used as fallback when no specific products are listed
    featured_products: specific KB product names shown in this ad
                     exactly as in KB 'Name' column
                     leave as [] if you only know the brand
    featured_prices  : dict of {product name: price} for any featured products
    category_scope   : if the ad only features products from one category, specify it here
                     used to filter POS matches when product-level matching fails
    aliases          : list of old names this ad was known by
                     if Meta renames an ad, add the old name here
                     the spend loader will match on aliases too — nothing breaks

HOW TO ADD A NEW AD:
  1. In ad_name_map.py: add  "respond_io_ad_id": "Exact Ad Name"
     (copy the Ad ID from the Respond.io contacts export,
      copy the Ad name exactly as it appears in Meta Ads Manager)
  2. Here: add a new block using that same ad name as the key
  3. Set start_date from Meta Ads Manager
  4. Add featured_brands and featured_products where known
  5. Run enrich_ad_performance.py to regenerate the performance tables

  aliases: only needed if Meta renames an existing ad after launch.
           Under normal circumstances you won't need to populate this.

SPEND DATA:
  Spend is NOT stored here. Drop Meta Ads Manager CSV exports into:
    data/01_raw/meta_spend/
  The pipeline reads them at runtime and joins on ad name.

USD → KES CONVERSION:
  Set USD_TO_KES below. Update it when the rate changes materially (>2%).
  All USD spend from Meta will be converted to KES using this rate.
"""


"""
TEMPLATE FOR NEW ADS (copy-paste and fill in the blanks):

"{Ad Name Here}": 
    {
        "name":            "{Ad Name Here}",
        "campaign_name":   "{Campaign Name from Meta Ads Manager}",
        "start_date":      "YYYY-MM-DD",
        "end_date":        "None or YYYY-MM-DD",
        "featured_brands": ["{Brand Name if known}"],         
        "featured_products": 
            [
                "{Exact Product Name from KB if known}",
            ],
        "featured_prices": 
            {
                "{Exact Product Name from KB if known}": {price_in_KES},
            },
        "category_scope":    None or "{Category Name ifs store wide e.g. 
                            'Skincare' if the ad is e.g 10% off Skincare 
                            products}",
        "aliases": [],
    },

"""

# ── EXCHANGE RATE ─────────────────────────────────────────────────────────────
# Update this when the rate shifts materially.
USD_TO_KES: float = 129.5   # as of April 2026



AD_REGISTRY: dict[str, dict] = {
    # ── March Ads ────────────────────────────────────────────────────────

    # ── March Retargeting Sales ──────────────────────────────────────────

    "Owala Bottles": 
    {
        "name":            "Owala Bottles",
        "ad_id":             None,
        "campaign_name":   "March Retargeting Sales",
        "start_date":      "2026-03-01",
        "end_date":        None,
        "featured_brands": ["Owala"],         # not in KB — used for description matching only
        "featured_products": [
            "Owala Bottle",                   # match against POS Description tokens
            "Starbucks Bear Cup",             # second product in same ad
        ],
        "featured_prices": {                  # for reference / price validation
            "Owala Bottle": 2650,
            "Starbucks Bear Cup": 2000,
        },
        "category_scope": None,
        "aliases": [],
    },

    "Easter Skincare Sale": 
    {
        "name":            "Easter Skincare Sale",
        "ad_id":             None,
        "campaign_name":   "March Retargeting Sales",
        "start_date":      "2026-03-01",
        "end_date":        None,
        "featured_brands": [],         
        "featured_products": [],
        "featured_prices": {},
        "category_scope":    "Skincare",
        "aliases": [],
    },

    "Vaseline": 
    {
        "name":            "Vaseline",
        "ad_id":             None,
        "campaign_name":   "March Retargeting Sales",
        "start_date":      "2026-03-01",
        "end_date":        None,
        "featured_brands": ["Vaseline"],         
        "featured_products": 
            [
            "Vaseline Gluta-Vitamin Serum",           
            "Vaseline Gluta-Vitamin Serum Burst Lotion",
            ],
        "featured_prices": 
            {
                "Vaseline Gluta-Vitamin Serum":              1500,
                "Vaseline Gluta-Vitamin Serum Burst Lotion": 1650,
            },
        "category_scope":    None,
        "aliases": [],
    },


    "LRP Effaclar Medicated": 
    {
        "name":            "LRP Effaclar Medicated",
        "ad_id":             None,
        "campaign_name":   "March Retargeting Sales",
        "start_date":      "2026-03-01",
        "end_date":        None,
        "featured_brands": ["La Roche Posay"],         
        "featured_products": 
            [
            "La Roche Posay Effaclar Medicated Gel Cleanser - 200ml", 
            ],
        "featured_prices": 
            {
                "La Roche Posay Effaclar Medicated Gel Cleanser - 200ml": 2500,
                
            },
        "category_scope":    None,
        "aliases": [],
    },

    "Olay Body Lotions": 
    {
        "name":            "Olay Body Lotions",
        "ad_id":             None,
        "campaign_name":   "March 18th Engagement",
        "start_date":      "2026-03-01",
        "end_date":        None,
        "featured_brands": ["Olay"],         
        "featured_products": 
            [
            "Olay Age Defying with Niacinamide Serum 502ml", 
            ],
        "featured_prices": 
            {
                "Olay Age Defying with Niacinamide Serum 502ml": 3500,
            },
        "category_scope":    None,
        "aliases": [],
    },

    "Anti-aging routine": 
    {
        "name":            "Olay Body Lotions",
        "ad_id":             None,
        "campaign_name":   "March 18th Engagement",
        "start_date":      "2026-03-01",
        "end_date":        None,
        "featured_brands": ["CeraVe"],         
        "featured_products": 
            [
            "CeraVe Hydrating Cream to foam Cleanser 236ml",
            "CeraVe Skin Renewing Vitamin C Serum With Hyaluronic Acid 1 Oz",
            "CeraVe Skin Renewing Gel Oil",
            "CeraVe Am Facial Moisturizing Lotion SPF 30",
            "CeraVe Skin Renewing Retinol Serum 30ml",
            "CeraVe Skin Renewing Eye Cream",
            "CeraVe Pm Facial Lotion 60ml"

            ],
        "featured_prices": 
            {
                "CeraVe Hydrating Cream to foam Cleanser 236ml": 3950,
                "CeraVe Skin Renewing Vitamin C Serum With Hyaluronic Acid 1 Oz": 6500,
                "CeraVe Skin Renewing Gel Oil": 3000,
                "CeraVe Am Facial Moisturizing Lotion SPF 30": 3000,
                "CeraVe Skin Renewing Retinol Serum 30ml": 3000,
                "CeraVe Skin Renewing Eye Cream": 3000,
                "CeraVe Pm Facial Lotion 60ml": 3000

            },
        "category_scope":    None,
        "aliases": [],
    },

    # -------------------------------------------------------------------------
    # APRIL ADS
    # -------------------------------------------------------------------------
    
    "Vaseline Lotions": 
    {
        "name":            "Vaseline Lotions",
        "campaign_name":   "Retargeting",
        "ad_id":             None,
        "start_date":      "2026-03-01",
        "end_date":        None,
        "featured_brands": [""],         
        "featured_products": 
            [
            ],
        "featured_prices": 
            {
            
            },
        "category_scope":    None,
        "aliases": [],
    },
    

    
    


}


# ── RUNTIME HELPERS (used by enrich_ad_performance.py) ────────────────────────

def get_ad_by_name(name: str) -> dict | None:
    """
    Look up an ad by its exact name or any of its aliases.
    Returns the registry entry or None if not found.
    """
    entry = AD_REGISTRY.get(name)
    if entry:
        return entry

    # Alias fallback — handles renamed ads
    name_lower = str(name).strip().lower()
    for ad_name, meta in AD_REGISTRY.items():
        all_names = [ad_name.lower()] + [a.lower() for a in meta.get("aliases", [])]
        if name_lower in all_names:
            return meta

    return None


def build_alias_index() -> dict[str, str]:
    """
    Returns flat dict: every known name/alias → canonical ad_name.
    Used by the spend loader to normalise Meta export rows in one vectorised pass.
    e.g. {"la roche bogof": "La Roche BOGOF", "lrp bundles": "LRP Bundles / Products", ...}
    """
    index = {}
    for ad_name, meta in AD_REGISTRY.items():
        index[ad_name.strip().lower()] = ad_name
        for alias in meta.get("aliases", []):
            index[alias.strip().lower()] = ad_name
    return index