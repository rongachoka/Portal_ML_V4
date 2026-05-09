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
    start_date       : "YYYY-MM" — month the ad started running (from Meta)
    end_date         : "YYYY-MM" or None (None = still active)
    featured_brands  : brands shown in this ad, exactly as in KB 'Brand' column
                     used as fallback when no specific products are listed
                     leave as [] for category-wide promos
    featured_products: specific KB product names shown in this ad
                     exactly as in KB 'Name' column
                     leave as [] if the whole brand is on promo (not specific products)
    category_scope   : populate only for store-wide / category-wide promos
                     e.g. "Skincare" for "10% off all Skincare"
                     leave as None for brand-specific or product-specific ads
    aliases          : list of old names this ad was known by
                     if Meta renames an ad, add the old name here
                     the spend loader will match on aliases too — nothing breaks

AD TYPE GUIDE:
  Specific products   -> featured_brands + featured_products populated, category_scope = None
  Brand-wide promo    -> featured_brands populated, featured_products = [],  category_scope = None
  Category-wide promo -> featured_brands = [],      featured_products = [],  category_scope = "Category"

HOW TO ADD A NEW AD:
  1. In ad_name_map.py:  add  "respond_io_ad_id": "Exact Ad Name"
     (copy the Ad ID from the Respond.io contacts export,
      copy the Ad name exactly as it appears in Meta Ads Manager)
  2. Here: add a new block using that same ad name as the key
  3. Set start_date from Meta Ads Manager (YYYY-MM)
  4. Add featured_brands and featured_products where known
  5. Run enrich_ad_performance.py to regenerate the performance tables

  aliases: only needed if Meta renames an existing ad after launch.

SPEND DATA:
  Spend is NOT stored here. Drop Meta Ads Manager CSV exports into:
    data/01_raw/meta_spend/
  The pipeline reads them at runtime and joins on ad name.
  Dates are YYYY-MM — the pipeline treats them as the 1st of that month.

USD -> KES CONVERSION:
  Set USD_TO_KES below. Update it when the rate changes materially (>2%).
  All USD spend from Meta will be converted to KES using this rate.
"""

# TEMPLATE (copy-paste for new ads)
# -----------------------------------------------------------------------
# "{Ad Name Here}":
#     {
#         "name":              "{Ad Name Here}",
#         "campaign_name":     "{Campaign Name from Meta Ads Manager}",
#         "start_date":        "YYYY-MM",
#         "end_date":          "YYYY-MM",  # or None if still active
#         "featured_brands":   ["{Brand exactly as in KB}"],
#         "featured_products": [
#             "{Exact Product Name from KB}",
#         ],
#         "category_scope":    None,  # or "Skincare" / "Supplements" etc
#         "aliases":           [],
#     },
# -----------------------------------------------------------------------

# ---- EXCHANGE RATE -------------------------------------------------------
# Update this when the rate shifts materially (>2%).
USD_TO_KES: float = 129.5   # as of April 2026


# ---- REGISTRY ------------------------------------------------------------
AD_REGISTRY: dict[str, dict] = {

    # ── MARCH 2026 ────────────────────────────────────────────────────────

    "Owala Bottles": {
        "name":              "Owala Bottles",
        "campaign_name":     "March Retargeting Sales",
        "ad_id":             None,
        "start_date":        "2026-03",
        "end_date":          None,
        "featured_brands":   ["Owala"],
        "featured_products": [
            "Owala Bottle",
            "Starbucks Bear Cup",
        ],
        "category_scope":    None,
        "aliases":           [],
    },

    "Easter Skincare Sale": {
        "name":              "Easter Skincare Sale",
        "campaign_name":     "March Retargeting Sales",
        "ad_id":             None,
        "start_date":        "2026-03",
        "end_date":          None,
        "featured_brands":   [],
        "featured_products": [],
        "category_scope":    "Skincare",
        "aliases":           [],
    },

    # "Vaseline": {
    #     "name":              "Vaseline",
    #     "campaign_name":     "March Retargeting Sales",
    #     "ad_id":             None,
    #     "start_date":        "2026-03",
    #     "end_date":          None,
    #     "featured_brands":   ["Vaseline"],
    #     "featured_products": [
    #         "Vaseline Gluta-Vitamin Serum",
    #         "Vaseline Gluta-Vitamin Serum Burst Lotion",
    #     ],
    #     "category_scope":    None,
    #     "aliases":           [],
    # },

    "LRP Effaclar Medicated": {
        "name":              "LRP Effaclar Medicated",
        "campaign_name":     "March Retargeting Sales",
        "ad_id":             None,
        "start_date":        "2026-03",
        "end_date":          None,
        "featured_brands":   ["La Roche-Posay"],
        "featured_products": [
            "La Roche Posay Effaclar Medicated Gel Cleanser - 200ml",
        ],
        "category_scope":    None,
        "aliases":           [],
    },

    "Olay Body Lotions": {
        "name":              "Olay Body Lotions",
        "campaign_name":     "March 18th Engagement",
        "ad_id":             None,
        "start_date":        "2026-03",
        "end_date":          None,
        "featured_brands":   ["Olay"],
        "featured_products": [
            "Olay Age Defying with Niacinamide Serum 502ml",
        ],
        "category_scope":    None,
        "aliases":           [],
    },

    "Anti-aging routine": {
        "name":              "Anti-aging routine",
        "campaign_name":     "March 18th Engagement",
        "ad_id":             None,
        "start_date":        "2026-03",
        "end_date":          None,
        "featured_brands":   ["CeraVe"],
        "featured_products": [
            "CeraVe Hydrating Cream to foam Cleanser 236ml",
            "CeraVe Skin Renewing Vitamin C Serum With Hyaluronic Acid 1 Oz",
            "CeraVe Skin Renewing Gel Oil",
            "CeraVe Am Facial Moisturizing Lotion SPF 30",
            "CeraVe Skin Renewing Retinol Serum 30ml",
            "CeraVe Skin Renewing Eye Cream",
            "CeraVe Pm Facial Lotion 60ml",
        ],
        "category_scope":    None,
        "aliases":           [],
    },


    # ── APRIL 2026 ────────────────────────────────────────────────────────

    "Vaseline Lotions": {
        "name":              "Vaseline Lotions",
        "campaign_name":     "Retargeting", 
        "ad_id":             "6981427798440",       
        "start_date":        "2026-04",
        "end_date":          None,
        "featured_brands":   ["Vaseline"],
        "featured_products": [
            "Vaseline Intensive Care Body Oil"
        ],          # add specific products when known
        "category_scope":    None,
        "aliases":           [],
    },

    "Vaseline Lotions 2": {
        "name":              "Vaseline Lotions",
        "campaign_name":     "Retargeting", 
        "ad_id":             "6982634246240",       
        "start_date":        "2026-04",
        "end_date":          None,
        "featured_brands":   ["Vaseline"],
        "featured_products": [
            "Vaseline Intensive Care Body Oil"
        ],          # add specific products when known
        "category_scope":    None,
        "aliases":           [],
    },

    "Vaseline": {
        "name":              "Vaseline",
        "campaign_name":     "Retargeting",
        "ad_id":             "6971216835440",         
        "start_date":        "2026-04",
        "end_date":          None,
        "featured_brands":   ["Vaseline"],
        "featured_products": [
            "Vaseline Gluta-Vitamin Serum",
            "Vaseline Gluta-Vitamin Serum Burst Lotion",
        ],          # add specific products when known
        "category_scope":    None,
        "aliases":           [],
    },


    "New in Stock - Copy": {
        "name":              "New in Stock - Copy",
        "campaign_name":     "April Sales 2 Campaign - Copy Working",
        "ad_id":             "6971216835440",         
        "start_date":        "2026-04",
        "end_date":          None,
        "featured_brands":   ["Owala"],
        "featured_products": [
            "Owala Cups",
            "Starbucks Bear Cup",
        ],          # add specific products when known
        "category_scope":    None,
        "aliases":           [],
    },


    "Cerave dry skin routine": {
        "name":              "Cerave dry skin routine",
        "campaign_name":     "April Engagement Campaign Ad Set",
        "ad_id":             "6981368972640",         
        "start_date":        "2026-04",
        "end_date":          None,
        "featured_brands":   ["Cerave"],
        "featured_products": [
            "Cerave Hydrating Cleanser 473ml",
            "Cerave Moisturising Cream 340g",
            "Cerave Advanced Repair Ointment"
        ],          # add specific products when known
        "category_scope":    None,
        "aliases":           [],
    },

    "Olay Retinol Eyes": {
        "name":              "Olay Retinol Eyes",
        "campaign_name":     "April Engagement Campaign Ad Set",
        "ad_id":             "6976748055040",         
        "start_date":        "2026-04",
        "end_date":          None,
        "featured_brands":   ["Olay"],
        "featured_products": [
            "Olay Retinol 24 Night Eye Cream"
        ],          # add specific products when known
        "category_scope":    None,
        "aliases":           [],
    },

    "New in Stock": {
        "name":              "New in Stock",
        "campaign_name":     "April Sales 2 Campaign",
        "ad_id":             "6981423287040",         
        "start_date":        "2026-04",
        "end_date":          None,
        "featured_brands":   ["Cetaphil", "Old Spice", "Crest", "Sure", "Life Extension", "Cottonelle"],
        "featured_products": [
            "Old Spice Swagger Body Wash",
            "Cetaphil Acne Mattifying Moisturizer",
            "Cetaphil Moisturizer SPF 35",
            "Crest Complete Toothpaste",
            "Crest Advanced Toothpaste",
            "Sure Thermal Instant Heat Pack",
            "Life Extension High Potency Optimized Folate",
            "Cottonelle Flushable Wipes"
        ],          # add specific products when known
        "category_scope":    None,
        "aliases":           [],
    },

    "Neutrogena Products": {
        "name":              "Neutrogena Products",
        "campaign_name":     "April Sales 2 Campaign - Copy Working",
        "ad_id":             "6982634246640",         
        "start_date":        "2026-04",
        "end_date":          None,
        "featured_brands":   ["Neutrogena"],
        "featured_products": [
            "Neutrogena Hydro Boost Hydrating Cleanser with Hyaluronic Acid",
            "Neutrogena Hydro Boost No-Rinse Cleansing Water",
            "Neutrogena Alcohol-Free Toner",
            "Neutrogena Blackhead Eliminating 0.5% Salicylic Acid Cleansing Toner",
            "Neutrogena Hydro Boost Illuminating Serum"
        ],          # add specific products when known
        "category_scope":    None,
        "aliases":           [],
    },


}


# ---- RUNTIME HELPERS (used by enrich_ad_performance.py) ------------------

def get_ad_by_name(name: str) -> dict | None:
    """
    Look up an ad by its exact name or any of its aliases.
    Returns the registry entry or None if not found.
    """
    entry = AD_REGISTRY.get(name)
    if entry:
        return entry

    name_lower = str(name).strip().lower()
    for ad_name, meta in AD_REGISTRY.items():
        all_names = [ad_name.lower()] + [a.lower() for a in meta.get("aliases", [])]
        if name_lower in all_names:
            return meta

    return None


def build_alias_index() -> dict[str, str]:
    """
    Returns flat dict: every known name/alias -> canonical ad_name.
    Used by the spend loader to normalise Meta export rows in one vectorised pass.
    """
    index = {}
    for ad_name, meta in AD_REGISTRY.items():
        index[ad_name.strip().lower()] = ad_name
        for alias in meta.get("aliases", []):
            index[alias.strip().lower()] = ad_name
    return index


def parse_ad_date(date_str: str | None):
    """
    Parses YYYY-MM registry dates into a Timestamp (1st of that month).
    Returns pd.NaT if the value is None or unparseable.
    Call this wherever the pipeline reads start_date or end_date.
    """
    import pandas as pd
    if not date_str:
        return pd.NaT
    try:
        return pd.to_datetime(str(date_str) + "-01")
    except Exception:
        return pd.NaT