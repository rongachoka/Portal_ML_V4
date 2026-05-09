"""
ad_name_map.py
==============
Maps Meta Ad IDs to human-readable campaign names.

Location: Portal_ML_V4/src/config/ad_name_map.py

HOW TO UPDATE:
  1. Get the Ad ID from Meta Ads Manager (or from the Respond.io ad attribution export)
  2. Add a new line:  "AD_ID_HERE": "Descriptive Campaign Name",
  3. Save — the pipeline picks it up automatically on next run.

The Ad ID is a numeric string (no spaces). The name should be descriptive enough
that someone reading a PowerBI report understands what the ad was promoting.
"""

AD_NAME_MAP = {
    # ── La Roche-Posay ───────────────────────────────────────────
    "6823164424240":        "La Roche BOGOF",
    "6921500406040":        "LRP Bundles / Products",
    "6943721123640":        "LRP Anthelios",
    "6935232756840":        "LRP Anthelios Jan",
    "6948905615040":        "LRP Giveaway",
    "6882975829240":        "10% Off La Roche Anthelios",
    "6934756495640":        "LRP Acne Routine",
    "6921591641240":        "LRP Anthelios Sale",
    "6921585634640":        "LRP Effaclar Bundle",
    "6921542846640":        "LRP Lipikar Bundle",

    # ── CeraVe ───────────────────────────────────────────────────
    "6941964717640":        "CeraVe Acne Routine",

    # ── CosRx ────────────────────────────────────────────────────
    "6817550498240":        "CosRx Centella",

    # ── Supplements ──────────────────────────────────────────────
    "6941964715840":        "Ashwaghanda",
    "6941951457840":        "Ashwaghanda",
    "6937045949440":        "Magnesium Glycinate + Ashwaghanda",
    "6937053336840":        "Magnesium + Ashwaghanda",
    "6941964716840":        "Magnesium + Ashwaghanda",
    "6936990612240":        "3 Supplements Every Woman Should Take",
    "6941964718040":        "Hormonal Care",
    "6941964715640":        "Supplements Reel",
    "6948906394440":        "Supplements for Glowing Skin",

    # ── Skincare (General) ────────────────────────────────────────
    "6826292826840":        "10% Off Categories",
    "6955912833840":        "10% Off Skincare Products",
    "6945414530240":        "Skincare Offers",
    "6945406805440":        "Skincare Reel",
    "6948905752240":        "Moisturizers for Oily Skin",
    "6948903906440":        "Anua + Mandelic Acid",
    "6949654359440":        "Reedle Shot",

    # ── Vichy ────────────────────────────────────────────────────
    "6949654216840":        "Vichy Dandruff",
    "6955681219240":        "Vichy Dandruff Sales",
    "6915084614640":        "The Vichy Aqualia Thermal Night Spa",

    # ── Sales & Bundles ──────────────────────────────────────────
    "6948909029040":        "Bundles",
    "120221326858920683":   "Clearance Sale",
    "120239692754220696":   "For The Girlies",
    "120239692341380696":   "This is your Sign to Stock Up",
    "6942365246240":        "This is your Sign to Stock Up",
    "6945471200240":        "Unsure what to gift this Valentine's?",
    "6943617269440":        "Valentines Offers",
    "6945403929440":        "Skincare Sale - Valentines",
    "6941964716240":        "February Campaign 2 Whatsapp/ Instagram",

    # ── Lip & Beauty ─────────────────────────────────────────────
    "6943706631440":        "Lip Products",
    "6943627224640":        "EOS",
    "6943617269240":        "BBW",

    # ── Men's ─────────────────────────────────────────────────────
    "6955913814640":        "Male Adult Acne",

   # ── March ─────────────────────────────────────────────────────
   "6959681615440" :       "March Supplement Picks",
   "6948903906240" :       "Mid March Sales Campaign",
   "6948909030840" :       "Mid March Engagement Campaign",
   "6955681219040" :       "March Retargeting Sales",
   "6959689452440" :       "March Engagement Campaign 2 Campaign",
   "6962426440440" :       "L-Glutamine",
   "6964772065040" :       "Instagram Post: New In. Trending. You Need This",
   "6962469215240" :       "Mid March New Stock",
   "6962468802840" :       "10% Off Ordinary Products",
   "6962470340440" :       "Byoma Products",
   "6959682773240" :       "LRP Effaclar Medicated",
   "6959684105040" :       "Body Hyperpigmentation",
   "6962426450840" :       "March 18th Engagement Campaign"

   



}