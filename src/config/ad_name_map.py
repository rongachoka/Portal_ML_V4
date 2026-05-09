"""
ad_name_map.py
==============
Maps Respond.io Ad IDs (numeric strings) to the canonical ad name.

The ad name is the stable join key that bridges:
  Respond.io  (knows Ad ID, known name)
  Meta export (knows Ad name, not ID)
  AD_REGISTRY (keyed by ad name)

HOW TO ADD A NEW AD:
  1. Run a new Respond.io contacts export after the campaign launches.
  2. Find the new Ad ID in the 'Ad ID' column.
  3. Add one line here:  "new_id_here": "Exact Ad Name From Meta",
  4. Add a matching entry in ad_registry.py using the same ad name string.
  That's it. No other files need to change.

IMPORTANT: The ad name string here must exactly match:
  - The 'Ad name' column in your Meta Ads Manager export
  - The key in AD_REGISTRY
  If Meta renames an ad, add the old name to 'aliases' in AD_REGISTRY
  (don't change this file — the ID→name mapping is permanent).
"""

AD_NAME_MAP: dict[str, str] = {

    # ── LA ROCHE-POSAY ───────────────────────────────────────────────────────
    "6823164424240":      "La Roche BOGOF",
    "6921500406040":      "LRP Bundles / Products",
    "6943721123640":      "LRP Anthelios",
    "6935232756840":      "LRP Anthelios Jan",
    "6948905615040":      "LRP Giveaway",

    # ── COSRX ────────────────────────────────────────────────────────────────
    "6817550498240":      "CosRx Centella",

    # ── SUPPLEMENTS ──────────────────────────────────────────────────────────
    "6941964715840":      "Ashwaghanda",
    "6937045949440":      "Magnesium Glycinate + Ashwaghanda",
    "6937053336840":      "Magnesium + Ashwaghanda",
    "6936990612240":      "3 Supplements Every Woman Should Take",
    "6941964718040":      "Hormonal Care",
    "6941964715640":      "Supplements Reel",
    "6948906394440":      "Supplements for Glowing Skin",
    "120239692341380696": "This is your Sign to Stock Up",

    # ── CERAVE ───────────────────────────────────────────────────────────────
    "6941964717640":      "CeraVe Acne Routine",

    # ── VICHY ────────────────────────────────────────────────────────────────
    "6949654216840":      "Vichy Dandruff",

    # ── ANUA ─────────────────────────────────────────────────────────────────
    "6948903906440":      "Anua + Mandelic Acid",

    # ── LIP / PERSONAL CARE ──────────────────────────────────────────────────
    "6943706631440":      "Lip Products",
    "6943627224640":      "EOS",
    "6943617269240":      "BBW",

    # ── REEDLE SHOT ──────────────────────────────────────────────────────────
    "6949654359440":      "Reedle Shot",

    # ── PROMOTIONAL / MULTI-BRAND ────────────────────────────────────────────
    "6948909029040":      "Bundles",
    "6826292826840":      "10% Off Categories",
    "120221326858920683": "Clearance Sale",
    "120239692754220696": "For The Girlies",
    "6945414530240":      "Skincare Offers",
    "6945471200240":      "Unsure what to gift this Valentine's?",
    "6948905752240":      "Moisturizers for Oily Skin",
    "6870933759840" :       "The ordinary glycolic 13.8.2025",

    # ── March ─────────────────────────────────────────────────────
    "6959681615440" :       "March Supplement Picks",
    "6948903906240" :       "Mid March Sales Campaign",
    "6948909030840" :       "Mid March Engagement Campaign",
    "6955681219040" :       "March Retargeting Sales",
    "6959689452440" :       "March Engagement Campaign 2 Campaign",
    "6962426440440" :       "L-Glutamine",
    "6964772065040" :       "Instagram Post: New In. Trending. You Need This",
    "6971216405840" :       "Instagram Post: Easter just got better for skin",
    "6962469215240" :       "Mid March New Stock",
    "6962468802840" :       "10% Off Ordinary Products",
    "6962470340440" :       "Byoma Products",
    "6959682773240" :       "LRP Effaclar Medicated",
    "6959684105040" :       "Body Hyperpigmentation",
    "6962426450840" :       "March 18th Engagement Campaign",
    "6971216562840" :       "Owala Bottles",
    "6971217420240" :       "Olay Body Lotions",


# ── April ─────────────────────────────────────────────────────
    "6971216835440":        "Vaseline",
    "6971216449440" :       "Easter Skincare sale",
    "6971398195440" :       "Easter Skincare Sale",
    "6971403563240" :       "Easter Skincare Sale - Copy",
    "6971214466040" :       "Easter Sale",
    "6962426450440" :       "L-Gluatamine",
    "6971217664040" :       "Anti-aging routine",
    "6976727987840" :       "Cerave dry skin routine",
    "6976748055040" :       "Olay retinol eyes",
    "6976742186240" :       "LRP Mela B3",
    "6959691095640" :       "10% off all cosrx products",
    "6978150251240" :       "La Roche Posay samples",
    "6934756495640" :       "LRP Acne Routine",
    "6978150250840" :       "Vaseline Body Oils",
    "6909980927840" :       "Magnesium Glycinate",
    "6921592828240" :       "Magnesium Glycinate",
    "6934748029840" :       "Children's Supplements",
    "6904131040040" :       "BN LRP Reel",
    "6981368972640" :       "Cerave Routine",
    "6981414655240" :       "Owala Cups",
    "6893875546040" :       "Medicube Kojic Acid Sale",
    "6981423287040" :       "New in Stock",
    "6982634246840" :       "New in Stock - Copy",
    "6981427798440" :       "Vaseline Lotions",
    "6982634246240" :       "Vaseline Lotions",
    "6982634246640" :       "Neutrogena Products",



}