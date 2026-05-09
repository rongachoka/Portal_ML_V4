"""
brands.py
=========
Master brand reference data for Portal Pharmacy.

Exports:
    BRAND_LIST      — Canonical list of all stocked brand names used for
                      brand detection in chat sessions and POS matching.
    BRAND_ALIASES   — Dict mapping common shorthand/misspellings to the
                      canonical brand name (e.g. "lrp" → "La Roche-Posay").
"""

# src/config/brands.py

# =============================================================================
# MASTER BRAND LIST (Canonical Names)
# =============================================================================
BRAND_LIST = [
    # A
    "ACM", "Accu-Chek", "Acnes", "Action", "Actilife", "Advanced Clinicals", 
    "African Pride", "Alberto", "Alka Seltzer", "Aloe Pura", "Aloedent", 
    "American Dream", "Andolex", "Animal Parade", "Anthisan", 
    "Anua", "Aptamil", "Aqua Oleum", "Aquafresh", "Aquaphor", 
    "Ascoril", "Ashton & Parson's", "Astral", "Aunt Jackie's", "Avalife", 
    "Aveeno", "Avene", "Avent", "Axe Oil", "AZO",  
    # B
    "Baby Shark", "Bach", "Bactigras", "Bath and Body Works", "Batiste", 
    "Beauty Formula", "Beauty of Joseon", "Beechams", "Bells", "Bennetts", 
    "Benylin", "Bepanthen", "Berocca", "Best Immune", "Betadine", "BetterYou", 
    "Bio Oil", "BioGaia", "Bioderma", "Biretix", "Black Girl Sunscreen", 
    "Blistex", "Bonjela", "Bonnisan", "Bronchicum", "Brookethorne Naturals", 
    "Brufen", "Brustan", "Brut", "Brylcreem", "Bubble T", "Bulldog", "Bump Patrol", 
    # C
    "CS Beauty", "CS Medic", "Cadistin", "Calpol", "Cantu", "Carefree", "Careway", 
    "Carmex", "Carnation", "Cartiflex", "Cartil", "Centrum", "CeraVe", "Cetamol", 
    "Cetaphil", "Chupa Chups", "Cipla", "Clairol", "Clear & Simple", 
    "Clearblue", "Coldcap", "Colgate", "Compeed", "Corsodyl", "CosRx", 
    "Covonia", "Cow & Gate", "Cura-Heat", 
    # D
    "Dadacare", "Daktarin", "Dax", "Deep Freeze", "Deep Heat", "Deep Relief", 
    "Delased", "Demelan", "Denman", "Dentogel", "Dettol", "Diarim", "Disney", 
    "Dove", "Dr Organics", "Duofilm", "Durex", 
    # E
    "E45", "EOS", "Eco Styler", "Efferalgan", "Elastoplast", "Emami", "Eno", 
    "Enterobasila", "Enterogermina", "Epimax", "Epimol", "Erovita", "Eugica", 
    "Everysun", "Exevate-MF", "Eucerin", 
    # F
    "Femfresh", "Fenty", "Fino", "First Steps", "Fisherman's Friend", 
    "Fludex", "Flugone", "Forever Living", "Fruit-tella", "Futsil", "Fybogel", 
    # G
    "Garnier", "Gaviscon", "General", "Gentian", "Geratherm", "Gillette", 
    "Gisou", "Glucometer", "Gluconova", "Gofen", "Good Essentials", 
    "Got2b", 
    # H
    "Haliborange", "Halls", "Hanan", "HealthAid", "Hedex", "Heliocare", 
    "Herbigor", "Hipp Organic", "Huggies", "Hyponidd", 
    # I
    "Ibumex", "Infacol", "Innovate", "Isntree",
    # J
    "Jada", "Jamieson", "Johnson's", "Joint Eze",
    # K
    "K3 Pro", "Katya", "Kedley", "Keto Plus", "Kings", "Kly", "Kosas", "Kotex", 
    # L
    "L'Oreal", "La Roche Posay", "Lemsip", "Ligastrap", "Lil-lets", "Liptomil", 
    "Listerine", "Lotta", "Lyclear", 
    # M
    "Maalox", "Mack's", "Madagascar Centella", "Mara Moja", "Marvel", 
    "Masterplast", "Medicare", "Medicott", "Medicube", "Medisure", "Medix", 
    "Medtextile", "Miadi", "Microlife", "Mielle", "Milton", "Mitchum", "Mizani", 
    "Molped", 
    # N
    "NK Lip Gel", "NOW Foods", "Nasosal", "Natures Aid", "NaturesPlus", "Natures Truth",
    "Naturium", "Nebulizer", "Neocell", "Neutrogena", "Niacin - B", "Nicorette", 
    "Nivea", "Nizoral", "Noise X", "Norditalia", "Nuage", "Nurofen", 
    # O
    "O'Keeffe's", "Oatveen", "Oilatum", "Olay", "Olbas", "Omron", 
    "Oral-B", "Oxygen Botanicals", 
    # P
    "Palmer's", "Pampers", "Panadol", "Panoxyl", "Pantocid", "Paula's Choice", 
    "Paw Patrol", "Pears", "Peppa Pig", "Pepto-Bismol", "Ph Care",
    "Pillmate", "Piriton", "Pixi", "Primapore", "ProFoot", "Profertil", 
    "Promimba", "Puremar", "Pyramid", 
    # Q
    "Quest", "Quick Freeze", "Quies", 
    # R
    "Radox", "Rain Argan", "Ranferon", "Reedle Shot", "Refresh", "Regaine", 
    "Relcer", "Rennie", "Replens", "Rhinathiol", "Ricola", 
    # S
    "SMA", "Safe and Sound", "Saferon", "Salonpas", "Sambucol", "Savlon", 
    "Scabees", "Scholl", "Seapuri", "Sebamed", "Selsun Blue", "Sensodyne", 
    "Seven Seas", "Shea Moisture", "Simple", "Skilax", "Skyn", "Snufflebabe", 
    "Sol de Janeiro", "Solvin", "Sterimar", "Strepsils", "Suave", "Sudafed", 
    "Sudocrem", "Summer Fridays","Swiss Energy", 
    # T
    "Tampax", "Tena", "The Body Shop", "The Ordinary", "Thuasne", 
    "Tiger Balm", "Tixylix", "Tommee Tippee", "Topicals", "Tot'hema", "Tropical", 
    # U
    "Trubliss", 
    "Ulgicid", "Ultra Chloraseptic", "Uncover", 
    # V
    "Value Health", "Vaseline", "Veet", "Velvex", "Vichy", "Vicks", 
    "Vitabiotics", "Vital-Pro", 
    # W
    "Wallace", "Webber Naturals", "Wild Earth Botanics", "Wisdom", "Woodwards", 
    # X
    "Xykaa", 
    # Y
    "Yucran - D", 
    # Z
    "Zedcal", "Zelaton", "Zentel"
]

# =============================================================================
# BRAND ALIASES (Mapping Variations -> Canonical)
# =============================================================================
BRAND_ALIASES = {
    # La Roche Posay
    "lrp": "La Roche Posay",
    "la ro": "La Roche Posay",
    "la roche": "La Roche Posay",
    "la roche posay": "La Roche Posay",
    "larocheposay": "La Roche Posay",
    "la roche-posay": "La Roche Posay",
    "laroche": "La Roche Posay",
    "posay": "La Roche Posay",
    "roche": "La Roche Posay",
    "effaclar": "La Roche Posay",
    "anthelios": "La Roche Posay",
    "lipikar": "La Roche Posay",

    # L'Oreal
    "loreal": "L'Oreal",
    "l'oreal": "L'Oreal",
    "l oreal": "L'Oreal",
    "Lâ€™Oreal": "L'Oreal",

    # The Body Shop
    "body shop": "The Body Shop",
    "the body shop": "The Body Shop",

    # Palmer's
    "palmers": "Palmer's",
    "palmer": "Palmer's",

    # Neutrogena
    "neut": "Neutrogena",

    # Dr Organic
    "dr. organic": "Dr Organics",
    "dr organics": "Dr Organics",
    "dr.organic": "Dr Organics",

    # Johnson's
    "johnsons": "Johnson's",
    "johnson's baby": "Johnson's",

    # Shea Moisture
    "sheamoisture": "Shea Moisture",
    "shea moisture": "Shea Moisture",
    "shea/m": "Shea Moisture",

    # Mizani
    "mizan": "Mizani",

    # Natures Truth
    "\bNT\b": "Natures Truth",

    # Oxygen
    "oxygen": "Oxygen Botanicals",
    "oxygen botanicals": "Oxygen Botanicals",

    # Philips / Avent
    "avent": "Avent",
    "philips avent": "Avent",

    # Bio-Oil
    "bio oil": "Bio Oil",
    "bio-oil": "Bio Oil",

    # Black Girl Sunscreen
    "black girl": "Black Girl Sunscreen",
    "bgs": "Black Girl Sunscreen",

    # Clean & Clear
    "clean and clear": "Clean & Clear",

    # E45
    "e 45": "E45",
    "e45": "E45",

    # Paula's Choice
    "paulas choice": "Paula's Choice",
    "paula choice": "Paula's Choice",

    # Oral-B
    "oral b": "Oral-B",
    "oral-b": "Oral-B",
    "oralb": "Oral-B",

    # NOW Foods
    "now foods": "NOW Foods",

    # Forever Living
    "forever": "Forever Living",
    "forever aloe": "Forever Living",

    # Hip / Hipp
    "hipp": "Hipp Organic",

    # The Ordinary
    "ordinary": "The Ordinary",
    "the ordinary": "The Ordinary",
    "ordinary niacinamide" : "The Ordinary",

    # Cerave
    "cereva": "CeraVe",
    "cereve": "CeraVe",
    "cera ve": "CeraVe",

    # Sport supplies
    "SS" : "Sport Supplies",

    # Deep Heat/Freeze
    "deep heat": "Deep Heat",
    "deep freeze": "Deep Freeze",
    
    # Generic catch-alls
    "general": "General",
    "personal care": "General"
}

# O(1) Lookup Map
BRAND_KEYMAP = {b.lower(): b for b in BRAND_LIST}