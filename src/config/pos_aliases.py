# src/config/pos_aliases.py
# =============================================================================
# POS TERM ALIASES, TYPO CORRECTIONS & BRAND ALIASES
# =============================================================================
# Central source of truth for all POS abbreviations, typos and brand shorthands.
# Imported by:
#   - build_product_map.py
#   - enrich_attribution_products.py
#   - analytics.py (brand detection)
#   - any other matching/enrichment script
#
# To add a new entry:
#   1. Find the right section below
#   2. Add your entry
#   3. Run build_product_map.py --force to reprocess all descriptions
# =============================================================================


# =============================================================================
# TERM ALIASES
# Abbreviations and typos that appear in POS product descriptions.
# Keys are matched case-insensitively using word boundaries.
# =============================================================================

TERM_ALIASES: dict[str, str] = {

    # ── Form factors ──────────────────────────────────────────────────────────
    "TABS":     "TABLETS",
    "TAB":      "TABLET",
    "VIT":     "VITAMIN",
    "CAPS":     "CAPSULES",
    "CAP":      "CAPSULE",
    "SUPS":     "SUSPENSION",
    "SUSP":     "SUSPENSION",
    "SYR":      "SYRUP",
    "SOLN":     "SOLUTION",
    "SOLUTION":  "SOLUTION",
    "SOL":      "SOLUTION",
    "CRM":      "CREAM",
    "CRE":      "CREAM",
    "INJ":      "INJECTION",
    "OINT":     "OINTMENT",
    "LOT":      "LOTION",
    "LTN":      "LOTION",
    "EFF":      "EFFERVESCENT",
    "SOOTH":     "SOOTHING",
    "SHAM":      "SHAMPOO",
    "SHAMPO":    "SHAMPOO",
    "DAY/NIGHT": "DAY AND NIGHT",
    "DAY & NIGHT":"DAY AND NIGHT",

    # ── Skin / product descriptors ────────────────────────────────────────────
    "MOIST":    "MOISTURIZING",
    "MOISTUR":  "MOISTURIZING",
    "HYDRAT":   "HYDRATING",
    "CLEANSR":  "CLEANSER",
    "CLEANS":   "CLEANSER",
    "SQUAL":    "SQUALANE",
    "V.DRY":    "VERY DRY",
     

    # ── Product / ingredient names ────────────────────────────────────────────
    "EFFACL":   "EFFACLAR",
    "ANTIHLS":  "ANTHELIOS",
    "CICAPLS":  "CICAPLAST",
    "NIACIN":   "NIACINAMIDE",
    "RETINL":   "RETINOL",
    "S/ACID":  "SALICYLIC ACID",
    "SALICYL": "SALICYLIC ACID",
    "SALIC":     "SALICYLIC ACID",

    # ── Quantity shorthands ───────────────────────────────────────────────────
    "X1":       "1PC",
    "X2":       "2PCS",
    "QTY":      "QUANTITY",

    # ── Typos — misspellings seen in POS exports ──────────────────────────────
    "MOISTIRISING":  "MOISTURIZING",
    "MOISTIRI":      "MOISTURIZING",
    "MOISTURIZ":     "MOISTURIZING",
    "MOISTURIZI":    "MOISTURIZING",


    # ── Brand shorthands appearing in product name fields ─────────────────────
    "J&J":      "JOHNSON'S",
    "J & J":    "JOHNSON'S",

    # ── Add new entries below this line ───────────────────────────────────────
    # Example: "FOAMG": "FOAMING",
    # Example: "SPRY":  "SPRAY",

}


# =============================================================================
# BRAND ALIASES
# Maps POS brand shorthands, typos and variants → canonical brand name.
# Used by brand detection in matching and analytics pipelines.
# Keys matched case-insensitively.
# =============================================================================

BRAND_ALIASES: dict[str, str] = {

    # ── La Roche Posay ────────────────────────────────────────────────────────
    "lrp":              "La Roche Posay",
    "la ro":            "La Roche Posay",
    "la roche":         "La Roche Posay",
    "la roche posay":   "La Roche Posay",
    "larocheposay":     "La Roche Posay",
    "la roche-posay":   "La Roche Posay",
    "laroche":          "La Roche Posay",
    "posay":            "La Roche Posay",
    "roche":            "La Roche Posay",
    "effaclar":         "La Roche Posay",
    "anthelios":        "La Roche Posay",
    "lipikar":          "La Roche Posay",
    "cicaplast baume":  "La Roche Posay",

    # ── L'Oreal ───────────────────────────────────────────────────────────────
    "loreal":           "L'Oreal",
    "l'oreal":          "L'Oreal",
    "l oreal":          "L'Oreal",
    "lâ€™oreal":        "L'Oreal",

    # ── The Body Shop ─────────────────────────────────────────────────────────
    "body shop":        "The Body Shop",
    "the body shop":    "The Body Shop",

    # ── Palmer's ──────────────────────────────────────────────────────────────
    "palmers":          "Palmer's",
    "palmer":           "Palmer's",

    # ── Neutrogena ────────────────────────────────────────────────────────────
    "neut":             "Neutrogena",

    # ── Dr Organics ───────────────────────────────────────────────────────────
    "dr. organic":      "Dr Organic",
    "dr organics":      "Dr Organic",
    "dr.organic":       "Dr Organic",
    "dr charcoal":     "Dr Organic",
    "dr coffee":       "Dr Organic",
    "dr ginseng":      "Dr Organic",
    "dr moroccan":     "Dr Organic",
    "dr skin":         "Dr Organic",
    "dr tea tree":      "Dr Organic",
    "dr ageless":      "Dr Organic",
    "dr aloe":        "Dr Organic",
    "dr aroma":        "Dr Organic",
    "dr coconut":      "Dr Organic",
    "dr guavea":       "Dr Organic",
    "dr hemp oil":         "Dr Organic",
    "dr manuka honey":     "Dr Organic",
    "dr olive oil":        "Dr Organic",
    "dr pomegranate":      "Dr Organic",
    "dr rose":            "Dr Organic",
    "dr royal jelly":      "Dr Organic",
    "dr snail":             "Dr Organic",
    "dr.snail":             "Dr Organic",
    "dr vitamin":           "Dr Organic",


    # ── Johnson's ─────────────────────────────────────────────────────────────
    "johnsons":         "Johnson's",
    "johnson's baby":   "Johnson's",
    "j&j":              "Johnson's",
    "j & j":            "Johnson's",

    # ── Shea Moisture ─────────────────────────────────────────────────────────
    "sheamoisture":     "Shea Moisture",
    "shea moisture":    "Shea Moisture",
    "shea/m":           "Shea Moisture",

    # ── Mizani ────────────────────────────────────────────────────────────────
    "mizan":            "Mizani",

    # ── Natures Truth ─────────────────────────────────────────────────────────
    "nt":               "Natures Truth",

    # ── Oxygen Botanicals ─────────────────────────────────────────────────────
    "oxygen radiance":           "Oxygen",
    "oxygen botanicals":         "Oxygen",
    "oxygen squalene":           "Oxygen",

    # ── Avent ─────────────────────────────────────────────────────────────────
    "avent":            "Avent",
    "philips avent":    "Avent",

    # ── Bio Oil ───────────────────────────────────────────────────────────────
    "bio oil":          "Bio Oil",
    "bio-oil":          "Bio Oil",

    # ── Black Girl Sunscreen ──────────────────────────────────────────────────
    "black girl":       "Black Girl Sunscreen",
    "bgs":              "Black Girl Sunscreen",

    # ── Clean & Clear ─────────────────────────────────────────────────────────
    "clean and clear":  "Clean & Clear",

    # ── E45 ───────────────────────────────────────────────────────────────────
    "e 45":             "E45",
    "e45":              "E45",

    # ── Paula's Choice ────────────────────────────────────────────────────────
    "paulas choice":    "Paula's Choice",
    "paula choice":     "Paula's Choice",

    # ── Oral-B ────────────────────────────────────────────────────────────────
    "oral b":           "Oral-B",
    "oral-b":           "Oral-B",
    "oralb":            "Oral-B",

    # ── NOW Foods ─────────────────────────────────────────────────────────────
    "now foods":        "NOW Foods",

    # ── Forever Living ────────────────────────────────────────────────────────
    "forever":          "Forever Living",
    "forever aloe":     "Forever Living",

    # ── Hipp Organic ──────────────────────────────────────────────────────────
    "hipp":             "Hipp Organic",

    # ── The Ordinary ──────────────────────────────────────────────────────────
    "ordinary":             "The Ordinary",
    "the ordinary":         "The Ordinary",
    "the ordi":              "The Ordinary",
    "ordinary niacinamide": "The Ordinary",

    # ── CeraVe ────────────────────────────────────────────────────────────────
    "cereva":           "CeraVe",
    "cereve":           "CeraVe",
    "cera ve":          "CeraVe",

    # ── Deep Heat / Deep Freeze ───────────────────────────────────────────────
    "deep heat":        "Deep Heat",
    "deep freeze":      "Deep Freeze",

    # ── Seven Seas ─────────────────────────────────
    "s/seas":         "Seven Seas",
    "sevenseas":        "Seven Seas",

    # ── Regaine ─────────────────────────────────
    "regain":         "Regaine",

    # ── Vital Proteins ────────────────────────────────────────────
    "vital prot":       "Vital Proteins",
    "vital proteins":   "Vital Proteins",

    # ── Advanced Clinicals ────────────────────────────────────────────
    "advanced clinicl": "Advanced Clinicals",

    # ── sport supplies ────────────────────────────────────────────
    "ss":               "Sport Supplies",

    # ── sol de janeiro ────────────────────────────────────────────
    "sol de janeiro":   "Sol De Janeiro",
    "sdj":              "Sol De Janeiro",

    # ── NOW FOODS ────────────────────────────────────────────
    "now":              "NOW Foods",
    
    # ── Uncover ────────────────────────────────────────────
    "i am":           "Uncover",

    # ── Vitabiotics ────────────────────────────────────────────
    "wellkid":         "Vitabiotics",
    "wellman":         "Vitabiotics",
    "wellwoman":       "Vitabiotics",
    "wellbaby":         "Vitabiotics",
    "wellkids":         "Vitabiotics",
    "wellteen":         "Vitabiotics",
    "pregnacare":       "Vitabiotics",

    # ── Always ────────────────────────────────────────────
    "always":         "Always P&G",

    # ── Beauty of Joseon ────────────────────────────────────────────
    "boj":             "Beauty of Joseon",

    # ── Macks ────────────────────────────────────────────
    "mack's":         "Macks",

    # ── Tiam ────────────────────────────────────────────
    "tia'm":           "Tiam",

    # ── New Leaf ────────────────────────────────────────────
    "nl":              "New Leaf",

    # ── Quest ────────────────────────────────────────────
    "qst":              "Quest",

    # ── Bath & BOdy ────────────────────────────────────────────
    "bbw":              "Bath & Body Works",

    # ── LA Colours ────────────────────────────────────────────
    




    # ── Add new brand aliases below this line ─────────────────────────────────
    # Example: "vichy":  "Vichy",
    # Example: "cosrx":  "CosRx",

}