# ----------------------------
# Default config (change paths if needed)
# ----------------------------
DEFAULT_INPUT_MESSAGES = "C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\chat_data\\test_run\\Data Export Respond.io Messages.csv"
DEFAULT_INPUT_CONTACTS = "C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\chat_data\\test_run\\Data Export Respond.io Contacts.csv"
DEFAULT_INPUT_CONVERSATIONS = "C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\chat_data\\test_run\\Data Export Respond.io.csv"
DEFAULT_PRODUCT_LIST = "C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\chat_data\\Combined Qty List 311025.csv"
DEFAULT_PRODUCT_CATEGORIES = "C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\chat_data\\Brand Categories Full.csv"

# ----------------------------
# Product Category files config 
# ----------------------------
product_raw = "C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\product_files\\All Website Product Info.csv"

# output folder (will be created)
DEFAULT_OUTPUT_FOLDER = r"C:\\Users\\Portal Pharmacy\\Documents\\Portal ML Analys\\Ron's Work (2)\\Portal_ML\\Portal_ML_V2\\outputs"

# models / thresholds
CROSS_ENCODER_MODEL = "cross-encoder/nli-MiniLM2-L6-H768"  # fast cross-encoder
FUZZY_PRODUCT_THRESHOLD = 90
FUZZY_BRAND_THRESHOLD = 95
CATEGORY_CONFIDENCE_THRESHOLD = 0.80
HIGH_VALUE_THRESHOLD = 10000  # KES for "High-Value Customer"
LOYAL_PAYMENTS_THRESHOLD = 3
HIGH_LTV_SUM_THRESHOLD = 50000
SESSION_GAP_HOURS = 24
