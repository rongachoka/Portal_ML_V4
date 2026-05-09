# import pandas as pd
# import glob
# from Portal_ML_V4.src.config.settings import BASE_DIR

# # Check the raw cashier file for Galleria
# cashier_path = BASE_DIR / "data" / "01_raw" / "pos_data" / "galleria"

# cashier_files = glob.glob(str(cashier_path / "*Cashier*"))
# print(cashier_files)

# for f in cashier_files:
#     df = pd.read_excel(f)
#     df.columns = df.columns.str.strip()
#     if 'Receipt Txn No' in df.columns:
#         match = df[df['Receipt Txn No'].astype(str).str.strip() == '1001909']
#         if not match.empty:
#             print(f"\nFound in: {f}")
#             print(match[['Receipt Txn No', 'Client Name', 'Phone Number', 'Amount']].to_string())

import pandas as pd
from Portal_ML_V4.src.config.settings import BASE_DIR

cashier_path = BASE_DIR / "data" / "01_raw" / "pos_data" / "galleria"
files = [
    cashier_path / "Galleria  Daily Cashier report  Feb 2026.xlsm",
    cashier_path / "Galleria  Daily Cashier report  Jan 2026.xlsm"
]

for f in files:
    print(f"\n📂 File: {f.name}")
    xls = pd.ExcelFile(f)
    for sheet in xls.sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet)
            # Fix: convert all column names to string before stripping
            df.columns = [str(c).strip() for c in df.columns]
            
            for col in df.columns:
                try:
                    match = df[col].astype(str).str.strip().str.replace('.0','', regex=False).eq('1001909')
                    if match.any():
                        print(f"  ✅ Found in sheet: '{sheet}', column: '{col}'")
                        print(f"  Columns available: {list(df.columns)}")
                        print(df[match].to_string())
                except Exception:
                    continue
        except Exception as e:
            print(f"  ⚠️ Could not read sheet '{sheet}': {e}")
            continue