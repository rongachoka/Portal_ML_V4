import pandas as pd
from pathlib import Path
from Portal_ML_V4.src.config.settings import PROCESSED_DATA_DIR, DB_CONNECTION_STRING
from Portal_ML_V4.src.utils.name_cleaner import resolve_best_name

def run_client_export():
    print("\n📋 EXPORTING CLIENT LIST...")

    try:
        from sqlalchemy import create_engine
        engine = create_engine(DB_CONNECTION_STRING)

        df = pd.read_sql("SELECT * FROM vw_client_list", engine)

        print(f"   ✅ {len(df):,} clients loaded.")

        if 'client_name' in df.columns:
            df['client_name'] = df.apply(
                lambda r: resolve_best_name(r['client_name'], r.get('name_audit_flag')),
                axis=1
            )
            print("   🧹 Names resolved.")

        platinum = df[df['lifetime_tier'] == 'Platinum'].copy()
        gold     = df[df['lifetime_tier'] == 'Gold'].copy()
        silver   = df[df['lifetime_tier'] == 'Silver'].copy()
        bronze = df[df['lifetime_tier'] == 'Bronze'].copy()

        print(f"   💎 Platinum: {len(platinum):,}")
        print(f"   🥇 Gold:     {len(gold):,}")
        print(f"   🥈 Silver:   {len(silver):,}")
        print(f"   🥉 Bronze:   {len(bronze):,}")

        OUTPUT_PATH = PROCESSED_DATA_DIR / "Portal_Client_List.xlsx"
        with pd.ExcelWriter(OUTPUT_PATH, engine='openpyxl') as writer:
            df.to_excel(writer,       sheet_name='All Clients', index=False)
            platinum.to_excel(writer, sheet_name='Platinum',    index=False)
            gold.to_excel(writer,     sheet_name='Gold',        index=False)
            silver.to_excel(writer,   sheet_name='Silver',      index=False)
            bronze.to_excel(writer,   sheet_name='Bronze',      index=False)

        print(f"   📂 Client list saved to: {OUTPUT_PATH}")

    except Exception as e:
        print(f"   ❌ Client export failed: {e}")

    finally:
        engine.dispose()

if __name__ == "__main__":
    run_client_export()