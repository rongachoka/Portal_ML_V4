import pandas as pd
import psycopg2
import psycopg2.extras
import os
from dotenv import load_dotenv
from Portal_ML_V4.src.config.settings import BASE_DIR

load_dotenv()

KB_PATH = BASE_DIR / "data" / "01_raw" / "Final_Knowledge_Base_PowerBI.csv"

def load_kb_to_postgres():
    print("📖 Loading Knowledge Base to PostgreSQL...")

    df = pd.read_csv(KB_PATH)
    df.columns = df.columns.str.strip()

    # Normalise column names to match DB
    col_rename = {
        'Code 1':           'code_1',
        'Code 2':           'code_2',
        'ItemCode':         'item_code',
        'Name':             'name',
        'Brand':            'brand',
        'Canonical_Category': 'canonical_category',
        'Sub_Category':     'sub_category',
        'Concerns':         'concerns',
        'Target_Audience':  'target_audience',
        'Price':            'price',
        'Quantity':         'quantity',
        'Product_Link':     'product_link',
        'Detailed_Desc':    'detailed_desc',
    }
    df = df.rename(columns=col_rename)

    # Keep only columns that exist in both CSV and rename map
    valid_cols = [c for c in col_rename.values() if c in df.columns]
    df = df[valid_cols].copy()

    # Clean types
    df['price']    = pd.to_numeric(df['price'],    errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')

    # Replace NaN with None for psycopg2
    df = df.where(pd.notna(df), None)

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
    )
    cur = conn.cursor()

    # Truncate first so re-running is safe
    cur.execute("TRUNCATE TABLE dim_knowledge_base RESTART IDENTITY;")

    rows = [tuple(row) for row in df[valid_cols].itertuples(index=False)]

    INSERT_SQL = f"""
        INSERT INTO dim_knowledge_base ({', '.join(valid_cols)})
        VALUES %s
    """

    psycopg2.extras.execute_values(cur, INSERT_SQL, rows, page_size=500)
    conn.commit()

    print(f"   ✅ {len(df):,} products loaded to dim_knowledge_base")

    # Quick verification
    cur.execute("""
        SELECT canonical_category, COUNT(*) 
        FROM dim_knowledge_base 
        GROUP BY canonical_category 
        ORDER BY COUNT(*) DESC;
    """)
    print("\n   Category breakdown:")
    for row in cur.fetchall():
        print(f"      {str(row[0] or 'None'):<30} {row[1]:>5,}")

    cur.close()
    conn.close()

if __name__ == "__main__":
    load_kb_to_postgres()