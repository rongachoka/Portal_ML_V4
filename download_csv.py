import os
import subprocess
from dotenv import load_dotenv
import shutil

load_dotenv()



host     = os.getenv("DB_HOST")
port     = os.getenv("DB_PORT", "5432")
password = os.getenv("DB_PASSWORD")
database = os.getenv("DB_NAME")
user     = "portal_user"
out_dir  = r"D:\Documents\Portal ML Analys\Portal_ML\Portal_ML_V4\data\03_processed\db_exports"

missing = {k: v for k, v in {"DB_HOST": host, "DB_PASSWORD": password, "DB_NAME": database}.items() if v is None}
if missing:
    print("❌ Missing .env variables:", list(missing.keys()))
    exit()

print("✅ Credentials loaded:")
print(f"   Host: {host}")
print(f"   DB:   {database}")
print(f"   Pass: {'*' * len(password)}")

os.makedirs(out_dir, exist_ok=True)

views = ["vw_sales_base", "vw_sales_with_margin", "vw_dead_stock", 
         "mv_transaction_master", "mv_client_list", "dim_branch",
         "dim_departments", "vw_aligned_date", "vw_inventory_snapshot"]

env = os.environ.copy()
env["PGPASSWORD"] = password

psql = r"C:\Program Files\PostgreSQL\18\bin\psql.exe"

for view in views:
    out_file = os.path.join(out_dir, f"{view}.csv").replace("\\", "/")
    cmd = [
        psql, "-h", host, "-p", port, "-U", user, "-d", database,
        "-c", f"\\copy (SELECT * FROM {view}) TO '{out_file}' WITH (FORMAT CSV, HEADER)"
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {view}")
    else:
        print(f"❌ {view}: {result.stderr.strip()}")