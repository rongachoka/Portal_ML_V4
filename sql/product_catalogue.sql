CREATE TABLE IF NOT EXISTS dim_product_catalogue AS
WITH normalised AS (
    SELECT
        LTRIM(item_lookup_code, '0')    AS item_lookup_code,
        description,
        department,
        category,
        supplier,
        COUNT(*)                        AS usage_count
    FROM stg_qty_list
    WHERE item_lookup_code IS NOT NULL
      AND TRIM(item_lookup_code) != ''
      AND LOWER(TRIM(item_lookup_code)) NOT IN ('nan', 'none')
      AND description IS NOT NULL
      AND TRIM(description) != ''
    GROUP BY LTRIM(item_lookup_code, '0'), description, department, category, supplier
),
ranked AS (
    SELECT
        item_lookup_code,
        description,
        department,
        category,
        supplier,
        usage_count,
        ROW_NUMBER() OVER (
            PARTITION BY item_lookup_code
            ORDER BY usage_count DESC, LENGTH(description) DESC
        ) AS rn
    FROM normalised
)
SELECT
    item_lookup_code,
    description         AS canonical_name,
    department,
    category,
    supplier,
    usage_count         AS times_used
FROM ranked
WHERE rn = 1;

CREATE UNIQUE INDEX idx_product_catalogue_barcode
    ON dim_product_catalogue(item_lookup_code);

CREATE INDEX idx_product_catalogue_name
    ON dim_product_catalogue(canonical_name);

CREATE INDEX idx_product_catalogue_department
    ON dim_product_catalogue(department);