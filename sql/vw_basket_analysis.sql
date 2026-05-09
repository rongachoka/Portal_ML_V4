CREATE OR REPLACE VIEW vw_basket_analysis AS
SELECT
    t.transaction_id,
    t.location                                        AS branch,
    t.sale_date,
    t.client_name,
    t.phone_number,

    -- Core basket metrics
    SUM(t.line_revenue)                               AS basket_value,
    COUNT(DISTINCT t.department)                      AS departments_per_txn,
    COUNT(DISTINCT t.description)                     AS unique_products_per_txn,
    SUM(t.qty_sold)                                   AS total_units_per_txn,

    -- Department flags
    MAX(CASE WHEN t.department = 'Skincare'               THEN 1 ELSE 0 END) AS has_skincare,
    MAX(CASE WHEN t.department = 'Cosmetics'              THEN 1 ELSE 0 END) AS has_cosmetics,
    MAX(CASE WHEN t.department = 'General Prescription'   THEN 1 ELSE 0 END) AS has_rx,
    MAX(CASE WHEN t.department = 'OTC'                    THEN 1 ELSE 0 END) AS has_otc,
    MAX(CASE WHEN t.department = 'Supplements'            THEN 1 ELSE 0 END) AS has_supplements,
    MAX(CASE WHEN t.department = 'Baby Care'              THEN 1 ELSE 0 END) AS has_baby_care,
    MAX(CASE WHEN t.department = 'Hair Care'              THEN 1 ELSE 0 END) AS has_hair_care,
    MAX(CASE WHEN t.department = 'Medicine & Treatment'   THEN 1 ELSE 0 END) AS has_medicine,
    MAX(CASE WHEN t.department = 'Antibiotics'            THEN 1 ELSE 0 END) AS has_antibiotics,
    MAX(CASE WHEN t.department = 'Feminine Care'          THEN 1 ELSE 0 END) AS has_feminine_care,
    MAX(CASE WHEN t.department = 'First Aid'              THEN 1 ELSE 0 END) AS has_first_aid,
    MAX(CASE WHEN t.department = 'Lip Care'               THEN 1 ELSE 0 END) AS has_lip_care,
    MAX(CASE WHEN t.department = 'Medical Devices & Kits' THEN 1 ELSE 0 END) AS has_medical_devices,
    MAX(CASE WHEN t.department = 'Men Care'               THEN 1 ELSE 0 END) AS has_men_care,
    MAX(CASE WHEN t.department = 'Oral Care'              THEN 1 ELSE 0 END) AS has_oral_care,
    MAX(CASE WHEN t.department = 'Perfumes'               THEN 1 ELSE 0 END) AS has_perfumes,
    MAX(CASE WHEN t.department = 'Homeopathy'             THEN 1 ELSE 0 END) AS has_homeopathy,
    MAX(CASE WHEN t.department = 'Accessories'            THEN 1 ELSE 0 END) AS has_accessories,
    MAX(CASE WHEN t.department = 'Jewellery'              THEN 1 ELSE 0 END) AS has_jewellery,
    MAX(CASE WHEN t.department = 'Cards & Airtime'        THEN 1 ELSE 0 END) AS has_cards_airtime,
    MAX(CASE WHEN t.department = 'Vaccines'               THEN 1 ELSE 0 END) AS has_vaccines,
    MAX(CASE WHEN t.department = 'ANTIFUNGAL'             THEN 1 ELSE 0 END) AS has_antifungal,
    MAX(CASE WHEN t.department = 'CONTRACEPTIVES'         THEN 1 ELSE 0 END) AS has_contraceptives,
    MAX(CASE WHEN t.department = 'PPI'                    THEN 1 ELSE 0 END) AS has_ppi

FROM vw_sales_with_margin t
WHERE t.transaction_id IS NOT NULL
  AND t.department IS NOT NULL
  AND t.department NOT IN ('DELIVERY', 'SAMPLES', 'Interbranch', 'Wholesale')
  AND t.line_revenue > 0
  AND t.description NOT ILIKE '%GOODS%'
  AND t.description NOT ILIKE '%#NULL#%'
  AND t.description <> ''
GROUP BY
    t.transaction_id,
    t.location,
    t.sale_date,
    t.client_name,
    t.phone_number;