-- ============================================================
-- VIEW: vw_basket_analysis
-- Database: portal_pharmacy
-- ============================================================

CREATE OR REPLACE VIEW vw_basket_analysis AS
 SELECT transaction_id,
    location AS branch,
    sale_date,
    client_name,
    phone_number,
    sum(line_revenue) AS basket_value,
    count(DISTINCT department) AS departments_per_txn,
    count(DISTINCT description) AS unique_products_per_txn,
    sum(qty_sold) AS total_units_per_txn,
    max(CASE WHEN (department = 'Skincare'::text)              THEN 1 ELSE 0 END) AS has_skincare,
    max(CASE WHEN (department = 'Cosmetics'::text)             THEN 1 ELSE 0 END) AS has_cosmetics,
    max(CASE WHEN (department = 'General Prescription'::text)  THEN 1 ELSE 0 END) AS has_rx,
    max(CASE WHEN (department = 'OTC'::text)                   THEN 1 ELSE 0 END) AS has_otc,
    max(CASE WHEN (department = 'Supplements'::text)           THEN 1 ELSE 0 END) AS has_supplements,
    max(CASE WHEN (department = 'Baby Care'::text)             THEN 1 ELSE 0 END) AS has_baby_care,
    max(CASE WHEN (department = 'Hair Care'::text)             THEN 1 ELSE 0 END) AS has_hair_care,
    max(CASE WHEN (department = 'Medicine & Treatment'::text)  THEN 1 ELSE 0 END) AS has_medicine,
    max(CASE WHEN (department = 'Antibiotics'::text)           THEN 1 ELSE 0 END) AS has_antibiotics,
    max(CASE WHEN (department = 'Feminine Care'::text)         THEN 1 ELSE 0 END) AS has_feminine_care,
    max(CASE WHEN (department = 'First Aid'::text)             THEN 1 ELSE 0 END) AS has_first_aid,
    max(CASE WHEN (department = 'Lip Care'::text)              THEN 1 ELSE 0 END) AS has_lip_care,
    max(CASE WHEN (department = 'Medical Devices & Kits'::text) THEN 1 ELSE 0 END) AS has_medical_devices,
    max(CASE WHEN (department = 'Men Care'::text)              THEN 1 ELSE 0 END) AS has_men_care,
    max(CASE WHEN (department = 'Oral Care'::text)             THEN 1 ELSE 0 END) AS has_oral_care,
    max(CASE WHEN (department = 'Perfumes'::text)              THEN 1 ELSE 0 END) AS has_perfumes,
    max(CASE WHEN (department = 'Homeopathy'::text)            THEN 1 ELSE 0 END) AS has_homeopathy,
    max(CASE WHEN (department = 'Accessories'::text)           THEN 1 ELSE 0 END) AS has_accessories,
    max(CASE WHEN (department = 'Jewellery'::text)             THEN 1 ELSE 0 END) AS has_jewellery,
    max(CASE WHEN (department = 'Cards & Airtime'::text)       THEN 1 ELSE 0 END) AS has_cards_airtime,
    max(CASE WHEN (department = 'Vaccines'::text)              THEN 1 ELSE 0 END) AS has_vaccines,
    max(CASE WHEN (department = 'ANTIFUNGAL'::text)            THEN 1 ELSE 0 END) AS has_antifungal,
    max(CASE WHEN (department = 'CONTRACEPTIVES'::text)        THEN 1 ELSE 0 END) AS has_contraceptives,
    max(CASE WHEN (department = 'PPI'::text)                   THEN 1 ELSE 0 END) AS has_ppi
   FROM vw_sales_with_margin t
  WHERE ((transaction_id IS NOT NULL)
    AND (department IS NOT NULL)
    AND (department <> ALL (ARRAY['DELIVERY'::text, 'SAMPLES'::text, 'Interbranch'::text, 'Wholesale'::text, 'BRANCH'::text, 'BRANCHES'::text]))
    AND (line_revenue > (0)::numeric)
    AND (description !~~* '%#NULL#%'::text)
    AND (description <> ''::text))
  GROUP BY transaction_id, location, sale_date, client_name, phone_number;
