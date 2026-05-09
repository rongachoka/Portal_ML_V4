ALTER TABLE dim_products
ADD COLUMN IF NOT EXISTS selling_price_incl_vat NUMERIC;

UPDATE dim_products
SET selling_price_incl_vat = CASE
    WHEN department_mapped IN (
        'OTC',
        'General Prescription',
        'Medicine & Treatment',
        'Supplements',
        'Antibiotics',
        'PPI',
        'Vaccines',
        'CONTRACEPTIVES',
        'ANTIFUNGAL',
        'First Aid',
        'Homeopathy'
    ) THEN selling_price
    ELSE selling_price * 1.16
END
WHERE selling_price IS NOT NULL;