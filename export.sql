\copy (SELECT * FROM vw_sales_base) TO 'D:/Documents/Portal ML Analys/vw_sales_base.csv' CSV HEADER
\copy (SELECT * FROM vw_sales_with_margin) TO 'D:/Documents/Portal ML Analys/vw_sales_with_margin.csv' CSV HEADER
\copy (SELECT * FROM mv_transaction_master) TO 'D:/Documents/Portal ML Analys/mv_transaction_master.csv' CSV HEADER
\copy (SELECT * FROM mv_client_list) TO 'D:/Documents/Portal ML Analys/mv_client_list.csv' CSV HEADER
\copy (SELECT * FROM vw_dead_stock) TO 'D:/Documents/Portal ML Analys/vw_dead_stock.csv' CSV HEADER
\copy (SELECT * FROM dim_branch) TO 'D:/Documents/Portal ML Analys/dim_branch.csv' CSV HEADER
\copy (SELECT * FROM dim_date) TO 'D:/Documents/Portal ML Analys/dim_date.csv' CSV HEADER