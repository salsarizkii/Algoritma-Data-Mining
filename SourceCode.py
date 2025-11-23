import pandas as pd
df = pd.read_csv('Groceries_dataset.csv')
print(df.head())

# mengelompokkan per transaksi
grouped = df.groupby(['Member_number', 'Date'])['itemDescription'].apply(list)
transactions = grouped.tolist()

from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

from mlxtend.frequent_patterns import apriori, association_rules

# mencari frequent itemsets
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

print(frequent_itemsets.sort_values(by='support', ascending=False).head())

# membuat aturan asosiasi
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# menambah kolom lift dan urut berdasarkan lift tertinggi
rules = rules.sort_values(by="lift", ascending=False)

print(frequent_itemsets.head())
print(rules.head())