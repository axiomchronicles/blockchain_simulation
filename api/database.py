from axiomelectrus import Electrus

client = Electrus()
db = client["blockchain"]

## Collections
wallets_collection = db["wallets"]
tx_collection = db["transactions"]
blocks_collection = db["blocks"]