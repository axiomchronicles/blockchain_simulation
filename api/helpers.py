from typing import Dict, Any
from simulation.crypto import Transaction

# -------------------------
# Helpers
# -------------------------

def serialize_transaction(tx: Transaction) -> Dict[str, Any]:
    return {
        "tx_id": tx.tx_id,
        "sender": tx.sender,
        "recipient": tx.recipient,
        "amount": tx.amount,
        "nonce": tx.nonce,
        "gas_price": tx.gas_price,
        "gas_limit": tx.gas_limit,
        "tx_type": tx.tx_type.name,
        "data": tx.data,
        "timestamp": tx.timestamp,
        "signature": tx.signature.hex() if tx.signature else None,
        "shard_id": tx.shard_id,
        "contract_code": tx.contract_code,
    }
