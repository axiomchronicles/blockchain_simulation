import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_create_wallet():
    print("Creating Wallet...")
    resp = requests.post(f"{BASE_URL}/wallets")
    print("Status:", resp.status_code)
    print("Response:", resp.json())
    return resp.json()

def test_get_wallet(wallet_id):
    print(f"Getting Wallet {wallet_id} ...")
    resp = requests.get(f"{BASE_URL}/wallets/{wallet_id}")
    print("Status:", resp.status_code)
    print("Response:", resp.json())

def test_create_transaction(sender, recipient, amount):
    print("Creating Transaction...")
    payload = {
        "sender": sender,
        "recipient": recipient,
        "amount": amount,
        "nonce": 0,
        "gas_price": 1.0,
        "gas_limit": 21000,
        "tx_type": "TRANSFER"
    }
    resp = requests.post(f"{BASE_URL}/transactions", json=payload)
    print("Status:", resp.status_code)
    print("Response:", resp.json())
    return resp.json()

def test_get_transaction(tx_id):
    print(f"Getting Transaction {tx_id} ...")
    resp = requests.get(f"{BASE_URL}/transactions/{tx_id}")
    print("Status:", resp.status_code)
    print("Response:", resp.json())

def test_verify_transaction(tx_id):
    print(f"Verifying Transaction {tx_id} ...")
    resp = requests.get(f"{BASE_URL}/transactions/{tx_id}/verify")
    print("Status:", resp.status_code)
    print("Response:", resp.json())

def test_execute_contract(contract_code):
    print("Executing Contract...")
    payload = {
        "contract_code": contract_code,
        "gas_limit": 1000000
    }
    resp = requests.post(f"{BASE_URL}/contracts/execute", json=payload)
    print("Status:", resp.status_code)
    print("Response:", resp.json())

def test_switch_consensus(consensus_type):
    print(f"Switching Consensus to {consensus_type} ...")
    payload = {"consensus_type": consensus_type}
    resp = requests.post(f"{BASE_URL}/consensus/switch", json=payload)
    print("Status:", resp.status_code)
    print("Response:", resp.json())

def test_validate_block(block, chain_state=None):
    print("Validating Block...")
    payload = {"block": block}
    if chain_state:
        payload["chain_state"] = chain_state
    resp = requests.post(f"{BASE_URL}/consensus/validate_block", json=payload)
    print("Status:", resp.status_code)
    print("Response:", resp.json())

if __name__ == "__main__":
    # Create wallet and get address
    wallet = test_create_wallet()
    wallet_id = wallet["wallet_id"]
    address = wallet["addresses"][0]

    # Get wallet info
    test_get_wallet(wallet_id)

    # Create a second wallet to send coins to
    wallet2 = test_create_wallet()
    recipient_address = wallet2["addresses"][0]

    # Create a transaction from wallet1 to wallet2
    tx = test_create_transaction(sender=address, recipient=recipient_address, amount=10.0)
    tx_id = tx["tx_id"]

    # Get transaction details
    test_get_transaction(tx_id)

    # Verify transaction signature
    test_verify_transaction(tx_id)

    # Execute a simple contract that pushes 10 and 20, adds them, and returns
    contract_code = [
        0, 10,  # PUSH 10
        0, 20,  # PUSH 20
        10,     # ADD
        32      # RETURN
    ]
    test_execute_contract(contract_code)

    # Switch consensus protocol
    test_switch_consensus("PoS")

    # Validate a dummy block (fill with realistic data in real tests)
    dummy_block = {
        "hash": "abc123",
        "previous_hash": "0000",
        "merkle_root": "def456",
        "timestamp": 1691599999.0,
        "height": 1,
        "nonce": 12345,
        "difficulty": 4,
        "tx_count": 1,
        "shard_id": 0
    }
    test_validate_block(dummy_block)
