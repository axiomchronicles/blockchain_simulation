
# Ultimate Blockchain API

---

## 🌐 Project Overview

Welcome to **Ultimate Blockchain API**, the **most advanced, modular, and scalable blockchain framework** built with Python and Aquilify. This project implements a cutting-edge blockchain system featuring:

- 🔄 **Multi-Consensus:** Support for PoW, PoS, DPoS, and PBFT with dynamic switching  
- 🛡️ **Quantum-Resistant Cryptography:** Secure key generation and signatures with fallback classical algorithms  
- 🌐 **Advanced P2P Networking:** Kademlia DHT, gossip protocols, and peer discovery  
- 💻 **Smart Contracts VM:** Stack-based VM with gas metering, event logging, and error handling  
- ⚡ **High Performance:** Async batch validation, caching, multiprocessing, and concurrency optimizations  
- 🔗 **Sharding & Cross-Chain:** Scalable sharding architecture with cross-shard communication and atomic swaps  
- 💼 **HD Wallets & Governance:** BIP32-style hierarchical wallets, multi-signature support, on-chain voting, and treasury management  
- 📊 **Real-Time Analytics:** Integrated blockchain explorer, transaction metrics, and monitoring dashboards  
- 🔒 **Robust Security:** Rate limiting, DDoS protection, formal verification hooks, and secure cryptographic primitives  

---

## ⚙️ Technologies & Core Libraries

- Python 3.10+ leveraging modern async/await and concurrency libraries  
- [Aquilify](https://github.com/aquiladev/aquilify) — ultra-lightweight, async web framework for Python  
- [Electrus](https://github.com/axiomchron/electrus) — performant async JSON-based NoSQL database backend  
- [Pydantic](https://pydantic.dev/) — powerful data validation, parsing, and serialization  
- NumPy for efficient blockchain data structures and type enforcement  
- Custom quantum-resistant cryptography algorithms with fallback classical methods  
- Dataclasses for domain modeling and clean code organization  
- ThreadPoolExecutor and multiprocessing for parallel workloads  
- Logging with structured JSON output for enhanced observability  

---

## 🛠️ API Endpoints

| Path                         | Method | Description                                              |
|------------------------------|--------|----------------------------------------------------------|
| `/wallet/create`              | POST   | Create a new HD wallet with hierarchical deterministic addresses |
| `/wallet/{wallet_id}`         | GET    | Retrieve wallet details and all associated addresses     |
| `/transaction/create`         | POST   | Create, sign, and submit a new blockchain transaction     |
| `/transaction/{tx_id}`        | GET    | Retrieve transaction details by transaction ID            |
| `/transaction/verify/{tx_id}` | GET    | Verify the cryptographic signature and validity of a transaction |
| `/contract/execute`           | POST   | Execute smart contract bytecode within the stack-based VM |
| `/consensus/switch`           | POST   | Dynamically switch blockchain consensus protocol (PoW, PoS, DPoS, PBFT) |
| `/block/validate`             | POST   | Validate a submitted block against consensus rules and chain state |

---

## 🔥 Quick Start Guide

### 1. Clone & Setup Environment

```bash
git clone https://github.com/axiomchronicles/blockchain_simulation.git
cd blockchain_simulation
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional)

Create a `.env` file to customize ports and consensus settings:

```env
P2P_PORT=9000
RPC_PORT=8000
WS_PORT=8001
DHT_PORT=9001
CONSENSUS=PoW  # Options: PoW, PoS, DPoS, PBFT
```

### 3. Run the API Server

```bash
aquilify run api.core:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the APIs using Python `requests`

```python
import requests

# Create a new wallet
wallet_resp = requests.post("http://localhost:8000/api/v1/wallet/create")
print("Create Wallet:", wallet_resp.status_code, wallet_resp.json())

# Extract wallet ID and addresses
wallet_id = wallet_resp.json().get("wallet_id")
addresses = wallet_resp.json().get("addresses")
sender_address = addresses[0]

# Create a transaction
tx_payload = {
    "sender": sender_address,
    "recipient": "recipient_address_here",
    "amount": 10.0,
    "nonce": 0,
    "gas_price": 1.0,
    "gas_limit": 21000,
    "tx_type": "TRANSFER"
}
tx_resp = requests.post("http://localhost:8000/transaction/create", json=tx_payload)
print("Create Transaction:", tx_resp.status_code, tx_resp.json())

# Verify transaction signature
tx_id = tx_resp.json().get("tx_id")
verify_resp = requests.get(f"http://localhost:8000/transaction/verify/{tx_id}")
print("Verify Transaction:", verify_resp.status_code, verify_resp.json())
```

---

## 🏗️ Code Architecture & Directory Structure

```text
ultimate-blockchain/
├── api/
│   ├── core.py               # Aquilify API routes and request handlers
│   ├── schema.py             # Pydantic models for data validation and serialization
│   ├── helpers.py            # Utility functions (serialization, conversions)
│   ├── database.py           # Async Electrus collections and DB interfacing
│
├── simulation/
│   ├── crypto.py             # Wallet management & quantum-resistant crypto implementations
│   ├── networking.py         # Consensus engines, block validation, P2P networking
│   ├── vm.py                 # Smart contract virtual machine and opcode execution
│
├── tests/                    # Unit and integration tests for API and core modules
│
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation and usage guide
└── aquilify_config.py        # Server config and Aquilify application setup
```

---

## 🔐 Security Considerations

- This is a **prototype-level implementation** aimed at research and learning.  
- Replace simulated quantum-resistant cryptography with audited real-world algorithms before production.  
- Secure private key storage and management is critical; never expose private keys or seeds.  
- Add thorough rate limiting, authentication, and authorization in production environments.  
- Smart contract code must be audited and formally verified where possible.  
- Regularly monitor and update dependencies to patch security vulnerabilities.

---

## 🤝 How to Contribute

- Report issues or bugs via GitHub issues.  
- Submit pull requests with well-documented improvements.  
- Participate in design discussions and feature proposals.  
- Add comprehensive tests to cover new features or fixes.

---

## 📜 License

Licensed under the **MIT License** — see [LICENSE](LICENSE) file for full details.

---

## 📬 Contact

**Maintainer:** Your Name  
**Email:** your.email@example.com  
**GitHub:** [https://github.com/axiomchronicles](https://github.com/axiomchronicles)  

---

Thank you for exploring **Ultimate Blockchain API** — powering innovation in decentralized systems!

