"""
blockchain_inmem.py – Pure-Python, in-memory blockchain with:
• NumPy for UTXO/set and block storage
• Built-in `hashlib` for signatures and hashes
• Advanced algorithms: vectorised UTXO selection, dynamic difficulty retarget
• Bloom filter + heap for mempool prioritisation
• Pluggable consensus with adaptive PoW
"""

import json, time, uuid, threading, hashlib, heapq, math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────────
INITIAL_DIFFICULTY = 4
REWARD             = 50.0
TARGET_BLOCK_TIME  = 10.0        # seconds per block
RETARGET_INTERVAL  = 10          # adjust difficulty every 10 blocks
UTXO_RESET_PERIOD  = 1000        # prune array every 1000 blocks
# ────────────────────────────────────────────────────────────────────────────────


# ────────────────────────────────────────────────────────────────────────────────
# Enums & dtypes
# ────────────────────────────────────────────────────────────────────────────────
class TxType(Enum):
    TRANSFER = 0
    COINBASE = 1

TX_DTYPE = np.dtype([
    ('tx_id',     'U64'),
    ('sender',    'U42'),
    ('recipient', 'U42'),
    ('amount',    'f8'),
    ('nonce',     'i4'),
    ('fee',       'f8'),
    ('timestamp', 'f8'),
    ('tx_type',   'i2')
])

UTXO_DTYPE = np.dtype([
    ('tx_id',        'U64'),
    ('out_idx',      'i4'),
    ('recipient',    'U42'),
    ('amount',       'f8'),
    ('is_spent',     'b1'),
    ('created_height','i4')
])

# ────────────────────────────────────────────────────────────────────────────────
# Utility: built-in crypto using hashlib (fallback signature)
# ────────────────────────────────────────────────────────────────────────────────
class Crypto:
    @staticmethod
    def sha256(data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def sign(private: str, msg: str) -> str:
        # simple HMAC-like fallback: priv + msg
        return hashlib.sha256((private + msg).encode()).hexdigest()

    @staticmethod
    def verify(public: str, signature: str, msg: str) -> bool:
        # public is hashed priv; verify by regenerating signature
        return signature == hashlib.sha256((public + msg).encode()).hexdigest()


# ────────────────────────────────────────────────────────────────────────────────
# Transaction
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Transaction:
    sender:     str
    recipient:  str
    amount:     float
    nonce:      int = 0
    fee:        float = 0.0
    tx_type:    TxType = TxType.TRANSFER
    tx_id:      str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp:  float = field(default_factory=time.time)
    signature:  Optional[str] = None
    inputs:     List[str] = field(default_factory=list)
    outputs:    List[Dict[str, Any]] = field(default_factory=list)

    def core(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop('signature')
        d['tx_type'] = self.tx_type.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.core(), sort_keys=True)

    def hash(self) -> str:
        return Crypto.sha256(self.to_json())

    def sign(self, private_key: str):
        self.signature = Crypto.sign(private_key, self.to_json())

    def verify(self, public_key: str) -> bool:
        if self.tx_type is TxType.COINBASE:
            return True
        return bool(self.signature and Crypto.verify(public_key, self.signature, self.to_json()))


# ────────────────────────────────────────────────────────────────────────────────
# Bloom Filter for deduplication
# ────────────────────────────────────────────────────────────────────────────────
class BloomFilter:
    def __init__(self, capacity:int=100_000, err_rate:float=1e-3):
        m = int(-capacity * math.log(err_rate) / (math.log(2)**2))
        k = int((m / capacity) * math.log(2))
        self.bits = np.zeros(m, dtype=bool)
        self.m, self.k = m, k

    def _hashes(self, item: str) -> List[int]:
        h1 = hash(item) % self.m
        h2 = hash(item[::-1]) % self.m
        return [(h1 + i*h2) % self.m for i in range(self.k)]

    def add(self, item: str):
        for h in self._hashes(item):
            self.bits[h] = True

    def __contains__(self, item: str) -> bool:
        return all(self.bits[h] for h in self._hashes(item))


# ────────────────────────────────────────────────────────────────────────────────
# Mempool: priority queue by fee/size + bloom
# ────────────────────────────────────────────────────────────────────────────────
class Mempool:
    def __init__(self, capacity:int=10_000):
        self.cap = capacity
        self.txs: Dict[str, Transaction] = {}
        self.heap: List[Tuple[float, float, str]] = []
        self.bloom = BloomFilter(capacity*2)
        self.nonces: Dict[str,int] = {}
        self.lock = threading.RLock()

    def add(self, tx: Transaction) -> bool:
        with self.lock:
            if tx.tx_id in self.bloom: return False
            if len(self.txs) >= self.cap:    return False
            if tx.tx_type is not TxType.COINBASE:
                if tx.nonce != self.nonces.get(tx.sender,0):
                    return False
            size = len(tx.to_json().encode())
            prio = tx.fee / max(size,1)
            heapq.heappush(self.heap, (-prio, tx.timestamp, tx.tx_id))
            self.txs[tx.tx_id] = tx
            self.bloom.add(tx.tx_id)
            if tx.tx_type is not TxType.COINBASE:
                self.nonces[tx.sender] = tx.nonce + 1
            return True

    def take(self, limit:int=1000) -> List[Transaction]:
        with self.lock:
            chosen, buffer = [], []
            while self.heap and len(chosen)<limit:
                pr, ts, txid = heapq.heappop(self.heap)
                if txid in self.txs:
                    chosen.append(self.txs[txid])
                    buffer.append((pr,ts,txid))
            for entry in buffer:
                heapq.heappush(self.heap, entry)
            return chosen

    def purge(self, txids:List[str]):
        with self.lock:
            for tid in txids:
                self.txs.pop(tid, None)
            self.heap = [(p,t,tid) for (p,t,tid) in self.heap
                         if tid in self.txs]
            heapq.heapify(self.heap)


# ────────────────────────────────────────────────────────────────────────────────
# UTXO Set: NumPy array + vectorised operations
# ────────────────────────────────────────────────────────────────────────────────
class UTXOSet:
    def __init__(self):
        self.arr = np.empty(0, dtype=UTXO_DTYPE)
        self.idx: Dict[str,int] = {}
        self.addr_map: Dict[str,List[int]] = {}
        self.lock = threading.RLock()
        self.height = 0

    def add(self, tx_id:str, out_idx:int, recipient:str,
            amount:float, height:int):
        with self.lock:
            uid = f"{tx_id}:{out_idx}"
            rec = np.array([(tx_id,out_idx,recipient,amount,False,height)],
                           dtype=UTXO_DTYPE)
            self.arr = np.append(self.arr, rec)
            pos = len(self.arr)-1
            self.idx[uid] = pos
            self.addr_map.setdefault(recipient,[]).append(pos)

    def spend(self, tx_id:str, out_idx:int) -> bool:
        with self.lock:
            uid = f"{tx_id}:{out_idx}"
            if uid not in self.idx: return False
            pos = self.idx[uid]
            if self.arr[pos]['is_spent']: return False
            self.arr[pos]['is_spent'] = True
            return True

    def balance(self, addr:str) -> float:
        with self.lock:
            positions = self.addr_map.get(addr,[])
            if not positions: return 0.0
            subset = self.arr[positions]
            mask   = ~subset['is_spent']
            return float(subset['amount'][mask].sum())

    def select(self, addr:str, total_req:float) -> Tuple[List[Dict],float]:
        with self.lock:
            poss = [i for i in self.addr_map.get(addr,[])
                    if not self.arr[i]['is_spent']]
            poss.sort(key=lambda i:self.arr[i]['amount'],reverse=True)
            chosen, acc = [], 0.0
            for i in poss:
                ut = self.arr[i]
                chosen.append({
                    'tx_id': ut['tx_id'],
                    'out_idx': int(ut['out_idx']),
                    'amount': float(ut['amount'])
                })
                acc += ut['amount']
                if acc>=total_req:
                    break
            return chosen, acc

    def prune(self):
        # Every UTXO_RESET_PERIOD, rebuild arrays to drop spent
        mask = ~self.arr['is_spent']
        self.arr = self.arr[mask]
        self.idx.clear(); self.addr_map.clear()
        for i,ut in enumerate(self.arr):
            uid = f"{ut['tx_id']}:{int(ut['out_idx'])}"
            self.idx[uid] = i
            self.addr_map.setdefault(ut['recipient'],[]).append(i)


# ────────────────────────────────────────────────────────────────────────────────
# Block & adaptive Proof-of-Work
# ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Block:
    index:       int
    prev_hash:   str
    txs:         List[Transaction]
    timestamp:   float = field(default_factory=time.time)
    nonce:       int = 0
    merkle_root: str = ""
    hash:        str = ""

    def compute_hash(self) -> str:
        hdr = json.dumps({
            'idx': self.index,
            'prev': self.prev_hash,
            'root': self.merkle_root,
            'ts': self.timestamp,
            'nonce': self.nonce
        }, sort_keys=True)
        return Crypto.sha256(hdr)

    def seal(self):
        # simple Merkle root
        leaves = [tx.hash() for tx in self.txs]
        if not leaves: self.merkle_root=""
        else:
            if len(leaves)%2: leaves.append(leaves[-1])
            while len(leaves)>1:
                leaves = [Crypto.sha256(leaves[i]+leaves[i+1])
                          for i in range(0,len(leaves),2)]
                if len(leaves)%2 and len(leaves)>1:
                    leaves.append(leaves[-1])
            self.merkle_root = leaves[0]
        self.hash = self.compute_hash()


class PoW:
    def __init__(self, difficulty:int=INITIAL_DIFFICULTY):
        self.diff = difficulty

    def mine(self, block: Block) -> None:
        block.seal()
        target = '0'*self.diff
        while not block.hash.startswith(target):
            block.nonce += 1
            block.hash = block.compute_hash()


# ────────────────────────────────────────────────────────────────────────────────
# Blockchain Core with dynamic difficulty and pruning
# ────────────────────────────────────────────────────────────────────────────────
class Blockchain:
    def __init__(self):
        self.diff    = INITIAL_DIFFICULTY
        self.reward  = REWARD
        self.chain: List[Block] = []
        self.utxo    = UTXOSet()
        self.mempool = Mempool()
        self.pow     = PoW(self.diff)
        self.lock    = threading.RLock()
        self._create_genesis()

    def _create_genesis(self):
        cb = Transaction("coinbase","miner0",self.reward,tx_type=TxType.COINBASE)
        blk = Block(0, "0"*64, [cb])
        self.pow.mine(blk)
        self.chain.append(blk)
        # update UTXO
        for idx,tx in enumerate(blk.txs):
            self.utxo.add(tx.tx_id, 0, tx.recipient, tx.amount, 0)

    def _retarget(self):
        if len(self.chain)%RETARGET_INTERVAL!=0: return
        span = self.chain[-1].timestamp - self.chain[-RETARGET_INTERVAL].timestamp
        ideal = TARGET_BLOCK_TIME * RETARGET_INTERVAL
        adj   = ideal/span
        # cap adjustment to ±25%
        adj = max(0.75, min(1.25, adj))
        self.diff = max(1, int(self.diff * adj))
        self.pow.diff = self.diff

    def create_tx(self, sender:str, recipient:str, amount:float, fee:float=0.001) -> Optional[Transaction]:
        with self.lock:
            bal = self.utxo.balance(sender)
            if bal < amount+fee: return None
            inputs, tot = self.utxo.select(sender, amount+fee)
            change = tot - amount - fee
            nonce  = self.mempool.nonces.get(sender,0)
            tx = Transaction(sender,recipient,amount,nonce,fee)
            tx.inputs = [f"{u['tx_id']}:{u['out_idx']}" for u in inputs]
            outs = [{'recipient':recipient,'amount':amount}]
            if change>0:
                outs.append({'recipient':sender,'amount':change})
            tx.outputs = outs
            return tx

    def submit_tx(self, tx:Transaction, priv_key:str, pub_key:str) -> bool:
        # sign and verify in one go
        tx.sign(priv_key)
        if not tx.verify(pub_key): return False
        return self.mempool.add(tx)

    def mine_pending(self, miner_addr:str="miner0", batch:int=500):
        with self.lock:
            txs = self.mempool.take(batch)
            cb  = Transaction("coinbase",miner_addr,self.reward,tx_type=TxType.COINBASE)
            block = Block(len(self.chain), self.chain[-1].hash, txs+[cb])
            self.pow.mine(block)
            # append and update
            self.chain.append(block)
            h = block.hash
            height = block.index
            for tx in block.txs:
                # spend inputs
                if tx.tx_type is not TxType.COINBASE:
                    for inp in tx.inputs:
                        txid,oi = inp.split(":")
                        self.utxo.spend(txid,int(oi))
                # add outputs
                for idx,o in enumerate(tx.outputs):
                    self.utxo.add(tx.tx_id, idx, o['recipient'], o['amount'], height)
            self.mempool.purge([t.tx_id for t in txs])
            # retarget difficulty & prune UTXO periodically
            self._retarget()
            if height % UTXO_RESET_PERIOD == 0:
                self.utxo.prune()
            print(f"Mined block {height}, hash={h[:16]}, diff={self.diff}")

    def balance(self, addr:str) -> float:
        return self.utxo.balance(addr)


# ────────────────────────────────────────────────────────────────────────────────
# Example usage
# ────────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chain = Blockchain()
    # Simulate keypair (fallback)
    priv = uuid.uuid4().hex
    pub  = hashlib.sha256(priv.encode()).hexdigest()[:42]

    # Mine genesis reward to miner1
    chain.mine_pending("miner1")

    # Create & submit a transfer
    tx = chain.create_tx("miner1", pub, 5.0, fee=0.001)
    if tx and chain.submit_tx(tx, priv, pub):
        print("Transaction accepted")

    # Mine again to include it
    chain.mine_pending("miner1")

    print("miner1 balance:", chain.balance("miner1"))
    print("recipient balance:", chain.balance(pub))
