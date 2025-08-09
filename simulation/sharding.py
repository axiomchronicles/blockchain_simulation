
# ═══════════════════════════════════════════════════════════════════════════════
# SHARDING IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


import asyncio
import json
import secrets
import time
import uuid
import socket
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Set, Tuple
import threading
from collections import defaultdict, deque
from abc import ABC, abstractmethod

from .networking import P2PNetworkManager

from .core import log, Transaction, Block, ChainConfig, CryptoManager, TxType, AdvancedMempool, UTXOSetAdvanced, ThreadPoolExecutor, ConsensusManager, WalletManager, BlockchainMetrics, SmartContractVM

@dataclass
class Shard:
    shard_id: int
    validator_set: Set[str]
    current_state: Dict[str, Any]
    pending_transactions: List[Transaction]
    last_block_hash: str

class ShardManager:
    """Manages multiple shards for scalability"""

    def __init__(self, num_shards: int = 4):
        self.num_shards = num_shards
        self.shards: Dict[int, Shard] = {}
        self.cross_shard_queue: deque = deque()

        for i in range(num_shards):
            self.shards[i] = Shard(
                shard_id=i,
                validator_set=set(),
                current_state={},
                pending_transactions=[],
                last_block_hash=""
            )

    def get_shard_for_address(self, address: str) -> int:
        """Determine which shard handles this address"""
        return int(CryptoManager.hash_data(address), 16) % self.num_shards

    def add_transaction_to_shard(self, transaction: Transaction):
        """Route transaction to appropriate shard"""
        shard_id = self.get_shard_for_address(transaction.sender)
        transaction.shard_id = shard_id

        if transaction.recipient and self.get_shard_for_address(transaction.recipient) != shard_id:
            # Cross-shard transaction
            self.cross_shard_queue.append(transaction)
        else:
            # Intra-shard transaction
            self.shards[shard_id].pending_transactions.append(transaction)

    def process_cross_shard_transactions(self):
        """Process cross-shard transactions using atomic commits"""
        while self.cross_shard_queue:
            tx = self.cross_shard_queue.popleft()
            sender_shard = self.get_shard_for_address(tx.sender)
            recipient_shard = self.get_shard_for_address(tx.recipient)

            # Two-phase commit for cross-shard tx
            if self._prepare_cross_shard_tx(tx, sender_shard, recipient_shard):
                self._commit_cross_shard_tx(tx, sender_shard, recipient_shard)
            else:
                self._abort_cross_shard_tx(tx, sender_shard, recipient_shard)

    def _prepare_cross_shard_tx(self, tx: Transaction, sender_shard: int, recipient_shard: int) -> bool:
        """Prepare phase of cross-shard transaction"""
        # Check if sender has sufficient balance
        sender_balance = self._get_balance_in_shard(tx.sender, sender_shard)
        return sender_balance >= tx.amount + tx.calculate_fee()

    def _commit_cross_shard_tx(self, tx: Transaction, sender_shard: int, recipient_shard: int):
        """Commit phase of cross-shard transaction"""
        # Debit from sender shard
        self.shards[sender_shard].pending_transactions.append(
            Transaction(
                sender=tx.sender,
                recipient="cross_shard_bridge",
                amount=tx.amount + tx.calculate_fee(),
                nonce=tx.nonce,
                gas_price=tx.gas_price,
                gas_limit=tx.gas_limit,
                tx_type=TxType.TRANSFER
            )
        )

        # Credit to recipient shard
        self.shards[recipient_shard].pending_transactions.append(
            Transaction(
                sender="cross_shard_bridge",
                recipient=tx.recipient,
                amount=tx.amount,
                nonce=0,
                gas_price=0,
                gas_limit=0,
                tx_type=TxType.TRANSFER
            )
        )

    def _abort_cross_shard_tx(self, tx: Transaction, sender_shard: int, recipient_shard: int):
        """Abort cross-shard transaction"""
        log.info("Cross-shard transaction aborted", tx_id=tx.tx_id)

    def _get_balance_in_shard(self, address: str, shard_id: int) -> float:
        """Get balance for address in specific shard"""
        return self.shards[shard_id].current_state.get(f"balance:{address}", 0.0)

# ═══════════════════════════════════════════════════════════════════════════════
# GOVERNANCE AND VOTING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Proposal:
    proposal_id: str
    title: str
    description: str
    proposer: str
    voting_start: float
    voting_end: float
    proposal_type: str  # 'parameter_change', 'upgrade', 'treasury'
    parameters: Dict[str, Any]
    votes_for: Dict[str, float] = field(default_factory=dict)
    votes_against: Dict[str, float] = field(default_factory=dict)
    status: str = "active"  # active, passed, rejected, expired

    def get_total_votes_for(self) -> float:
        return sum(self.votes_for.values())

    def get_total_votes_against(self) -> float:
        return sum(self.votes_against.values())

    def is_passed(self, total_staked: float) -> bool:
        """Check if proposal has passed based on voting rules"""
        total_votes = self.get_total_votes_for() + self.get_total_votes_against()
        if total_votes < total_staked * 0.1:  # Minimum 10% participation
            return False

        return self.get_total_votes_for() > self.get_total_votes_against() * 1.5

class GovernanceSystem:
    """On-chain governance and voting system"""

    def __init__(self):
        self.proposals: Dict[str, Proposal] = {}
        self.treasury_balance: float = 0.0
        self.staking_balances: Dict[str, float] = {}
        self.voting_power: Dict[str, float] = {}

    def create_proposal(self, proposer: str, title: str, description: str, 
                       proposal_type: str, parameters: Dict[str, Any]) -> str:
        """Create new governance proposal"""
        # Check if proposer has sufficient stake
        if self.staking_balances.get(proposer, 0) < 1000:  # Minimum stake to propose
            raise Exception("Insufficient stake to create proposal")

        proposal_id = uuid.uuid4().hex
        voting_period = 7 * 24 * 3600  # 7 days

        proposal = Proposal(
            proposal_id=proposal_id,
            title=title,
            description=description,
            proposer=proposer,
            voting_start=time.time(),
            voting_end=time.time() + voting_period,
            proposal_type=proposal_type,
            parameters=parameters
        )

        self.proposals[proposal_id] = proposal
        log.info("New proposal created", proposal_id=proposal_id, title=title)

        return proposal_id

    def vote(self, proposal_id: str, voter: str, vote: bool, weight: float = None):
        """Cast vote on proposal"""
        if proposal_id not in self.proposals:
            raise Exception("Proposal not found")

        proposal = self.proposals[proposal_id]

        # Check voting period
        if time.time() > proposal.voting_end:
            raise Exception("Voting period has ended")

        # Calculate voting power
        if weight is None:
            weight = self.voting_power.get(voter, self.staking_balances.get(voter, 0))

        if vote:
            proposal.votes_for[voter] = weight
            proposal.votes_against.pop(voter, None)  # Remove any previous opposite vote
        else:
            proposal.votes_against[voter] = weight
            proposal.votes_for.pop(voter, None)

        log.info("Vote cast", proposal_id=proposal_id, voter=voter, vote=vote, weight=weight)

    def finalize_proposal(self, proposal_id: str) -> bool:
        """Finalize proposal after voting period"""
        if proposal_id not in self.proposals:
            return False

        proposal = self.proposals[proposal_id]

        if time.time() < proposal.voting_end:
            return False  # Voting still active

        total_staked = sum(self.staking_balances.values())

        if proposal.is_passed(total_staked):
            proposal.status = "passed"
            self._execute_proposal(proposal)
            log.info("Proposal passed", proposal_id=proposal_id)
            return True
        else:
            proposal.status = "rejected"
            log.info("Proposal rejected", proposal_id=proposal_id)
            return False

    def _execute_proposal(self, proposal: Proposal):
        """Execute approved proposal"""
        if proposal.proposal_type == "parameter_change":
            # Update blockchain parameters
            for param, value in proposal.parameters.items():
                self._update_parameter(param, value)

        elif proposal.proposal_type == "treasury":
            # Treasury spending proposal
            recipient = proposal.parameters.get("recipient")
            amount = proposal.parameters.get("amount", 0)

            if recipient and amount <= self.treasury_balance:
                self.treasury_balance -= amount
                # Transfer funds (would integrate with transaction system)
                log.info("Treasury funds transferred", recipient=recipient, amount=amount)

    def _update_parameter(self, param: str, value: Any):
        """Update blockchain parameter"""
        # This would integrate with the main blockchain configuration
        log.info("Parameter updated", param=param, value=value)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BLOCKCHAIN IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Block:
    index: int
    previous_hash: str
    transactions: List[Transaction]
    timestamp: float = field(default_factory=time.time)
    nonce: int = 0
    merkle_root: str = ""
    hash: str = ""
    difficulty: int = ChainConfig.INITIAL_DIFFICULTY
    gas_used: int = 0
    gas_limit: int = ChainConfig.GAS_LIMIT
    proposer: str = ""
    shard_id: int = 0

    def calculate_hash(self):
        """Calculate block hash"""
        header_data = {
            'index': self.index,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp,
            'nonce': self.nonce,
            'difficulty': self.difficulty,
            'gas_used': self.gas_used,
            'proposer': self.proposer,
            'shard_id': self.shard_id
        }

        self.hash = CryptoManager.hash_data(json.dumps(header_data, sort_keys=True))
        return self.hash

    def calculate_merkle_root(self):
        """Calculate Merkle root of transactions"""
        if not self.transactions:
            self.merkle_root = ""
            return

        tx_hashes = [tx.get_hash() for tx in self.transactions]
        self.merkle_root = CryptoManager.merkle_root(tx_hashes)

class UltimateBlockchain:
    """The ultimate blockchain implementation with all advanced features"""

    def __init__(self, node_id: str = None):
        self.node_id = node_id or uuid.uuid4().hex

        # Core components
        self.chain: List[Block] = []
        self.block_index: Dict[str, Block] = {}  # hash -> block
        self.height_index: Dict[int, Block] = {}  # height -> block

        # State management
        self.utxo_set = UTXOSetAdvanced()
        self.account_state: Dict[str, Dict[str, Any]] = {}
        self.contract_storage: Dict[str, Dict[str, Any]] = {}

        # Transaction processing
        self.mempool = AdvancedMempool()
        self.tx_pool = ThreadPoolExecutor(max_workers=4)

        # Advanced features
        self.shard_manager = ShardManager()
        self.consensus_manager = ConsensusManager()
        self.governance = GovernanceSystem()
        self.wallet_manager = WalletManager()
        self.vm = SmartContractVM()

        # Networking
        self.p2p_manager = P2PNetworkManager(self.node_id)

        # Performance monitoring
        self.metrics = BlockchainMetrics()

        # Threading and async
        self.lock = threading.RLock()
        self.event_loop = None

        # Initialize genesis block
        self._create_genesis_block()

        log.info("Ultimate blockchain initialized", node_id=self.node_id)

    def _create_genesis_block(self):
        """Create the genesis block"""
        genesis_tx = Transaction(
            sender="genesis",
            recipient="treasury",
            amount=1000000.0,  # Initial supply
            nonce=0,
            gas_price=0,
            gas_limit=0,
            tx_type=TxType.COINBASE
        )

        genesis_block = Block(
            index=0,
            previous_hash="0" * 64,
            transactions=[genesis_tx],
            proposer="genesis"
        )

        genesis_block.calculate_merkle_root()
        genesis_block.calculate_hash()

        self.chain.append(genesis_block)
        self.block_index[genesis_block.hash] = genesis_block
        self.height_index[0] = genesis_block

        # Initialize treasury
        self.governance.treasury_balance = genesis_tx.amount

    async def start(self):
        """Start the blockchain node"""
        # Start P2P networking
        await self.p2p_manager.start()

        # Start background tasks
        asyncio.create_task(self._mining_loop())
        asyncio.create_task(self._governance_loop())
        asyncio.create_task(self._metrics_loop())

        log.info("Blockchain node started")

    def create_transaction(self, sender: str, recipient: str, amount: float, 
                          gas_price: float = ChainConfig.MIN_GAS_PRICE,
                          gas_limit: int = 21000,
                          tx_type: TxType = TxType.TRANSFER,
                          data: Dict[str, Any] = None) -> Transaction:
        """Create new transaction"""
        wallet_info = self.wallet_manager.get_wallet_for_address(sender)
        if not wallet_info:
            raise Exception("Wallet not found for sender")

        wallet, account_index = wallet_info
        nonce = self._get_nonce(sender)

        tx = Transaction(
            sender=sender,
            recipient=recipient,
            amount=amount,
            nonce=nonce,
            gas_price=gas_price,
            gas_limit=gas_limit,
            tx_type=tx_type,
            data=data
        )

        # Sign transaction
        private_key = wallet.accounts[account_index]['private']
        tx.sign(private_key)

        return tx

    def submit_transaction(self, transaction: Transaction) -> bool:
        """Submit transaction to mempool"""
        # Validate transaction
        if not self._validate_transaction(transaction):
            return False

        # Add to mempool
        success = self.mempool.add_transaction(transaction)

        if success:
            # Broadcast to network
            asyncio.create_task(self.p2p_manager.broadcast_message(
                'transaction', 
                {'transaction': asdict(transaction)}
            ))

            self.metrics.transactions_submitted += 1
            log.info("Transaction submitted", tx_id=transaction.tx_id)

        return success

    def _validate_transaction(self, tx: Transaction) -> bool:
        """Validate transaction"""
        # Basic validation
        if tx.amount < 0 or tx.gas_price < 0 or tx.gas_limit < 0:
            return False

        # Check balance
        balance = self._get_balance(tx.sender)
        total_cost = tx.amount + (tx.gas_price * tx.gas_limit)
        if balance < total_cost:
            return False

        # Check nonce
        expected_nonce = self._get_nonce(tx.sender)
        if tx.nonce != expected_nonce:
            return False

        # Verify signature
        wallet_info = self.wallet_manager.get_wallet_for_address(tx.sender)
        if wallet_info:
            wallet, account_index = wallet_info
            public_key = wallet.accounts[account_index]['public']
            return tx.verify(public_key)

        return False

    async def _mining_loop(self):
        """Main mining/block production loop"""
        while True:
            try:
                await self._produce_block()
                await asyncio.sleep(1)  # Check for new blocks every second
            except Exception as e:
                log.error("Mining loop error", error=str(e))
                await asyncio.sleep(5)

    async def _produce_block(self):
        """Produce new block"""
        with self.lock:
            if len(self.mempool.transactions) == 0:
                return  # No transactions to process

            # Get transactions from mempool
            transactions = self.mempool.get_transactions_for_block()

            if not transactions:
                return

            # Get previous block
            previous_block = self.chain[-1]

            # Create new block using consensus
            new_block = await self.consensus_manager.create_block(
                transactions, 
                previous_block, 
                self._get_chain_state()
            )

            # Validate and add block
            if await self._add_block(new_block):
                # Remove transactions from mempool
                tx_ids = [tx.tx_id for tx in transactions]
                self.mempool.remove_transactions(tx_ids)

                # Broadcast block
                await self.p2p_manager.broadcast_message(
                    'block', 
                    {'block': asdict(new_block)}
                )

                self.metrics.blocks_mined += 1
                log.info("New block mined", 
                        height=new_block.index, 
                        hash=new_block.hash[:16],
                        tx_count=len(new_block.transactions))

    async def _add_block(self, block: Block) -> bool:
        """Add block to chain after validation"""
        # Validate block
        if not await self.consensus_manager.validate_block(block, self._get_chain_state()):
            return False

        # Process transactions
        for tx in block.transactions:
            await self._process_transaction(tx, block)

        # Add to chain
        self.chain.append(block)
        self.block_index[block.hash] = block
        self.height_index[block.index] = block

        return True

    async def _process_transaction(self, tx: Transaction, block: Block):
        """Process individual transaction"""
        if tx.tx_type == TxType.TRANSFER:
            self._transfer_funds(tx.sender, tx.recipient, tx.amount)

        elif tx.tx_type == TxType.COINBASE:
            self._mint_coins(tx.recipient, tx.amount)

        elif tx.tx_type == TxType.CONTRACT_CREATE:
            await self._create_contract(tx)

        elif tx.tx_type == TxType.CONTRACT_CALL:
            await self._call_contract(tx)

        # Update account nonce
        self._increment_nonce(tx.sender)

    def _get_chain_state(self) -> Dict[str, Any]:
        """Get current chain state for consensus"""
        return {
            'height': len(self.chain) - 1,
            'total_supply': sum(self.governance.staking_balances.values()),
            'difficulty': getattr(self.consensus_manager.current_engine, 'difficulty', 0)
        }

    async def _governance_loop(self):
        """Governance processing loop"""
        while True:
            try:
                # Check for proposals to finalize
                for proposal_id in list(self.governance.proposals.keys()):
                    self.governance.finalize_proposal(proposal_id)

                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                log.error("Governance loop error", error=str(e))
                await asyncio.sleep(60)

    async def _metrics_loop(self):
        """Metrics collection loop"""
        while True:
            try:
                self.metrics.update_metrics(self)
                await asyncio.sleep(30)  # Update every 30 seconds
            except Exception as e:
                log.error("Metrics loop error", error=str(e))
                await asyncio.sleep(30)

print("✓ Sharding, governance, and main blockchain implementation completed")
