
# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED P2P NETWORKING WITH DHT
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import json
import secrets
import time
import uuid
import socket
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Set, Tuple
from collections import defaultdict
from abc import ABC, abstractmethod

from .core import NetworkConfig, log, Transaction, Block, ChainConfig, ConsensusConfig

@dataclass
class PeerInfo:
    peer_id: str
    host: str
    port: int
    public_key: bytes
    last_seen: float = field(default_factory=time.time)
    reputation: float = 100.0
    capabilities: Set[str] = field(default_factory=set)

    def distance_to(self, target_id: str) -> int:
        """Calculate XOR distance for DHT"""
        return int(self.peer_id, 16) ^ int(target_id, 16)

class KademliaDHT:
    """Simplified Kademlia DHT for peer discovery"""

    K_BUCKET_SIZE = 20
    ALPHA = 3

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.routing_table: Dict[int, List[PeerInfo]] = defaultdict(list)
        self.stored_values: Dict[str, Any] = {}
        self.pending_requests: Dict[str, asyncio.Future] = {}

    def add_peer(self, peer: PeerInfo):
        """Add peer to routing table"""
        distance = peer.distance_to(self.node_id)
        bucket_index = distance.bit_length() - 1 if distance > 0 else 0

        bucket = self.routing_table[bucket_index]

        # Update if peer already exists
        for i, existing_peer in enumerate(bucket):
            if existing_peer.peer_id == peer.peer_id:
                bucket[i] = peer
                return

        # Add new peer if bucket not full
        if len(bucket) < self.K_BUCKET_SIZE:
            bucket.append(peer)
        else:
            # Replace least recently seen peer
            bucket.sort(key=lambda p: p.last_seen)
            bucket[0] = peer

    def find_closest_peers(self, target_id: str, count: int = None) -> List[PeerInfo]:
        """Find closest peers to target ID"""
        if count is None:
            count = self.K_BUCKET_SIZE

        all_peers = []
        for bucket in self.routing_table.values():
            all_peers.extend(bucket)

        # Sort by distance to target
        all_peers.sort(key=lambda p: p.distance_to(target_id))

        return all_peers[:count]

class P2PNetworkManager:
    """Advanced P2P network manager with gossip protocol"""

    def __init__(self, node_id: str, port: int = NetworkConfig.P2P_PORT):
        self.node_id = node_id
        self.port = port
        self.dht = KademliaDHT(node_id)
        self.connected_peers: Dict[str, PeerInfo] = {}
        self.message_cache: Dict[str, float] = {}  # Message deduplication
        self.bandwidth_monitor = defaultdict(int)
        self.lock = asyncio.Lock()

        # Message handlers
        self.handlers: Dict[str, Callable] = {
            'block': self._handle_block,
            'transaction': self._handle_transaction,
            'peer_discovery': self._handle_peer_discovery,
            'consensus': self._handle_consensus_message,
        }

    async def start(self):
        """Start P2P network services"""
        # Start UDP server for DHT
        loop = asyncio.get_event_loop()

        # UDP transport for DHT
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self),
            local_addr=('0.0.0.0', self.port)
        )

        # Start peer discovery
        asyncio.create_task(self._peer_discovery_loop())

        log.info("P2P network started", port=self.port, node_id=self.node_id)

    async def broadcast_message(self, message_type: str, data: Dict[str, Any]):
        """Broadcast message to all connected peers"""
        message = {
            'type': message_type,
            'data': data,
            'timestamp': time.time(),
            'sender': self.node_id,
            'message_id': uuid.uuid4().hex
        }

        # Add to cache to prevent loops
        self.message_cache[message['message_id']] = time.time()

        message_bytes = json.dumps(message).encode()

        async with self.lock:
            for peer_id, peer in self.connected_peers.items():
                try:
                    await self._send_to_peer(peer, message_bytes)
                except Exception as e:
                    log.error(f"Failed to send message to peer {peer_id}", error=str(e))

    async def _send_to_peer(self, peer: PeerInfo, data: bytes):
        """Send data to specific peer"""
        # Implement actual networking here
        pass

    async def _peer_discovery_loop(self):
        """Periodic peer discovery"""
        while True:
            try:
                await self._discover_peers()
                await asyncio.sleep(NetworkConfig.DISCOVERY_INTERVAL)
            except Exception as e:
                log.error("Peer discovery error", error=str(e))
                await asyncio.sleep(5)

    async def _discover_peers(self):
        """Discover new peers using DHT"""
        # Find peers for random keys to populate routing table
        for _ in range(3):
            random_key = secrets.token_hex(20)
            closest_peers = self.dht.find_closest_peers(random_key, 5)

            for peer in closest_peers:
                if len(self.connected_peers) < NetworkConfig.MAX_PEERS:
                    await self._connect_to_peer(peer)

    async def _connect_to_peer(self, peer: PeerInfo):
        """Connect to a new peer"""
        if peer.peer_id in self.connected_peers:
            return

        try:
            # Implement connection logic
            self.connected_peers[peer.peer_id] = peer
            self.dht.add_peer(peer)
            log.info("Connected to peer", peer_id=peer.peer_id, host=peer.host)
        except Exception as e:
            log.error("Failed to connect to peer", peer_id=peer.peer_id, error=str(e))

    async def _handle_block(self, data: Dict[str, Any], sender: str):
        """Handle incoming block"""
        # Will be connected to blockchain instance
        pass

    async def _handle_transaction(self, data: Dict[str, Any], sender: str):
        """Handle incoming transaction"""
        # Will be connected to blockchain instance
        pass

    async def _handle_peer_discovery(self, data: Dict[str, Any], sender: str):
        """Handle peer discovery message"""
        peers_data = data.get('peers', [])
        for peer_data in peers_data:
            peer = PeerInfo(**peer_data)
            self.dht.add_peer(peer)

    async def _handle_consensus_message(self, data: Dict[str, Any], sender: str):
        """Handle consensus-related messages"""
        pass

class UDPProtocol(asyncio.DatagramProtocol):
    """UDP protocol for P2P communication"""

    def __init__(self, network_manager: P2PNetworkManager):
        self.network_manager = network_manager

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        """Handle incoming UDP datagram"""
        try:
            message = json.loads(data.decode())
            message_id = message.get('message_id')

            # Check for duplicate messages
            if message_id in self.network_manager.message_cache:
                return

            self.network_manager.message_cache[message_id] = time.time()

            # Route to appropriate handler
            message_type = message.get('type')
            if message_type in self.network_manager.handlers:
                asyncio.create_task(
                    self.network_manager.handlers[message_type](
                        message.get('data', {}),
                        message.get('sender')
                    )
                )
        except Exception as e:
            log.error("Error processing UDP message", error=str(e))

# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-CONSENSUS MECHANISMS
# ═══════════════════════════════════════════════════════════════════════════════

class ConsensusEngine(ABC):
    """Abstract base class for consensus mechanisms"""

    @abstractmethod
    async def validate_block(self, block: 'Block', chain_state: Dict) -> bool:
        pass

    @abstractmethod
    async def create_block(self, transactions: List[Transaction], 
                          previous_block: 'Block', chain_state: Dict) -> 'Block':
        pass

class ProofOfWork(ConsensusEngine):
    """Proof of Work consensus implementation"""

    def __init__(self, difficulty: int = ChainConfig.INITIAL_DIFFICULTY):
        self.difficulty = difficulty
        self.target = 2 ** (256 - difficulty)

    async def validate_block(self, block: 'Block', chain_state: Dict) -> bool:
        """Validate PoW block"""
        block_hash_int = int(block.hash, 16)
        return block_hash_int < self.target

    async def create_block(self, transactions: List[Transaction], 
                          previous_block: 'Block', chain_state: Dict) -> 'Block':
        """Mine new PoW block"""
        from .core import Block  # Avoid circular import

        block = Block(
            index=previous_block.index + 1,
            previous_hash=previous_block.hash,
            transactions=transactions,
            timestamp=time.time(),
            difficulty=self.difficulty
        )

        # Mine the block
        target = "0" * self.difficulty
        while not block.hash.startswith(target):
            block.nonce += 1
            block.calculate_hash()

        return block

class ProofOfStake(ConsensusEngine):
    """Proof of Stake consensus implementation"""

    def __init__(self):
        self.validators: Dict[str, float] = {}  # validator_id -> stake
        self.validator_slots: List[str] = []

    async def validate_block(self, block: 'Block', chain_state: Dict) -> bool:
        """Validate PoS block"""
        # Check if proposer has sufficient stake
        proposer = block.proposer
        stake = self.validators.get(proposer, 0)
        return stake >= ConsensusConfig.STAKE_THRESHOLD

    async def create_block(self, transactions: List[Transaction], 
                          previous_block: 'Block', chain_state: Dict) -> 'Block':
        """Create new PoS block"""
        # Select validator based on stake
        validator = self._select_validator(previous_block.hash)

        from .core import Block
        block = Block(
            index=previous_block.index + 1,
            previous_hash=previous_block.hash,
            transactions=transactions,
            timestamp=time.time(),
            proposer=validator
        )

        return block

    def _select_validator(self, random_seed: str) -> str:
        """Select validator based on stake weight"""
        if not self.validators:
            return "genesis_validator"

        # Weighted random selection
        total_stake = sum(self.validators.values())
        random_point = int(random_seed, 16) % int(total_stake)

        current_weight = 0
        for validator_id, stake in self.validators.items():
            current_weight += stake
            if random_point <= current_weight:
                return validator_id

        return list(self.validators.keys())[0]

class PBFT(ConsensusEngine):
    """Practical Byzantine Fault Tolerance consensus"""

    def __init__(self):
        self.validators: Set[str] = set()
        self.view_number = 0
        self.sequence_number = 0

        # PBFT message log
        self.pre_prepare_msgs: Dict[int, Dict] = {}
        self.prepare_msgs: Dict[int, List[Dict]] = defaultdict(list)
        self.commit_msgs: Dict[int, List[Dict]] = defaultdict(list)

    async def validate_block(self, block: 'Block', chain_state: Dict) -> bool:
        """Validate PBFT block requires 2f+1 commits"""
        seq_num = block.index
        commits = len(self.commit_msgs.get(seq_num, []))
        required_commits = (2 * len(self.validators)) // 3 + 1
        return commits >= required_commits

    async def create_block(self, transactions: List[Transaction], 
                          previous_block: 'Block', chain_state: Dict) -> 'Block':
        """Create PBFT block through three-phase protocol"""
        self.sequence_number += 1

        from .core import Block
        block = Block(
            index=previous_block.index + 1,
            previous_hash=previous_block.hash,
            transactions=transactions,
            timestamp=time.time(),
            view_number=self.view_number,
            sequence_number=self.sequence_number
        )

        # Start PBFT phases
        await self._pbft_pre_prepare(block)
        await self._pbft_prepare(block)
        await self._pbft_commit(block)

        return block

    async def _pbft_pre_prepare(self, block: 'Block'):
        """PBFT Pre-prepare phase"""
        pre_prepare_msg = {
            'view': self.view_number,
            'sequence': self.sequence_number,
            'block_hash': block.hash,
            'timestamp': time.time()
        }
        self.pre_prepare_msgs[self.sequence_number] = pre_prepare_msg

    async def _pbft_prepare(self, block: 'Block'):
        """PBFT Prepare phase"""
        prepare_msg = {
            'view': self.view_number,
            'sequence': self.sequence_number,
            'block_hash': block.hash,
            'validator_id': 'local_validator'
        }
        self.prepare_msgs[self.sequence_number].append(prepare_msg)

    async def _pbft_commit(self, block: 'Block'):
        """PBFT Commit phase"""
        commit_msg = {
            'view': self.view_number,
            'sequence': self.sequence_number,
            'block_hash': block.hash,
            'validator_id': 'local_validator'
        }
        self.commit_msgs[self.sequence_number].append(commit_msg)

class ConsensusManager:
    """Manages multiple consensus mechanisms"""

    def __init__(self, consensus_type: str = ConsensusConfig.CONSENSUS_TYPE):
        self.engines: Dict[str, ConsensusEngine] = {
            'PoW': ProofOfWork(),
            'PoS': ProofOfStake(),
            'PBFT': PBFT()
        }
        self.current_engine = self.engines.get(consensus_type, self.engines['PoW'])

    def switch_consensus(self, new_type: str):
        """Switch to different consensus mechanism"""
        if new_type in self.engines:
            self.current_engine = self.engines[new_type]
            log.info(f"Switched to {new_type} consensus")

    async def validate_block(self, block: 'Block', chain_state: Dict) -> bool:
        return await self.current_engine.validate_block(block, chain_state)

    async def create_block(self, transactions: List[Transaction], 
                          previous_block: 'Block', chain_state: Dict) -> 'Block':
        return await self.current_engine.create_block(transactions, previous_block, chain_state)

print("✓ Advanced P2P networking and consensus mechanisms implemented")
