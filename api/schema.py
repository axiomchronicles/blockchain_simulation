from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from simulation.crypto import TxType

class CreateWalletResponse(BaseModel):
    wallet_id: str
    addresses: List[str]

class WalletResponse(BaseModel):
    wallet_id: str
    addresses: List[str]

class CreateTransactionRequest(BaseModel):
    sender: str
    recipient: str
    amount: float
    nonce: Optional[int] = 0
    gas_price: Optional[float] = 1.0
    gas_limit: Optional[int] = 21000
    tx_type: Optional[str] = "TRANSFER"  # Use TxType names

    @validator('tx_type')
    def validate_tx_type(cls, v):
        if v not in TxType.__members__:
            raise ValueError(f"tx_type must be one of {list(TxType.__members__.keys())}")
        return v

class CreateTransactionResponse(BaseModel):
    tx_id: str
    signature: str

class TransactionResponse(BaseModel):
    tx_id: str
    sender: str
    recipient: str
    amount: float
    nonce: int
    gas_price: float
    gas_limit: int
    tx_type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: float
    signature: Optional[str]
    shard_id: int
    contract_code: Optional[List[int]] = None

class VerifyTransactionResponse(BaseModel):
    tx_id: str
    valid: bool

class ExecuteContractRequest(BaseModel):
    contract_code: List[int]
    gas_limit: Optional[int] = 1000000

class ExecuteContractResponse(BaseModel):
    stack: List[Any]
    gas_used: int

class SwitchConsensusRequest(BaseModel):
    consensus_type: str

class SwitchConsensusResponse(BaseModel):
    message: str

class ValidateBlockRequest(BaseModel):
    block: Dict[str, Any]
    chain_state: Optional[Dict[str, Any]] = {}

class ValidateBlockResponse(BaseModel):
    valid: bool
