import time
import traceback
from typing import Dict, Any, List

from aquilify.responses import JsonResponse as JSONResponse
from aquilify.wrappers import Request
from aquilify.core.status import HTTP_400_BAD_REQUEST, HTTP_404_NOT_FOUND

from simulation.crypto import (
    WalletManager, Transaction, TxType,
    SmartContractVM, VMState
)

from simulation.networking import ConsensusManager, Block
from .helpers import serialize_transaction

from .schema import (
    CreateTransactionRequest, CreateTransactionResponse,
    CreateWalletResponse, SwitchConsensusRequest,
    SwitchConsensusResponse, ExecuteContractRequest,
    ExecuteContractResponse, WalletResponse,
    TransactionResponse, VerifyTransactionResponse, ValidateBlockRequest, ValidateBlockResponse
)

from .database import tx_collection, blocks_collection, wallets_collection

# Managers / singletons
wallet_manager = WalletManager()
smart_contract_vm = SmartContractVM()
consensus_manager = ConsensusManager()


# -----------------------
# Helpers: dict -> Transaction / Block
# -----------------------
def dict_to_transaction(tx_dict: Dict[str, Any]) -> Transaction:
    """
    Convert a transaction dict (from JSON / DB) into a Transaction instance.
    Handles tx_type provided as name or numeric value and signature hex string.
    """
    # tx_type handling: accept name (str) or numeric
    tx_type_val = tx_dict.get("tx_type")
    if isinstance(tx_type_val, str):
        # either 'TRANSFER' or numeric string like '0'
        if tx_type_val.isdigit():
            tx_type = TxType(int(tx_type_val))
        else:
            # try name
            tx_type = TxType[tx_type_val]
    elif isinstance(tx_type_val, (int, float)):
        tx_type = TxType(int(tx_type_val))
    else:
        tx_type = TxType.TRANSFER

    # signature: allow hex string or bytes
    sig = tx_dict.get("signature", b"")
    if isinstance(sig, str):
        try:
            signature = bytes.fromhex(sig)
        except Exception:
            signature = sig.encode()  # fallback
    elif isinstance(sig, (bytes, bytearray)):
        signature = bytes(sig)
    else:
        signature = b""

    return Transaction(
        sender=tx_dict.get("sender", ""),
        recipient=tx_dict.get("recipient", ""),
        amount=float(tx_dict.get("amount", 0.0)),
        nonce=int(tx_dict.get("nonce", 0)),
        gas_price=float(tx_dict.get("gas_price", 0.0)),
        gas_limit=int(tx_dict.get("gas_limit", 0)),
        tx_type=tx_type,
        data=tx_dict.get("data"),
        tx_id=tx_dict.get("tx_id", ""),
        timestamp=float(tx_dict.get("timestamp", time.time())),
        signature=signature,
        shard_id=int(tx_dict.get("shard_id", 0)),
        contract_code=tx_dict.get("contract_code"),
    )


def dict_to_block(block_dict: Dict[str, Any]) -> Block:
    """
    Convert a block dict into a Block instance.
    - Accepts either 'index' or 'height' for block index.
    - Converts nested transactions via dict_to_transaction.
    """
    # Support either 'index' or 'height'
    index = block_dict.get("index", block_dict.get("height", 0))

    transactions_data = block_dict.get("transactions", block_dict.get("txs", []))
    transactions: List[Transaction] = []
    for tx in transactions_data:
        # if tx already looks like Transaction instance, keep it
        if isinstance(tx, Transaction):
            transactions.append(tx)
        else:
            transactions.append(dict_to_transaction(tx))

    blk = Block(
        index=int(index),
        previous_hash=block_dict.get("previous_hash", block_dict.get("prev_hash", "")),
        transactions=transactions,
        timestamp=float(block_dict.get("timestamp", time.time())),
        nonce=int(block_dict.get("nonce", 0)),
        merkle_root=block_dict.get("merkle_root", ""),
        hash=block_dict.get("hash", ""),
        difficulty=int(block_dict.get("difficulty", getattr(block_dict, "difficulty", None) or 0)),
        gas_used=int(block_dict.get("gas_used", 0)),
        gas_limit=int(block_dict.get("gas_limit", getattr(block_dict, "gas_limit", 0))),
        proposer=block_dict.get("proposer", ""),
        shard_id=int(block_dict.get("shard_id", 0)),
    )

    # If merkle_root or hash missing but transactions available, optionally compute
    try:
        if not blk.merkle_root and blk.transactions:
            blk.calculate_merkle_root()
        if not blk.hash:
            blk.calculate_hash()
    except Exception:
        # be tolerant: don't fail conversion if calculate_* raises
        pass

    return blk


# -----------------------
# API handlers
# -----------------------

async def create_wallet(request: Request):
    wallet = wallet_manager.create_wallet()
    wallet_doc = {
        "wallet_id": wallet.wallet_id,
        "addresses": [wallet.get_address(i) for i in wallet.accounts.keys()]
    }
    # Insert to DB (matching your existing DB API)
    await wallets_collection.insertOne(wallet_doc)
    return JSONResponse(CreateWalletResponse(**wallet_doc).model_dump())


async def get_wallet(request: Request):
    try:
        wallet_id = request.path_params["wallet_id"]
        # Query - using your DB style
        res = await wallets_collection.find().where(wallet_id=wallet_id).select("wallet_id", "addresses").execute()

        if not getattr(res, "acknowledged", False):
            return JSONResponse({"error": "Wallet not found"}, status=HTTP_404_NOT_FOUND)

        raw = res.raw_result[0] if res.raw_result else None
        if not raw:
            return JSONResponse({"error": "Wallet not found"}, status=HTTP_404_NOT_FOUND)

        return JSONResponse(WalletResponse(**raw).model_dump())
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Internal server error"}, status=HTTP_400_BAD_REQUEST)


async def create_transaction(request: Request):
    try:
        data = await request.json()
        req = CreateTransactionRequest(**data)

        wallet_info = wallet_manager.get_wallet_for_address(req.sender)
        if not wallet_info:
            return JSONResponse({"error": "Sender wallet/address not found"}, status=HTTP_404_NOT_FOUND)
        wallet, account_index = wallet_info
        private_key = wallet.accounts[account_index]["private"]

        tx = Transaction(
            sender=req.sender,
            recipient=req.recipient,
            amount=req.amount,
            nonce=req.nonce,
            gas_price=req.gas_price,
            gas_limit=req.gas_limit,
            tx_type=TxType[req.tx_type]
        )
        tx.sign(private_key)

        tx_doc = serialize_transaction(tx)
        await tx_collection.insertOne(tx_doc)

        resp = CreateTransactionResponse(tx_id=tx.tx_id, signature=tx_doc["signature"])
        return JSONResponse(resp.model_dump())

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status=HTTP_400_BAD_REQUEST)


async def get_transaction(request: Request):
    try:
        tx_id = request.path_params["tx_id"]
        res = await tx_collection.find().where(tx_id=tx_id).execute()
        if not getattr(res, "acknowledged", False):
            return JSONResponse({"error": "Transaction not found"}, status=HTTP_404_NOT_FOUND)

        raw = res.raw_result[0] if res.raw_result else None
        if not raw:
            return JSONResponse({"error": "Transaction not found"}, status=HTTP_404_NOT_FOUND)

        return JSONResponse(TransactionResponse(**raw).model_dump())
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Internal server error"}, status=HTTP_400_BAD_REQUEST)


async def verify_transaction(request: Request):
    try:
        tx_id = request.path_params["tx_id"]
        res = await tx_collection.find().where(tx_id=tx_id).execute()
        if not getattr(res, "acknowledged", False):
            return JSONResponse({"error": "Transaction not found"}, status=HTTP_404_NOT_FOUND)

        raw = res.raw_result[0] if res.raw_result else None
        if not raw:
            return JSONResponse({"error": "Transaction not found"}, status=HTTP_404_NOT_FOUND)

        sender_addr = raw.get("sender")
        wallet_info = wallet_manager.get_wallet_for_address(sender_addr)
        if not wallet_info:
            return JSONResponse({"error": "Wallet not found for sender"}, status=HTTP_404_NOT_FOUND)
        wallet, account_index = wallet_info
        public_key = wallet.accounts[account_index]["public"]

        tx = Transaction(
            sender=raw["sender"],
            recipient=raw["recipient"],
            amount=raw["amount"],
            nonce=raw["nonce"],
            gas_price=raw["gas_price"],
            gas_limit=raw["gas_limit"],
            tx_type=TxType[raw["tx_type"]] if isinstance(raw["tx_type"], int) else TxType[raw["tx_type"]],
            data=raw.get("data"),
            tx_id=raw.get("tx_id", ""),
            timestamp=raw.get("timestamp", 0.0),
            signature=bytes.fromhex(raw["signature"]) if raw.get("signature") else b'',
            shard_id=raw.get("shard_id", 0),
            contract_code=raw.get("contract_code")
        )

        valid = tx.verify(public_key)
        resp = VerifyTransactionResponse(tx_id=tx.tx_id, valid=valid)
        return JSONResponse(resp.model_dump())
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Internal server error"}, status=HTTP_400_BAD_REQUEST)


async def execute_contract(request: Request):
    try:
        data = await request.json()
        req = ExecuteContractRequest(**data)

        state = VMState(gas_limit=req.gas_limit)
        new_state = smart_contract_vm.execute(req.contract_code, state, context={})
        resp = ExecuteContractResponse(stack=new_state.stack, gas_used=new_state.gas_used)
        return JSONResponse(resp.model_dump())
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status=HTTP_400_BAD_REQUEST)


async def switch_consensus(request: Request):
    try:
        data = await request.json()
        req = SwitchConsensusRequest(**data)
        consensus_manager.switch_consensus(req.consensus_type)
        resp = SwitchConsensusResponse(message=f"Switched consensus to {req.consensus_type}")
        return JSONResponse(resp.model_dump())
    except Exception:
        traceback.print_exc()
        return JSONResponse({"error": "Internal server error"}, status=HTTP_400_BAD_REQUEST)


async def validate_block(request: Request):
    try:
        data = await request.json()

        # expected payload: {'block': { ... }, 'chain_state': {...}}
        block_data = data.get("block")
        chain_state = data.get("chain_state", {})

        if block_data is None:
            return JSONResponse({"error": "Missing 'block' data"}, status=HTTP_400_BAD_REQUEST)

        # Convert dict -> Block instance (and nested transactions)
        block_obj = dict_to_block(block_data)

        # Use consensus manager to validate
        is_valid = await consensus_manager.validate_block(block_obj, chain_state)

        resp = ValidateBlockResponse(valid=is_valid)
        return JSONResponse(resp.model_dump())

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status=500)
