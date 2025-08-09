#!/bin/bash
# Simple run script to start your Aquilify server

export P2P_PORT=9000
export RPC_PORT=8000
export WS_PORT=8001
export DHT_PORT=9001
export CONSENSUS=PoW

uvicorn your_package.main:app --host 0.0.0.0 --port 8000 --reload
