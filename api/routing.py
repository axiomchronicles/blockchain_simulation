from aquilify.core.schematic.routing import rule

from . import core

# ROUTER configuration.

# The `ROUTER` list routes URLs to views.
# Examples:
# Function views
#     1. Add an import:  from my_app import views
#     2. Add a URL to ROUTER:  rule('/', views.home, name='home')
# Including another ROUTING
#     1. Import the include() function: from aquilify.core.routing import include, rule
#     2. Add a URL to ROUTER:  rule('/blog', include = include('blog.routing'))

ROUTER = [
    rule("/wallets", core.create_wallet, methods=["POST"]),
    rule("/wallets/{wallet_id}", core.get_wallet, methods=["GET"]),

    rule("/transactions", core.create_transaction, methods=["POST"]),
    rule("/transactions/{tx_id}", core.get_transaction, methods=["GET"]),
    rule("/transactions/{tx_id}/verify", core.verify_transaction, methods=["GET"]),

    rule("/contracts/execute", core.execute_contract, methods=["POST"]),

    rule("/consensus/switch", core.switch_consensus, methods=["POST"]),
    rule("/consensus/validate_block", core.validate_block, methods=["POST"]),
]
