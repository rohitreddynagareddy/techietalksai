from web3 import Web3
import json

w3 = Web3(Web3.HTTPProvider('http://localhost:8545'))
with open('build/contracts/Voting.json') as f:
    abi = json.load(f)['abi']
contract = w3.eth.contract(address='0x345cA3e014Aaf5dcA488057592ee47305D9B3e10', abi=abi)
account = "0x627306090abaB3A6e1400e9345bC60c78a8BEf57"
private_key = "0xc87509a1c067bbde78beb793e6fa76530b6382a4c0241e5e4a9ec0a0f44dc0d3"  # Replace with Ganache private key
tx = contract.functions.vote("Alice").build_transaction({
    "from": account,
    "nonce": w3.eth.get_transaction_count(account),
    "gas": 1000000,
    "gasPrice": w3.to_wei("20", "gwei")
})
signed_tx = w3.eth.account.sign_transaction(tx, private_key)
tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)  # Corrected line
w3.eth.wait_for_transaction_receipt(tx_hash)
print("Voted for Alice")
# Verify the vote
votes = contract.functions.getVotes("Alice").call()
print(f"Votes for Alice: {votes}")