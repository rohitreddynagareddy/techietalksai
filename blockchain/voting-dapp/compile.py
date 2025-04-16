import json
from solcx import compile_standard, set_solc_version

# Use the compiled-from-source solc
set_solc_version('0.8.9')
# Read Solidity source code
with open("contracts/Voting.sol", "r") as file:
    source_code = file.read()

# Compile Solidity source code
compiled_sol = compile_standard(
    {
        "language": "Solidity",
        "sources": {"Voting.sol": {"content": source_code}},
        "settings": {
            "outputSelection": {
                "*": {
                    "*": ["abi", "evm.bytecode"]
                }
            }
        },
    },
    solc_version="0.8.9",
)

# Save ABI and bytecode (existing code remains the same)
# ... (rest of your existing file content)

# Extract ABI and bytecode
abi = compiled_sol["contracts"]["Voting.sol"]["Voting"]["abi"]
bytecode = compiled_sol["contracts"]["Voting.sol"]["Voting"]["evm"]["bytecode"]["object"]

# Save ABI and bytecode to files
with open("Voting_abi.json", "w") as abi_file:
    json.dump(abi, abi_file, indent=2)

with open("Voting_bytecode.txt", "w") as bytecode_file:
    bytecode_file.write(bytecode)

print("Compilation successful. ABI and bytecode have been saved.")
