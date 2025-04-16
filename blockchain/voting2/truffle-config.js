module.exports = {
  networks: {
    ganache: {
      host: "ganache",
      port: 8545,
      network_id: "*" // Match any network ID
    }
  },
  compilers: {
    solc: {
      version: "0.8.0" // Match the Solidity version in Voting.sol
    }
  },
  contracts_directory: "./contracts",
  contracts_build_directory: "./build/contracts"
};
