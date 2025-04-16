// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Voting {
    mapping(address => bool) public voters;
    mapping(string => uint256) public votes;
    string[] public candidates;

    constructor(string[] memory candidateNames) {
        for (uint i = 0; i < candidateNames.length; i++) {
            candidates.push(candidateNames[i]);
            votes[candidateNames[i]] = 0;
        }
    }

    function vote(string memory candidate) public {
        require(!voters[msg.sender], "Already voted");
        require(validCandidate(candidate), "Invalid candidate");
        voters[msg.sender] = true;
        votes[candidate] += 1;
    }

    function getVotes(string memory candidate) public view returns (uint256) {
        require(validCandidate(candidate), "Invalid candidate");
        return votes[candidate];
    }

    function validCandidate(string memory candidate) private view returns (bool) {
        for (uint i = 0; i < candidates.length; i++) {
            if (keccak256(abi.encodePacked(candidates[i])) == keccak256(abi.encodePacked(candidate))) {
                return true;
            }
        }
        return false;
    }

    function getCandidates() public view returns (string[] memory) {
        return candidates;
    }
}
