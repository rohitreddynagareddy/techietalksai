// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Voting {
    struct Candidate {
        string name;
        uint256 voteCount;
    }

    address public owner;
    bool public votingOpen = true;

    mapping(address => bool) public hasVoted;
    Candidate[] public candidates;

    constructor(string[] memory _candidateNames) {
        owner = msg.sender;
        for (uint i = 0; i < _candidateNames.length; i++) {
            candidates.push(Candidate({name: _candidateNames[i], voteCount: 0}));
        }
    }

    function vote(uint candidateIndex) public {
        require(votingOpen, "Voting has ended");
        require(!hasVoted[msg.sender], "You have already voted");
        require(candidateIndex < candidates.length, "Invalid candidate");

        candidates[candidateIndex].voteCount += 1;
        hasVoted[msg.sender] = true;
    }

    function endVoting() public {
        require(msg.sender == owner, "Only owner can end voting");
        votingOpen = false;
    }

    function getCandidates() public view returns (Candidate[] memory) {
        return candidates;
    }

    function getWinner() public view returns (string memory winnerName, uint256 winnerVotes) {
        require(!votingOpen, "Voting still ongoing");

        uint maxVotes = 0;
        uint winnerIndex = 0;

        for (uint i = 0; i < candidates.length; i++) {
            if (candidates[i].voteCount > maxVotes) {
                maxVotes = candidates[i].voteCount;
                winnerIndex = i;
            }
        }

        return (candidates[winnerIndex].name, candidates[winnerIndex].voteCount);
    }
}
