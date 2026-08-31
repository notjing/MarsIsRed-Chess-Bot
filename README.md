# MarsIsRed - Chess Engine

MarsIsRed is an AlphaZero style chess engine, it utilises a dual-policy neural network and a Monte Carlo Tree Search (MCTS) to find the best move instead of relying on opening books / tablebase and human made evaluation functions. It is currently at a provisional 1800 ELO on Lichess blitz.

# Overview

To be clear, MarsIsRed is not a wrapper around an existing engine nor a pure AlphaZero clone.  

Here is the creation process at a high level:  

**Creating the Base** - The first version (V0) was created through supervised training on 10m positions from the [Lichess Elite Database](https://database.nikonoel.fr/)  
**Self-Play** - The bot then plays against itself, creating 800k positions of data per iteration with some variance injected  
**Training** - Using this data, it then trains the existing champion to create a new iteration  
**Arena** - This new iteration then plays against the existing champion, if it performs better it replaces the existing champion  

# Model Information
## Inputs:  
Board Inputs: 
- Position of each piece (12 boards)
- Which squares are each type of piece attacking (12 boards)
- Enpassant (1 board)

Numerical Inputs:
- Castling Rights (4)
- Number of each piece (12)
- Material Delta (1)
- IsCheck (1)
- IsCheckmate (1)

## Outputs:
Value Head: A single number from [-1,1] indicating how likely the side to move is to win from this position  
- Loss: Mean Squared Error  

Policy Head: A probability distribution across 4672 indicies of how likely each move is to be played in this position  
- Loss: Categorial Cross Entropy

Note: The move indexing is the AlphaZero move encoding, click [here](https://ai.stackexchange.com/questions/27336/how-does-the-alpha-zeros-move-encoding-work) for more information.


# Data
The base model (V0) is trained off of 10m positions from the [Lichess Elite Database](https://database.nikonoel.fr/).  
In the RL part, each iteration creates 800,000 positions and a sliding window of 2.4m positions is kept to train the next iteration.  
Initially the supervised to self-generated data split is 60-40, but the supervised portion continues to drop until it hits 10-90

# How to Run
TBD


