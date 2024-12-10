# Deep Q Learning Agent with Replay Buffer

## Decaying Epsilon Greedy
Starting with high explore probabilities, gradually go low until it hits the lower bound. Reset when the game is stopped
## Replay Buffer & Target Net
The model maintains a replay buffer and samples play history to do the batch update. The agent keeps both the target net and policy net, only periodically updating the policy net, providing stability
## Broads Reconstruction
Broad info made of 1 and 2 are taken apart and reconstructed into two separated to avoid confusing the NN. Resulting input size is 2 * broad_size ^ 2 + 1(direction) + 1(pi)
## Training
Buffer replay mechanism and target net give the agent a certain amount of stability. However, sometimes the agents still go in a straight line. The solution is to adjust parameters to encourage more exploration, increase the buffer size for stability, and keep the batch size small to allow more frequent update
A diverse environment is kept in mind during training. Red/blue, dense/sparse is randomly chosen at the beginning of a new game. This way, the agent will not fixate on a single direction or playing mode.



## Customized Petting Zoo Hex Env
Got a nice visualization of the broad using pygame
![render](https://github.com/s87217647/Reinforcement_L/blob/main/hex/pygame%20render.png)
A more detailed description [here](https://github.com/sjsu-interconnect/ourhexgame)
