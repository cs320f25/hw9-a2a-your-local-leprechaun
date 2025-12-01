# AI Engineering Final Project
For this final project, I am creating an agent to play the beautiful game of Tak from the series Kingkiller Chronicles. 

This project involves first building out the game in Python with the correct rules and gameplay, then porting it to AlphaZero language so the agent can be trained on it.

## Progress
After finishing building the game in a seperate repo that works, tak-game-python, I tried to have Claude port it over to AlphaZero. It seemed to work, as it's hard to test a game on alphazero in my experience, so I started training. 

Claude also put together a play_interactive.py script that would allow me to play the best trained agent, and immediately there were problems. The number of pieces didn't check out, as the agent was playing multiple of a piece that should only be allowed to play one. So I'm not reworking the system in AlphaZero and trying to get a better understanding of how it works so i can properly code it in, as Claude seems to be doing a mediocer job at it. 

I currently haven't added my Github Agent, nor by ADK agent, but my ADK agent should be able to play a game when given a board state and return with the next move by the neural network.