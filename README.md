## Reach The Ball

In this project Reinforcement Learning agent will Continously Reach the ball . Below is the snapshot of the game:

![Reach The Ball](images/reacher.gif)


### Description
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

This problem statement can be solved using single agent or multiple agents:

1. The task is episodic, and in order to solve the environment, agent must get an average score of +30 over 100 consecutive episodes.

2. The second version of the environment is slightly different, to take into account the presence of many agents. In particular, agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,After each episode, we add up the rewards that each agent received , to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores. This yields an average score for each episode (where the average is over all 20 agents).

The code implemented in this project is via multiple agents. If you want to run on single agent, do as steps given below:

```#env = env_option("multiple_agent")
   env = env_option("single_agent")
```


### Dependencies
1. Python 3
2. Pytorch
3. Jupyter-notebook
4. Untiy Environment

### Getting Started
1. Clone this repository on your local machine
2. Open the project using jupyter-notebook
3. [Shift + Ent] to execute every cell. This particular code will start learning neural network and will run one episode in backgorund
4. You can also load the weights provided in this repositroy using code below:
   ```device = torch.device('cpu')
      model = TheModelClass(*args, **kwargs)
      model.load_state_dict(torch.load(PATH, map_location=device))

    ```
    [Using pytorch to load weights in your algorithm](https://pytorch.org/tutorials/beginner/saving_loading_models.html)
    

