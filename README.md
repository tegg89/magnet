# Deep Multi-Agent Reinforcement Learning with Relevance Graphs

Code for the [Deep Multi-Agent Reinforcement Learning with Relevance Graphs](https://arxiv.org/abs/1811.12557) accepted from NeurIPS Deep RL workshop. 


## Objective

The goal of this project is controlling multi-agents using reinforcement learning and graph neural networks. Multi-agent scenarios are usually sparsely rewarded. Graph neural networks have an advantage that each node can be trained robustly. With this property, we hypothesized that each agent in an environment can be controlled individually. Since there have been many research papers related to graph neural networks, we would like to apply it to reinforcement learning.

For the experiment, we will use [Pommerman](https://www.pommerman.com) environment. This has relatively strict constraints on environment settings and simple to deploy algorithms.


## Proposed methods

* The proposed architectures is structured with two stages, graph construction and optimal action execution.
* Inspired by the [curiosity-driven paper](https://arxiv.org/abs/1705.05363), we use self-supervised prediction to infer environments, constructing graph matrix. Taking the concatenation of previous states and actions, the graph is constructed. This stage is solving a regression problem in supervised learning.
* Afterward, the trained graph goes through [NerveNet](https://openreview.net/pdf?id=S1sqHMZCb) to perform an action. Also, the graph goes to MLP with concatenated state and action value to produce action value. Those two values are compared and trained using [DDPG](https://arxiv.org/abs/1509.02971) algorithm. 
* The design of the network is shown below. 

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/MAGNet.png?raw=true" width=50% title="network">
</p>


## Dependencies

The script has been tested running under Python 3.6.6, with the following packages installed (along with their dependencies):

* `numpy==1.14.5`
* `tensorflow==1.8.0`


## Experiments

### 1. Environements

* [Pommerman](https://github.com/MultiAgentLearning/playground) is sponsored by NVIDIA, FAIR, and Google AI.
* For each agent: 372 observation spaces (board, bomb_blast strength, bomb_life, position, blast strength, can kick, teammate, ammo, enemies) & 6 action spaces (up, down, right, left, bomb, stop)
* Free for all & Team match modes are available.

<p align="center">
  <img src="https://github.com/tegg89/DLCamp_Jeju2018/blob/master/asset/pommerman.png?raw=true" width=40% title="pommerman">
</p>

### 2. Results

#### 2-1. Algorithm comparisons

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/gs-nn.png?raw=true" width=50% title="results-overall">
</p>

* Top graph shows the performance of the proposed model and other RL algorithms. 
* Bottom graph shows the effectiveness of graph construction. 
* Graph sharing within the team and individual graphs per agents are tested, and with the shared graph construction model gains the better performance.

#### 2-2. Graph evaluations

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/graph-evaluation.png?raw=true" width=50% title="results-param">
</p>

* We experimented with an effectiveness of shared and individual graphs. We set individual graphs per agents to the opposite side and take the shared graph to the allied side. As we training the model, the shared graph has better performance over separated ones.

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/visualgraph.png?raw=true" width=50% title="results-param">
</p>

* The visualizations of constructed graphs are shown in this result. The left graph shows the shared graph, whereas the right graph shows the separated graphs. 
* At the beginning of the game (top), the graph is ordered with same structures with equally distributed edges.
* In the middle of the game (bottom), shared graph one shows the same team chases one of the opponent agents. In the separated graph, all agents are evenly chasing to each other.


## Authors

Tegg Taekyong Sung & Aleksandra Malysheva


## Acknowledgement
This was supported by [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/) which was organized by [TensorFlow Korea User Group](https://facebook.com/groups/TensorFlowKR/).


## License

Apache
