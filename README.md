# MAGNet: Deep learning Jeju camp 2018

* This project is currently under construction. 


## Authors

Tegg Taekyong Sung & Aleksandra Malysheva


## Objective

The goal of this project is controlling multi-agents using reinforcement learning and graph neural networks. Multi-agent scenarios are usually sparsely rewarded. Graph neural networks have an advantage that each node can be trained robustly. With this property, we hypothesized that each agent in an environment can be controlled individually. Since there have been many research papers related to graph neural networks, we would like to apply it to reinforcement learning.

For the experiment, we will use [Pommerman](https://www.pommerman.com) environment. This has relatively strict constraints on environment settings and simple to deploy algorithms.


## Proposed methods

* The proposed architectures is structured with two stages, graph construction and optimal action execution.
* Inspired by the [curiosity-driven paper](https://arxiv.org/abs/1705.05363), we use self-supervised prediction to infer environments, constructing graph matrix. Taking the concatenation of previous states and actions, the graph is constructed. This stage is solving a regression problem in supervised learning.
* Afterward, the trained graph goes through [NerveNet](https://openreview.net/pdf?id=S1sqHMZCb) to perform an action. Also, the graph goes to MLP with concatenated state and action value to produce action value. Those two values are compared and trained using [DDPG](https://arxiv.org/abs/1509.02971) algorithm. 
* The design of the network is shown below. 

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/curr-network.png?raw=true" width=70% title="network">
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

#### 2-1. Overall algorithm comparison

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/gs-nn.png?raw=true" width=70% title="results-overall">
</p>

* Top graph shows the performance of theproposed model and other RL algorithms. 
* Bottom graph shows the effectiveness of graph construction. 
* Graph sharing within the team and individual graphs per agents are tested, and with the shared graph construction model gains the better performance.

#### 2-2. Ablation studies

(The detail parameters of algorithms (A1~A12) shows in the paper.)


**Self-attention experiments**

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/nn1-team-selfattn-n-teamagent-s-a.png?raw=true" width=80% title="results-sa">
</p>

* Left graph shows the effectiveness of self-attention module based on the shared graph. A1~A3 are vanilla networks without self-attention module, and A4 & A5 are models with the self-attention module. 
* Right graph shows the self-attention tests based on individual graphs per agents. A10~A12 show the models with the self-attention module, and A13 shows the model without the module.
* Also, the model with self-attention module has better performance on graph regression problem.


**Hyperparameter experiments**

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/nn1-team-n-agent.png?raw=true" width=80% title="results-param">
</p>

* Left graph shows the experiment of self-attention in graph construction stage.
* Right graph shows the experiment of the way of constructing graph. A1~A3 refer to shared graph, and A10~A12 refer to individual graphs per agents.


**Graph evaluations**

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/graph-evaluation.png?raw=true" width=80% title="results-param">
</p>

* We experimented with an effectiveness of shared and individual graphs. We set individual graphs per agents to the opposite side and take the shared graph to the allied side. As we training the model, the shared graph has better performance over separated ones.

<p align="center">
  <img src="https://github.com/tegg89/magnet/blob/master/asset/processing_results/paper/graph-vis.png?raw=true" width=80% title="results-param">
</p>

* The visualizations of constructed graphs are shown in this result. The left graph shows the shared graph, whereas the right graph shows the separated graphs. 
* At the beginning of the game (top), the graph is ordered with same structures with equally distributed edges.
* In the middle of the game (bottom), shared graph one shows the same team chases one of the opponent agents. In the separated graph, all agents are evenly chasing to each other.


## TODO

- [x] Attach self-attention module at the graph generation
- [x] Substitute execution stage to NerveNet
- [x] Redraw network structure
- [x] Experimental comparison
- [x] Ablation study
- [x] Prepare arXiv paper


## References

* Graph Attention Networks [paper](https://arxiv.org/abs/1710.10903)
* Relational Deep Reinforcement Learning [paper](https://arxiv.org/abs/1806.01830)
* Nervenet: Learning structured policy with graph neural networks [paper](https://openreview.net/pdf?id=S1sqHMZCb)
* Curiosity-driven exploration by self-supervised prediction [paper](https://arxiv.org/abs/1705.05363)
* PlayGround: AI research into multi-agent learning [paper](https://github.com/MultiAgentLearning/playground)
* Zero-shot task generalization with multi-task deep reinforcement learning [paper](https://arxiv.org/abs/1706.05064)
* Gated graph sequence neural networks [paper](https://arxiv.org/abs/1511.05493)
* Few-shot learning with graph neural networks [paper](https://arxiv.org/abs/1711.04043)
* Backplay: ‘Man muss immer umkehren’ [paper](https://arxiv.org/abs/1807.06919)
* Continuous control with deep reinforcement learning [paper](https://arxiv.org/abs/1509.02971)


## Acknowledgement
This was supported by [Deep Learning Camp Jeju 2018](http://jeju.dlcamp.org/2018/) which was organized by [TensorFlow Korea User Group](https://facebook.com/groups/TensorFlowKR/).


## License

Apache
