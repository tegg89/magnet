# MAGNN: Deep learning Jeju camp 2018

* This project is currently under construction. 


## Authors

Tegg Taekyong Sung & Aleksandra Malysheva


## Objective

The goal of this project is controlling multi-agents using reinforcement learning and graph neural networks. Multi-agent scenarios are usually sparsely rewarded. Graph neural networks have an advantage that each node can be trained robustly. With this property, we hypothesized that each agent in an environment can be controlled individually. Since there have been many research papers related to graph neural networks, we would like to apply it to reinforcement learning.

For the experiment, we will use [Pommerman](https://www.pommerman.com) environment. This has relatively strict constraints on environment settings and simple to deploy algorithms.


## Proposed methods

* The proposed architectures is structured with two stages, graph construction and optimal action execution.
* Inspired by the [curiosity-driven paper](https://arxiv.org/abs/1705.05363), we use self-supervised prediction to infer environments, constructing graph matrix. Taking the concatenation of previous states and actions, the graph is constructed. This stage is solving regression problem in supervised learning.
* Afterward, the trained graph goes through [NerveNet](https://openreview.net/pdf?id=S1sqHMZCb) to perform action. Also the graph goes to MLP with concatenated state and action value to produce action value. Those two values are compared and trained using [DDPG](https://arxiv.org/abs/1509.02971) algorithm. 
* The design of network is shown below.
<!-- <p align="center">
  <img src="https://github.com/tegg89/DLCamp_Jeju2018/blob/master/asset/curr_network.jpg?raw=true" width=70% title="network">
</p> -->


## Dependencies
The script has been tested running under Python 3.6.6, with the following packages installed (along with their dependencies):
* `numpy==1.14.5`
* `tensorflow==1.8.0`


## Experiments

#### 1. Environements
* [Pommerman](https://github.com/MultiAgentLearning/playground) is sponsored by NVIDIA, FAIR, and Google AI.
* For each agent: 372 observation spaces (board, bomb_blast strength, bomb_life, position, blast strength, can kick, teammate, ammo, enemies) & 6 action spaces (up, down, right, left, bomb, stop)
* Free for all & Team match modes are available.

<p align="center">
  <img src="https://github.com/tegg89/DLCamp_Jeju2018/blob/master/asset/pommerman.png?raw=true" width=60% title="pommerman">
</p>

#### 2. Results
<!-- <p align="center">
  <img src="https://github.com/tegg89/DLCamp_Jeju2018/blob/master/asset/prev_result.png?raw=true" width=90% title="results">
</p> -->


## TODO

- [x] Attach self-attention module at the graph generation
- [x] Substitute execution stage to NerveNet
- [x] Redraw network structure
- [x] Experimental comparison
- [x] Ablation study
- [ ] Prepare arXiv paper


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
