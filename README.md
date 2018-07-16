# Deep learning Jeju camp 2018

* This project is built during '[Deep Learning Jeju Camp](http://jeju.dlcamp.org)' in 2018.
* This project is currently under construction. 

## Objective

The goal of this project is controlling multi-agents using graph neural networks. Multi-agent scenarios are usually sparsely rewarded. Graph neural networks have an advantage that each node can be trained robustly. With this property, we hypothesized that each agent in an environment can be controlled individually. Since there have been many research papers related to graph neural networks, we would like to apply it to reinforcement learning.

For the experiment, we will use [Pommerman](https://www.pommerman.com) environment. This has relatively strict constraints on environment settings and simple to deploy algorithms.


## Proposed methods

* The proposed architectures is structured with two stages, generating graph and executing optimal actions.
* Inspired by the [curiosity-driven paper](https://arxiv.org/abs/1705.05363), we use self-supervised prediction to infer environments. Taking previous states and actions, the first network is inferring the environment which can be generated to graph. 
* Afterward, each agents execute the optimal actions based on the trained graph.
* The network design of prototype is shown below.
* ![](https://github.com/tegg89/DLCamp_Jeju2018/blob/master/asset/prev_network.jpg?raw=true) <!-- .element height="50%" width="50%" -->


## Experiments

#### 1. Environements
* [Pommerman](https://github.com/MultiAgentLearning/playground) is sponsored by NVIDIA, FAIR, and Google AI.
* For each agent: 372 observation spaces (board, bomb_blast strength, bomb_life, position, blast strength, can kick, teammate, ammo, enemies) & 6 action spaces (stop, up, down, right, left, bomb)
* Free for all & Team match modes are available.
* ![](https://github.com/tegg89/DLCamp_Jeju2018/blob/master/asset/pommerman.png?raw=true) <!-- .element height="50%" width="50%" -->

#### 2. Results
* ![](https://github.com/tegg89/DLCamp_Jeju2018/blob/master/asset/prev_result.png?raw=true) <!-- .element height="50%" width="50%" -->


## TODO

- [ ] Attach self-attention module at the graph generation
- [ ] Substitute execution stage to Nervenet
- [ ] Compare with random and heuristic agents
- [ ] Prepare arXiv paper


## References

* [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
* [Relational Deep Reinforcement Learning](https://arxiv.org/abs/1806.01830)
* [Nervenet: Learning structured policy with graph neural networks](https://openreview.net/pdf?id=S1sqHMZCb)
* [Curiosity-driven exploration by self-supervised prediction](https://arxiv.org/abs/1705.05363)
* [PlayGround: AI research into multi-agent learning](https://github.com/MultiAgentLearning/playground)
* [Zero-shot task generalization with multi-task deep reinforcement learning](https://arxiv.org/abs/1706.05064)

## License

Apache