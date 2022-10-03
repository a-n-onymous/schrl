# Constrained Hierarchical Deep Reinforcement Learning with Differentiable Formal Specifications

## Demos

4 robots perform different tasks below. [Full demos](https://sites.google.com/view/schrl) are collected in a website.

<video src='https://user-images.githubusercontent.com/113462441/193688270-89516a17-7b6e-4416-94ee-aab547d01daa.mp4' width=180/> | <video src='https://user-images.githubusercontent.com/113462441/193688401-b724199b-f26b-41ce-8fe4-5fe33ebea0b6.mp4' width=180/>
:-: | :-:
<video src='https://user-images.githubusercontent.com/113462441/193688480-cd06b588-fc73-4c65-b880-baf32cbbf737.mp4' width=180/> | <video src='https://user-images.githubusercontent.com/113462441/193688428-47d389c7-9c84-4628-9d4c-626c2fdb6359.mp4' width=180/>

## Install

1. Install this package and dependencies (e.g., pybullet, pytorch, etc.)

```sh
pip instal -e .
```

2. Follow this [document](https://github.com/openai/mujoco-py) to configure and troubleshoot MuJoCo(-py).

## Examples

### Differentiable Specifications

We implemented a [simple LL parser](src/schrl/tltl/spec.py) for our differentiable TLTL specifications.

1. [Syntax Examples](examples/tltl/1_syntax.py): how to write TLTL specifications with pure python built-in operators.
2. [Quantitative Semantics Examples](examples/tltl/2_quantitative_semantics.py): forward and backward through TLTL
   specifications.
3. [Specifications](examples/tltl/3_specs.ipynb): 5 types of specifications and solving them with gradient.

### Goal-Conditioned Environments and Policies

[Reach Random Goals](examples/envs/reach_random_goals.py): An example shows how the goal-conditioned environments
and policies work.

### Pretrained Policies

[Run Pretrained Policies](examples/hrl/pretrained.py): Run our pretrained policies with

```shell
python examples/hrl/pretrained.py \
  --robot_name [point, car, doggo, drone] \
  --task_name [seq, cover, branch, loop, signal] \
  --gui \
  --n_eps [number_of_epochs]
```

for example, below command runs pretrained policy for `point` robot's `seq` task `5` times with gui

```shell
python examples/hrl/pretrained.py \
  --robot_name point \
  --task_name seq \
  --gui \
  --n_eps 5
```

## Reuse and Citation

This code repository is packed as a standard python package. Install it with `pip install -e .`, and one will be able to
revoke all the modules. Detailed examples for these modules are provided in `examples/`.

Tweaking configuration [here](src/schrl/config.py) can change the log level, PyTorch device, etc.

One related paper to this repository is under a double-blind review process. The BibTeX will be available after we can
make it public. 
