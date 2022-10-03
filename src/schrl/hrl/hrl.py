from schrl.config import DEFAULT_DEVICE
from schrl.envs import GoalEnv
from schrl.hrl.control_layer import ControlLayerBase, PPOControlLayer
from schrl.hrl.plan_layer import PlanLayerBase, LearnablePlanLayer
from schrl.tasks import TaskBase


class HRL:
    def __init__(self,
                 plan_layer: PlanLayerBase,
                 control_layer: ControlLayerBase):
        self.plan_layer = plan_layer
        self.control_layer = control_layer
        self.control_layer.register_callbacks(self.plan_layer.callback_list)
        self.env = plan_layer.env  # type: GoalEnv

    def learn(self,
              total_timesteps: int,
              log_interval: int = 4,
              *args,
              **kwargs):
        self.control_layer.learn(total_timesteps, log_interval, *args, **kwargs)

    def disable_learning_control(self):
        self.control_layer.disable_learning()

    def enable_learning_control(self):
        self.control_layer.enable_learning()

    def disable_learning_plan(self):
        self.plan_layer.disable_learning()

    def enable_learning_plan(self):
        self.plan_layer.enable_learning()

    def save(self, path: str):
        self.control_layer.save(path)
        self.plan_layer.save(path)

    @classmethod
    def load(cls, path: str, task: TaskBase, device=DEFAULT_DEVICE):
        plan_layer = LearnablePlanLayer.load(path, task, device=device)
        ctrl_layer = PPOControlLayer.load(path, task.env, device=device)
        return cls(plan_layer, ctrl_layer)
