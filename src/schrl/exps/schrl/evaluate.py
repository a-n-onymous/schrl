from schrl.hrl.hrl import HRL


def evaluate_hrl_policy(hrl: HRL, n_epochs: int = 10):
    task = hrl.plan_layer.task
    env = task.env
    ctrl_rewards = []

    for _ in range(n_epochs):
        env.reset()
        ctrl_reward = 0

        path = hrl.plan_layer.predict(env.get_state())
        reached_subgoal = True
        goal_indx = 0

        for _ in range(task.time_limit):
            if reached_subgoal:
                goal_indx += 1
                if goal_indx == len(path):
                    break
                env.set_goal(path[goal_indx])

            action, _ = hrl.control_layer.algo.predict(env.get_obs())
            _, r, _, info = env.step(action)
            reached_subgoal = info["reach"]
            ctrl_reward += r

            if task.enable_gui:
                env.render()

        ctrl_rewards.append(ctrl_reward)

    env.close()
    
    return {
        "ctrl_rewards": ctrl_rewards
    }
