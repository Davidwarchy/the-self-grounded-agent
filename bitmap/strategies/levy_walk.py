import random
import asyncio
import numpy as np
from .base_strategy import BaseStrategy

class LevyWalkStrategy(BaseStrategy):
    def __init__(self, alpha=1.6, min_step=1.0, max_step=200.0):
        parameters = {
            "alpha": alpha,
            "min_step": min_step,
            "max_step": max_step
        }
        super().__init__("levy_walk", parameters)
        
        self.alpha = alpha
        self.min_step = min_step
        self.max_step = max_step

    def _sample_truncated_pareto(self):
        """Sample from a truncated Pareto distribution."""
        u = np.random.uniform()
        x = self.min_step * (1 - u)**(-1.0 / self.alpha)
        while x > self.max_step:
            u = np.random.uniform()
            x = self.min_step * (1 - u)**(-1.0 / self.alpha)
        return int(max(1, round(x)))
    
    async def run(self, env):
        obs = env.reset()
        done = False

        current_run_length = 0
        current_direction = random.randint(0, 3)

        while not done:
            # Start a new run if finished the last one
            if current_run_length <= 0:
                current_direction = random.randint(0, 3)
                current_run_length = self._sample_truncated_pareto()

            # Step environment
            obs, reward, done, info = env.step(current_direction)

            # Render environment if needed
            if env.render_flag:
                env.render()

            # Update run counter
            current_run_length -= 1

            # Print step info occasionally
            if env.current_step % 100 == 0:
                print(
                    f"Step {env.current_step}, "
                    f"Direction: {info['action']}, "
                    f"Run steps left: {current_run_length}, "
                    f"Coverage: {info['coverage']:.2f}%"
                )

            # Uncomment to slow down
            # await asyncio.sleep(1.0 / env.fps)

        return env.current_step, env._get_coverage()