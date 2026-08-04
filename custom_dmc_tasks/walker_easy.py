from custom_dmc_tasks import walker

class EasyInitPlanarWalker(walker.PlanarWalker):
    def __init__(self, noise_std=0.05, **kwargs):
        self._noise_std = noise_std
        super().__init__(**kwargs)

    def initialize_episode(self, physics):
        # Reset to default QPOS (upright stance)
        with physics.reset_context():
            # Add small noise to default joint angles instead of full randomization
            noise = self.random.normal(scale=self._noise_std, size=physics.data.qpos.shape)
            physics.data.qpos[:] += noise
            physics.data.qvel[:] = 0.0  # start from rest
        
        # Skip the default randomize_limited_and_rotational_joints call
        walker.base.Task.initialize_episode(self, physics)