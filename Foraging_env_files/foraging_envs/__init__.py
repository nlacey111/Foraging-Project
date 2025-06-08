from gymnasium.envs.registration import register

register(
    id="foraging_envs/foraging-one",
    entry_point="foraging_envs.envs:ForagingClass",
)

register(
    id="foraging_envs/foraging-two",
    entry_point="foraging_envs.envs:ForagingClass2",
)

register(
    id="foraging_envs/foraging-three",
    entry_point="foraging_envs.envs:ForagingClass_tube",
)