from gymnasium import spaces, utils

from miniworld.entity import COLOR_NAMES, Ball, Box, Key
from miniworld.miniworld import MiniWorldEnv
from copy import deepcopy


class PickupObjects(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Room with multiple objects. The agent collects +1 reward for picking up
    each object. Objects disappear when picked up.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move_back                   |
    | 4   | pickup                      |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agents see.

    ## Rewards

    +1 when agent picked up object

    ## Arguments

    * `size`: size of world
    * `num_objs`: number of objects

    ```python
    env = gymnasium.make("Miniworld-PickupObjects-v0", size=12, num_objs=5)
    ```
    """

    def __init__(self, size=12, num_objs=5, **kwargs):
        assert size >= 2
        self.size = size
        self.num_objs = num_objs

        MiniWorldEnv.__init__(self, max_episode_steps=400, **kwargs)
        utils.EzPickle.__init__(self, size, num_objs, **kwargs)

        # Reduce the action space
        self.action_space = spaces.Discrete(self.actions.pickup + 1)
        self.world_objects = []


    def _gen_world(self):
        self.add_rect_room(
            min_x=0,
            max_x=self.size,
            min_z=0,
            max_z=self.size,
            wall_tex="brick_wall",
            floor_tex="asphalt",
            no_ceiling=True,
        )

        if len(self.world_objects) == 0:
            obj_types = [Ball, Box, Key]
            colorlist = list(COLOR_NAMES)

            for obj in range(self.num_objs):
                obj_type = obj_types[self.np_random.choice(len(obj_types))]
                color = colorlist[self.np_random.choice(len(colorlist))]

                if obj_type == Box:
                    ent = self.place_entity(Box(color=color, size=0.9))
                if obj_type == Ball:
                    ent = self.place_entity(Ball(color=color, size=0.9))
                if obj_type == Key:
                    ent = self.place_entity(Key(color=color))

                self.world_objects.append((obj_type, color, ent.pos, ent.dir))

            ent = self.place_agent()
            self.world_objects.append(("agent", None, ent.pos, ent.dir))

        else:
            # set the same objects
            for obj in self.world_objects:
                obj_type, color, pos, dir = obj
                if obj_type == Box:
                    ent = self.place_entity(Box(color=color, size=0.9), pos=pos, dir=dir)
                if obj_type == Ball:
                    ent = self.place_entity(Ball(color=color, size=0.9), pos=pos, dir=dir)
                if obj_type == Key:
                    ent = self.place_entity(Key(color=color), pos=pos, dir=dir)
                if obj_type == "agent":
                    ent = self.place_agent(pos=pos, dir=dir)

        self.num_picked_up = 0

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        entity_name = None
        if self.agent.carrying:
            entity_name = self.agent.carrying.mesh_name
            print(f"{entity_name}")
            termination = True
            # self.entities.remove(self.agent.carrying)
            # self.agent.carrying = None
            # self.num_picked_up += 1
            # reward = 1

            # if self.num_picked_up == self.num_objs:
            #     termination = True
        
        if entity_name is not None:
            info["event"] = entity_name

        return obs, reward, termination, truncation, info
