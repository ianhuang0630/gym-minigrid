#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gym_minigrid.minigrid import *
from gym_minigrid.register import register


class FourRoomsEnv(MiniGridEnv):
    """
    Classic 4 rooms gridworld environment.
    Can specify agent and goal position, if not it set at random.
    """

    def __init__(self, agent_pos=None, goal_pos=None):
        self._agent_default_pos = agent_pos
        self._goal_default_pos = goal_pos
        super().__init__(grid_size=19, max_steps=100)

    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height - 1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        self.room_w = width // 2
        self.room_h = height // 2

        # For each row of rooms
        for j in range(0, 2):

            # For each column
            for i in range(0, 2):
                xL = i * self.room_w
                yT = j * self.room_h
                xR = xL + self.room_w
                yB = yT + self.room_h

                # Bottom wall and door
                if i + 1 < 2:
                    self.grid.vert_wall(xR, yT, self.room_h)
                    pos = (xR, self._rand_int(yT + 1, yB))
                    self.grid.set(*pos, None)

                # Bottom wall and door
                if j + 1 < 2:
                    self.grid.horz_wall(xL, yB, self.room_w)
                    pos = (self._rand_int(xL + 1, xR), yB)
                    self.grid.set(*pos, None)
        
        self.agent_placement()
        self.goal_placement()

    def agent_placement(self):
        # Randomize the player start position and orientation
        if self._agent_default_pos is not None:
            self.agent_pos = self._agent_default_pos
            self.grid.set(*self._agent_default_pos, None)
            self.agent_dir = self._rand_int(0, 4)  # assuming random start direction
        else:
            self.place_agent()
    
    def goal_placement(self): 
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos)
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal())

        self.mission = 'Reach the goal'

    def step(self, action):
        obs, reward, done, info = MiniGridEnv.step(self, action)
        return obs, reward, done, info


"""
Below are implementations of the four-room environment with different sub-goals
"""
class FourRoomsEnv_ToRoom(FourRoomsEnv):
    """
    An environment where you get a reward for navigating to a specific room
    """
    def __init__(self, target_room_quad_x, target_room_quad_y, 
                agent_pos=None, goal_pos=None):
        super().__init__(agent_pos, goal_pos)
        
        self.quadrant_x = target_room_quad_x 
        self.quadrant_y = target_room_quad_y
        
    # overwriting 
    def reward_condition(self, fwd_cell, fwd_pos):
        """
        This checks to see if the agent is within the correct room.
        compares the fwd_cell to the raw coordinates 
        """

        # check if the fwd_cell is in the right position
        done = False 
        if fwd_pos[0] > self.quadrant_x * self.room_w and \
            fwd_pos[0] < (self.quadrant_x + 1)*self.room_w and \
            fwd_pos[1] > self.quadrant_y * self.room_h and \
            fwd_pos[1] < (self.quadrant_y+1)*self.room_h: 
            done = True
            return done, self._reward()
        return done, 0 
        
    # overwriting 
    def reset_reward_condition(self): 
        pass

# same environment, different target rooms.
class FourRoomsEnv_ToRoom1(FourRoomsEnv_ToRoom): 
    def __init__(self, agent_pos=None, goal_pos=None):
        super().__init__(0, 0, agent_pos, goal_pos)

class FourRoomsEnv_ToRoom2(FourRoomsEnv_ToRoom): 
    def __init__(self, agent_pos=None, goal_pos=None):
        super().__init__(1, 0, agent_pos, goal_pos)

class FourRoomsEnv_ToRoom3(FourRoomsEnv_ToRoom): 
    def __init__(self, agent_pos=None, goal_pos=None):
        super().__init__(0, 1, agent_pos, goal_pos)

class FourRoomsEnv_ToRoom4(FourRoomsEnv_ToRoom): 
    def __init__(self, agent_pos=None, goal_pos=None):
        super().__init__(1, 1, agent_pos, goal_pos)

class FourRoomsEnv_Get2Goal(FourRoomsEnv):
    """
    This one assumes that you're somewhere within the room containing the goal.
    """
    def __init__(self, agent_pos=None, goal_pos=None): 
        super().__init__(agent_pos, goal_pos)

    # code for ensuring that the initial positions respect the final goal position.
    def goal_placement(self):
        # where is the agent?
        def to_room_bounds(pos): 
            quadrant_x = int(pos[0] / self.room_w )
            quadrant_y = int(pos[1] / self.room_h )
            return quadrant_x, quadrant_y 
        
        quad_x, quad_y = to_room_bounds(self.agent_pos)
        
        def reject (env, pos):
            
            if pos[0] > quad_x * self.room_w and \
                pos[0] < (quad_x + 1)*self.room_w and \
                pos[1] > quad_y * self.room_h and \
                pos[1] < (quad_y+1)*self.room_h: 
            
                return False # Within the same room
                
            return True # rejecting everything else 
         
        if self._goal_default_pos is not None:
            goal = Goal()
            self.put_obj(goal, *self._goal_default_pos) 
            goal.init_pos, goal.cur_pos = self._goal_default_pos
        else:
            self.place_obj(Goal(), reject_fn=reject)
       
        self.mission = 'Reach the goal' 

register(
    id='MiniGrid-FourRooms-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv'
)

register(
    id='MiniGrid-FourRooms-ToRoom1-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv_ToRoom1'
) 
register(
    id='MiniGrid-FourRooms-ToRoom2-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv_ToRoom2'
) 
register(
    id='MiniGrid-FourRooms-ToRoom3-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv_ToRoom3'
) 
register(
    id='MiniGrid-FourRooms-ToRoom4-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv_ToRoom4'
) 
register(
    id='MiniGrid-FourRooms-Get2Goal-v0',
    entry_point='gym_minigrid.envs:FourRoomsEnv_Get2Goal'
)

