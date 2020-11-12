from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class ProceduralEnv(RoomGrid):
    """
    For procedural generation
    """

    def __init__(self, task_sequence, connections, room_layout, room2pos, room_possessions, starting_point, obj_type="ball", room_size=6, seed=None):
        """
        Input:
        task_sequence: a task of subtasks.
        connections: a dictionary with keys "hallways", "unlocked" and "locked". Their values are lists of tuples, with the first element being the room number
        room_layout: a matrix that shows the layout of rooms. -1 means unassigned.
        room2pos: a dictionary mapping room numbers to their row and column values.
        room_possessions: a dictionary mapping room numbers to the elements within the rooms
        starting_point = the room index where the agent starts.
        """

        self.task_sequence = task_sequence
        self.subtask_idx = 0  # this records which subtask should be the target right now.
        self.starting_point = starting_point
        self.connections = connections
        self.room_layout = room_layout
        num_rows, num_cols = self.room_layout.shape
        self.room_possessions = room_possessions
        self.obj_type = obj_type
        self.room_size = room_size
        self.room2pos = room2pos
        self.dir2dooridx = {'right': 0, 'down': 1, 'left': 2, 'up': 2}
        
        super().__init__(room_size=room_size,
                         num_rows=num_rows,
                         num_cols=num_cols,
                         max_steps=100,
                         seed=seed)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        # Add all the lock doors
        for tup in self.connections['locked']:
            room_idx, key_idx, direction = tup
            row, col = self.room2pos[room_idx]
            door_idx = self.dir2dooridx[direction]
            door_color = IDX_TO_COLOR[key_idx-1]
            self.add_door(col, row, door_idx=door_idx, color=door_color, locked=True)

        # Add all the unlocked doors
        for tup in self.connections['unlocked']:
            room_idx, direction = tup
            row, col = self.room2pos[room_idx]
            door_idx = self.dir2dooridx[direction]
            self.add_door(col, row, door_idx=door_idx, locked=False)

        # placing all the keys and the goal
        for room in self.room_possessions:
            row, col = self.room2pos[room]
            if self.room_possessions[room]['goal']:
                # place the goal in thie room
                self.add_object(col, row, kind=self.obj_type)
            for key_idx in self.room_possessions[room]['key']:
                # place the key in this room
                key_color = IDX_TO_COLOR[key_idx - 1]
                self.add_object(col, row, kind='key', color=key_color)

        for tup in self.connections['hallways']:
            room_idx, direction = tup
            row, col = self.room2pos[room_idx]
            wall_idx = self.dir2dooridx[direction]
            self.remove_wall(col, row, wall_idx)

        # place the agent at self.starting_point room
        row, col = self.room2pos[self.starting_point]
        self.place_agent(col, row)

        self.connect_all()

        self.mission = ""

    def reward_condition_postaction(self, prev_fwd_cell, action, fwd_cell, reward, obs):
        curr_subtask = self.task_sequence[self.subtask_idx]
        # evaluates post-action rewards.
        # this needs to check the reward conditions fo reach of the individual subtasks.
        # only issue reward when each of them have been attained.

        score_incremented = False
        # if objective can be seen.
        if curr_subtask.id == 'to_room_with_goal':
            object_idx_layer = obs['image'][:,:,0]
            objective_spotted = OBJECT_TO_IDX[self.obj_type] in object_idx_layer
            if objective_spotted and self.subtask_idx != len(self.task_sequence):
                self.subtask_idx += 1
                score_incremented = True

        # if object is attained: No prior state needed.
        elif curr_subtask.id == 'move2goal':
            objective_attained = False
            if self.obj_type == 'ball':
                objective_attained = isinstance(self.carrying, Ball)
            elif self.obj_type == 'goal':
                objective_attained = isinstance(fwd_cell, Goal)
            if objective_attained and self.subtask_idx != len(self.task_sequence):
                self.subtask_idx += 1
                score_incremented = True
        # if key is attained
        elif curr_subtask.id == 'move2key':
            key_attained = False
            if isinstance(prev_fwd_cell, Key) and fwd_cell is None:
                key_attained = isinstance(self.carrying, Key)
            if key_attained and self.subtask_idx != len(self.task_sequence):
                self.subtask_idx += 1
                score_incremented = True
        # if door is unlocked: need prior states!
        elif curr_subtask.id == 'unlock_correct_door':
            unlocked = False
            if action == self.actions.toggle and isinstance(fwd_cell, Door) and isinstance(prev_fwd_cell, Door):
                unlocked = not fwd_cell.is_locked and prev_fwd_cell.is_locked
            if unlocked and self.subtask_idx != len(self.task_sequence):
                self.subtask_idx += 1
                score_incremented = True
        reward = 0
        done = False
        # if self.subtask_idx == len(self.task_sequence):
        #     reward, done = 1, True
        if score_incremented:
            reward, done = float(self.subtask_idx/len(self.task_sequence)), self.subtask_idx == len(self.task_sequence)
        return done, reward

    def reset_reward_condition_postaction(self):
        self.subtask_idx = 0

    def reward_condition(self, fwd_cell, fwd_pos):
        return False, 0

    def reset_reward_condition(self):
        pass

# NOTE: no registration happens here. registration happens within the rl-starter-files.
