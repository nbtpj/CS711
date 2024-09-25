import copy
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, OrderedDict, Set, Union
import itertools


@dataclass
class Block:
    label: str
    color: str

    def __str__(self):
        return f"{self.label}_{self.color}"

    @property
    def latex(self) -> str:
        return self.label + "_{" + self.color + "}"


@dataclass
class State:
    pipes: Union[Dict[str, List[Block]], List[List[Block]]]
    n_block: int = None

    def __post_init__(self):
        if isinstance(self.pipes, List):
            pipes = {}
            for pipe in self.pipes:
                pipes[str(pipe[0])] = pipe
            self.pipes = pipes
        for block in list(itertools.chain(*[p for p in self.pipes.values()])):
            if str(block) not in self.pipes:
                self.pipes[str(block)] = []
        self.n_block = len(list(itertools.chain(*[p for p in self.pipes.values()])))

    @property
    def empty_pipes(self) -> List[str]:
        return [k for k, v in self.pipes.items() if len(v) == 0]

    @property
    def non_empty_pipes(self) -> Dict[str, List[Block]]:
        return {k: v for k, v in self.pipes.items() if len(v) > 0}

    @property
    def last_blocks(self) -> List[Block]:
        return [v[-1] for k, v in self.pipes.items() if len(v) > 0]

    @property
    def id(self) -> Set[str]:
        _id = []
        for k, v in self.pipes.items():
            if len(v) > 0:
                _id.append('_'.join([str(b) for b in v]))
        return set(_id)

    def __eq__(self, other):
        intersection = self.id & other.id
        return (len(intersection) == len(self.id) and len(intersection) == len(other.id))

    def __str__(self):
        output = []
        for k, v in self.non_empty_pipes.items():
            output.append('[' + ', '.join([str(b) for b in v]) + ']')
        return '\n'.join(output)

    @property
    def latex(self) -> str:
        lines = []
        for k, v in self.pipes.items():
            line = [b.latex for b in v] + ["0", ] * (self.n_block - len(v))
            line = '&\t'.join(line)
            lines.append(line)
        lines = (r" \\ ").join(lines)
        return r"""\begin{bmatrix}""" + lines + r"""\end{bmatrix}"""


_PLACING_NEW_COST: dict = {
    "orange": 2,
    "gray": 3,
    "yellow": 1,
    "green": 4,
    "other": 1
}


@dataclass
class Action:
    from_pipe_id: str
    to_pipe_id: str

    def transition(self, state: State) -> Tuple[State, float]:
        new_state = copy.deepcopy(state.pipes)
        cost = None
        if len(new_state[self.from_pipe_id]) > 0 and not (
                len(new_state[self.to_pipe_id]) == 0
                and self.to_pipe_id != str(new_state[self.from_pipe_id][-1])):
            blck = new_state[self.from_pipe_id][-1]
            new_state[self.to_pipe_id].append(blck)
            del new_state[self.from_pipe_id][-1]
            is_new_pipe = len(new_state[self.to_pipe_id]) == 1
            if is_new_pipe:
                cost = _PLACING_NEW_COST[blck.color]
            else:
                cost = _PLACING_NEW_COST["other"]
        return State(new_state), cost


def heuristic_estimation(state: State, terminal_state: State) -> float:
    # check new plate cost
    remaining_cost = 0
    for k in terminal_state.pipes:
        current_pipe = state.pipes[k]
        expected_pipe = terminal_state.pipes[k]

        n_match = 0
        for idx in range(min(len(current_pipe), len(expected_pipe))):
            if current_pipe[idx] != expected_pipe[idx]:
                break
            else:
                n_match += 1
        # remaining_cost = - n_match*_PLACING_NEW_COST["other"]
        remaining_cost += max(len(current_pipe) - n_match, len(expected_pipe) - n_match) * _PLACING_NEW_COST["other"]
        if n_match == 0 and len(expected_pipe) > 0:
            remaining_cost = remaining_cost - _PLACING_NEW_COST["other"] + _PLACING_NEW_COST[expected_pipe[0].color]

    return remaining_cost


if __name__ == "__main__":
    initial_state = State([
        [Block("A", "orange"), Block("B", "gray"), Block("A", "green")],
        [Block("C", "green"), Block("B", "yellow")]
    ])
    terminal_state = State([
        [Block("A", "green"), Block("A", "orange"), Block("B", "yellow"), Block("B", "gray"), Block("C", "green")],
    ])

    # A* search
    # buffer of shape list of tuples of (state, total cost, estimate heuristic cost, solution)
    buffer = [[initial_state, 0, heuristic_estimation(initial_state, terminal_state), [initial_state]]]
    searched = [initial_state]


    def print_buffer(iter_, buffer):
        print(f"Iteration {iter_}")
        print(f"Best State: \n{buffer[0][0]};\t\t\t cost: {buffer[0][1]};\t heuristic_cost: {buffer[0][2]}")
        print(f"Buffer:")

        for i, (reference_state, cost, heuristic_val, _) in enumerate(buffer):
            # if iter_ < 5:
            #     print(f"{reference_state.latex}")
            #     print(f"cost: {cost};\t heuristic_cost: {heuristic_val}")
            # else:
                print(f"{reference_state}\n{cost}/{heuristic_val}")
        print('-' * 30)


    for iteration in range(int(1e4)):
        # each state discovery
        buffer = sorted(buffer, key=lambda x: x[1] + x[2])
        # buffer = sorted(buffer, key=lambda x: x[2])

        # select best state to discovery: the state having the lowest value
        state = buffer[0][0]
        h = buffer[0][-1]
        current_cost = buffer[0][1]
        print_buffer(iteration, buffer)

        if state == terminal_state:
            print("Reached terminal state!")
            solutions = buffer[0][-1]
            print("SOLUTION: ")
            for i, h in enumerate(solutions):
                print("ITER: ", i)
                print(h)
            print(f"TOTAL COST: {current_cost}")
            print('-' * 30)
            break
        for from_pipe, v in state.non_empty_pipes.items():
            # from pipes
            for to_pipe, v in state.pipes.items():
                if from_pipe != to_pipe:
                    action = Action(from_pipe, to_pipe)
                    new_state, new_cost = action.transition(state)
                    if new_cost is None:
                        continue
                    new_cost = new_cost + current_cost
                    new_heuristic = heuristic_estimation(new_state, terminal_state)
                    to_add = True
                    if new_state in searched:
                        to_add = False
                        continue
                    for i, (reference_state, cost, heuristic_val, _) in enumerate(buffer):
                        if reference_state == new_state:
                            to_add = False
                            if cost + heuristic_val > new_cost + new_heuristic:
                                buffer[i][1] = new_cost
                                buffer[i][2] = new_heuristic
                                buffer[i][3] = [*h, new_state]
                    if to_add:
                        buffer.append([new_state, new_cost, new_heuristic, [*h, new_state]])
        searched.append(buffer[0][0])
        del buffer[0]
