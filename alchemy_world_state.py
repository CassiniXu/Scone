"""Contains class for the Alchemy world state."""

from fsa import WorldState
from alchemy_fsa import AlchemyFSA

# Immutable world state. Action execution returns a new state.
class AlchemyWorldState(WorldState):
    """ The Alchemy world state definition.

    Attributes:
        _beakers (list of list of str): Beakers in the world state.
    """
    def __init__(self, string=None):
        self._beakers = [[]] * 7
        if string:
            string = [beaker.split(':')[1] for beaker in string.split()]
            self._beakers = []
            for beaker in string:
                if beaker == '_':
                    self._beakers.append([])
                else:
                    self._beakers.append(list(beaker))
        else:
            self._beakers = [[]] * 7

    def __eq__(self, other):
        return isinstance(other, AlchemyWorldState) and self._beakers == other.beakers()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return ' '.join([str(i) +
                         ':' +
                         ''.join(beaker) if beaker else str(i) +
                         ':_' for i, beaker in zip(range(1, 8), self._beakers)])

    def __len__(self):
        return len(self._beakers)

    def __iter__(self):
        return self._beakers.__iter__()

    def beakers(self):
        """ Returns the beakers for the world state. """
        return self._beakers

    def components(self):
        """Returns the beakers."""
        return self.beakers()

    def set_beakers(self, beakers):
        """ Sets the beakers of this class to something else.

        Inputs:
            beakers (list of list of str): The beakers to set.
        """
        self._beakers = beakers

    def set_beaker(self, index, new_value):
        """ Resets the units for a specific beaker.

        Inputs:
            index (int): The beaker to reset.
            new_value (list of str): The new values for the beaker.
        """
        self._beakers[index] = new_value

    def pop(self, beaker):
        """ Removes a unit from a beaker.

        Inputs:
            beaker (int): The beaker to pop from.

        Returns:
            AlchemyWorldState, representing the world state after popping.
        """
        beaker -= 1
        if self._beakers[beaker]:
            new_world_state = AlchemyWorldState()
            new_world_state.set_beakers(self._beakers[:])
            new_world_state.set_beaker(beaker, self._beakers[beaker][:-1])
            return new_world_state
        return None

    def push(self, beaker, color):
        """ Adds a new unit to a beaker.

        Inputs:
            beaker (int): The beaker to add to.
            color (str): The color to add.
        Returns:
            AlchemyWorldState, representing the world state after pushing.
        """
        beaker -= 1
        new_world_state = AlchemyWorldState()
        new_world_state.set_beakers(self._beakers[:])
        new_world_state.set_beaker(beaker, self._beakers[beaker] + [color])
        return new_world_state

    def execute_seq(self, actions):
        fsa = AlchemyFSA(self)
        for action in actions:
            peek_world_state = fsa.peek_complete_action(*action)
            if peek_world_state:
                fsa.feed_complete_action(*action)
        return fsa.world_state()

    def distance(self, other_world_state):
        """Implement this if you want."""
        return 0.
