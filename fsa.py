""" Abstract FSA methods and classes """

from abc import abstractmethod

EOS = "_EOS"
ACTION_SEP = ";"
NO_ARG = "_NONE"

class WorldState():
    """Abstract class for a world state."""
    @abstractmethod
    def execute_seq(self, actions):
        """Execute a sequence of actions on a world state.

        Args:
            actions (list of any): The sequence of actions to execute.
        """
        pass

    @abstractmethod
    def distance(self, other_world_state):
        """ Computes a distance between itself and another world state of the same type.

        Args:
            other_world_state (WorldState): the world state to compare with.

        Returns:
            float, representing the distance.
        """
        pass


class ExecutionFSA():
    """Abstract class for an FSA that can execute various actions."""
    @abstractmethod
    def is_valid(self):
        """Returns whether the current FSA state is valid."""
        pass

    @abstractmethod
    def is_in_action(self):
        """Returns whether the current FSA state is in an action."""
        pass

    @abstractmethod
    def world_state(self):
        """Returns the current world state."""
        pass

    @abstractmethod
    def valid_feeds(self):
        """Returns the valid actions that can be executed."""
        pass

    @abstractmethod
    def peek_complete_action(self, action, arg1, arg2):
        """Returns the world state that would happen if executing action with arg1 and arg2."""
        pass

    @abstractmethod
    def feed_complete_action(self, action, arg1, arg2):
        """Updates the world state of the FSA using action with arg1 and arg2."""
        pass
