import numpy as np
import pytest

from napari_turbobox.undo import UndoStack


def test_undo_stack_initial_state():
    stack = UndoStack(max_size=5)
    assert stack.max_size == 5
    assert stack.size == 0
    assert stack.current_position == -1
    assert not stack.can_undo()
    assert not stack.can_redo()
    assert stack.get_undo_description() is None
    assert stack.get_redo_description() is None
    assert stack.get_memory_usage() == 0


def test_undo_stack_push():
    stack = UndoStack(max_size=3)
    state1 = np.array([[[0, 0], [1, 1]]])

    stack.push(state1, "First Action")
    assert stack.size == 1
    assert stack.current_position == 0
    assert (
        not stack.can_undo()
    )  # First state cannot be undone because current_index is 0, we need index > 0 to undo to
    assert not stack.can_redo()

    state2 = np.array([[[1, 1], [2, 2]]])
    stack.push(state2, "Second Action")
    assert stack.size == 2
    assert stack.current_position == 1
    assert stack.can_undo()
    assert not stack.can_redo()
    assert stack.get_undo_description() == "Second Action"


def test_undo_stack_undo_redo():
    stack = UndoStack(max_size=5)
    s0 = np.array([[[0, 0], [0, 0]]])
    s1 = np.array([[[1, 1], [1, 1]]])
    s2 = np.array([[[2, 2], [2, 2]]])

    stack.push(s0, "Initial")
    stack.push(s1, "Action 1")
    stack.push(s2, "Action 2")

    assert stack.can_undo()
    assert not stack.can_redo()
    assert stack.get_undo_description() == "Action 2"

    # Undo Action 2
    res = stack.undo()
    assert res is not None
    state, desc = res
    assert np.array_equal(state, s1)
    assert desc == "Action 1"
    assert stack.current_position == 1
    assert stack.can_undo()
    assert stack.can_redo()
    assert stack.get_redo_description() == "Action 2"
    assert stack.get_undo_description() == "Action 1"

    # Undo Action 1
    res = stack.undo()
    assert res is not None
    state, desc = res
    assert np.array_equal(state, s0)
    assert desc == "Initial"
    assert stack.current_position == 0
    assert not stack.can_undo()
    assert stack.can_redo()
    assert stack.get_redo_description() == "Action 1"

    # Undo past initial
    assert stack.undo() is None

    # Redo to Action 1
    res = stack.redo()
    assert res is not None
    state, desc = res
    assert np.array_equal(state, s1)
    assert desc == "Action 1"
    assert stack.current_position == 1

    # Redo to Action 2
    res = stack.redo()
    assert res is not None
    state, desc = res
    assert np.array_equal(state, s2)
    assert desc == "Action 2"
    assert stack.current_position == 2
    assert not stack.can_redo()

    # Redo past end
    assert stack.redo() is None


def test_undo_description_names_the_action_undone():
    """The label shown for undo is the action that undo reverts, not the one before it."""
    stack = UndoStack(max_size=5)
    stack.push(np.zeros((0, 2, 3)), "Initial state")
    stack.push(np.ones((1, 2, 3)), "Move/Resize box")
    stack.push(np.zeros((0, 2, 3)), "Delete 1 box(es)")
    assert stack.get_undo_description() == "Delete 1 box(es)"
    stack.undo()
    assert stack.get_undo_description() == "Move/Resize box"
    assert stack.get_redo_description() == "Delete 1 box(es)"
    stack.undo()
    assert stack.get_undo_description() is None  # the initial state is not an action
    assert stack.get_redo_description() == "Move/Resize box"
    stack.redo()
    assert stack.get_undo_description() == "Move/Resize box"


def test_layer_undo_info_after_a_setter_edit():
    """Tutorial 00 case: after a ``bounding_boxes`` edit the undo label is that edit."""
    from napari_turbobox import TurboBoxLayer

    layer = TurboBoxLayer(ndim=3, image_shape=(10, 100, 100))
    layer.add_boxes(np.array([[[0, 0, 0], [5, 50, 50]]]))
    assert layer.get_undo_info()["undo_description"] == "Add 1 box"
    boxes = layer.bounding_boxes.copy()
    boxes[0, 1, 2] = 60
    layer.bounding_boxes = boxes
    assert layer.get_undo_info()["undo_description"] == "Set boxes"
    assert layer.undo()
    assert layer.get_undo_info()["undo_description"] == "Add 1 box"
    assert layer.get_undo_info()["redo_description"] == "Set boxes"


def test_undo_stack_eviction():
    stack = UndoStack(max_size=3)
    s1 = np.array([[[1, 1]]])
    s2 = np.array([[[2, 2]]])
    s3 = np.array([[[3, 3]]])
    s4 = np.array([[[4, 4]]])

    stack.push(s1, "A1")
    stack.push(s2, "A2")
    stack.push(s3, "A3")
    assert stack.size == 3
    assert stack.current_position == 2

    # This push should evict s1
    stack.push(s4, "A4")
    assert stack.size == 3
    assert stack.current_position == 2

    # History is now: A2, A3, A4. Position is at A4.
    # Undo Action 4 -> A3
    res = stack.undo()
    assert res is not None
    assert res[1] == "A3"

    # Undo Action 3 -> A2
    res = stack.undo()
    assert res is not None
    assert res[1] == "A2"

    # Cannot undo past A2
    assert not stack.can_undo()


def test_undo_stack_truncate_on_push():
    stack = UndoStack(max_size=5)
    s0 = np.array([[[0]]])
    s1 = np.array([[[1]]])
    s2 = np.array([[[2]]])
    s3 = np.array([[[3]]])

    stack.push(s0, "A0")
    stack.push(s1, "A1")
    stack.push(s2, "A2")

    stack.undo()  # Now at s1 (position 1)

    # Pushing new state should truncate s2
    stack.push(s3, "A3")
    assert stack.size == 3
    assert stack.current_position == 2
    assert not stack.can_redo()

    res = stack.undo()
    assert res is not None
    assert res[1] == "A1"


def test_undo_stack_disable_enable():
    stack = UndoStack(max_size=5)
    s0 = np.array([[[0]]])
    s1 = np.array([[[1]]])

    stack.push(s0, "A0")
    stack.disable()
    stack.push(s1, "A1")  # Should not be recorded

    assert stack.size == 1
    assert stack.current_position == 0

    stack.enable()
    stack.push(s1, "A1")
    assert stack.size == 2
    assert stack.current_position == 1


def test_undo_stack_clear():
    stack = UndoStack(max_size=5)
    s0 = np.array([[[0]]])
    stack.push(s0, "A0")
    stack.push(s0, "A1")

    stack.clear()
    assert stack.size == 0
    assert stack.current_position == -1
    assert not stack.can_undo()
    assert not stack.can_redo()


def test_undo_stack_repr_and_mem():
    stack = UndoStack(max_size=5)
    s = np.zeros((10, 2, 2))
    stack.push(s, "A0")

    usage = stack.get_memory_usage()
    assert usage > s.nbytes

    r = repr(stack)
    assert "UndoStack" in r
    assert "size=1/5" in r


if __name__ == "__main__":
    pytest.main([__file__])
