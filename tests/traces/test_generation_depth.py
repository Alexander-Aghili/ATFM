"""Deep synthetic families must not depend on Python's recursion limit."""
from atfm.traces.generation import resolve_children
from atfm.sim.programs import Program, _clone


def test_depth_first_generation_preserves_child_result_order():
    visits = []

    def branch(depth):
        visits.append(('enter', depth))
        child = yield (depth - 1,) if depth else (None,)
        visits.append(('exit', depth))
        return child + 1

    def factory(depth):
        if depth is None:
            return 0
        return (yield from branch(depth))

    assert resolve_children(factory, (2000,)) == 2001
    assert visits[:2001] == [('enter', n) for n in range(2000, -1, -1)]
    assert visits[2001:] == [('exit', n) for n in range(2001)]


def test_program_clone_handles_deep_families_and_keeps_turns_shared():
    root = Program('0', 'background', 't', 7., [])
    parent = root
    for index in range(1, 2001):
        child = Program(str(index), 'background', 't', 0., [], parent=parent.session_id)
        parent.spawn_at_turn.append((0, child))
        parent = child
    copied = _clone(root, '#copy', 9.)
    assert copied.t_arrival == 9.
    for index in range(2001):
        assert copied.session_id == f'{index}#copy' and copied.turns is root.turns
        assert copied.parent == (f'{index - 1}#copy' if index else None)
        if index < 2000:
            root, copied = root.spawn_at_turn[0][1], copied.spawn_at_turn[0][1]
