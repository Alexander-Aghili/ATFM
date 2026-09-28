"""Depth-first synthetic generation without consuming Python recursion depth."""


def resolve_children(factory, arguments):
    """Drive generators that yield child argument tuples and receive child results.

    Suspend each parent until its child completes, preserving depth-first random
    draws and numbering. Heap memory grows with tree depth; call-stack depth does not.
    """
    stack = [factory(*arguments)]
    result = None
    while stack:
        try:
            child = stack[-1].send(result)
        except StopIteration as completed:
            result = completed.value
            stack.pop()
        else:
            stack.append(factory(*child))
            result = None
    return result
