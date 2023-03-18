from typing import List, Callable, Any


def call_chain(input_: Any, functions: List[Callable]) -> Any:
    for function in functions:
        input_ = function(input_)
    return input_