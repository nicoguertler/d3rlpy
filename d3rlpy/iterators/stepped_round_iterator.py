from tkinter import W
from typing import List

from ..dataset import Transition, SteppedTransitionMiniBatch
from .round_iterator import RoundIterator


class SteppedRoundIterator(RoundIterator):

    def __init__(
        self,
        transitions: List[Transition],
        batch_size: int,
        n_steps: int = 1,
        gamma: float = 0.99,
        steps: List[int] = [],
        real_ratio: float = 1.0,
        generated_maxlen: int = 100000,
        shuffle: bool = True,
    ):
        self._steps = steps
        n_frames = max(len(steps), 1)
        super().__init__(
            transitions=transitions,
            batch_size=batch_size,
            n_steps=n_steps,
            gamma=gamma,
            n_frames=n_frames,
            real_ratio=real_ratio,
            generated_maxlen=generated_maxlen,
            shuffle=shuffle
        )

    def __next__(self) -> SteppedTransitionMiniBatch:
        if len(self._generated_transitions) > 0:
            real_batch_size = self._real_batch_size
            fake_batch_size = self._batch_size - self._real_batch_size
            transitions = [self.get_next() for _ in range(real_batch_size)]
            transitions += self._sample_generated_transitions(fake_batch_size)
        else:
            transitions = [self.get_next() for _ in range(self._batch_size)]

        batch = SteppedTransitionMiniBatch(
            transitions,
            steps_list=self._steps,
            n_steps=self._n_steps,
            gamma=self._gamma,
        )

        self._count += 1

        return batch