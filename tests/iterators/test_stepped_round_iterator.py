import numpy as np
import pytest

from d3rlpy.dataset import Episode, Transition
from d3rlpy.iterators.stepped_round_iterator import SteppedRoundIterator


@pytest.mark.parametrize("episode_size", [100])
@pytest.mark.parametrize("n_episodes", [2])
@pytest.mark.parametrize("observation_size", [10])
@pytest.mark.parametrize("action_size", [2])
@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("steps", [[], [0, 1, 5]])
@pytest.mark.parametrize("shuffle", [False, True])
@pytest.mark.parametrize("real_ratio", [0.5])
@pytest.mark.parametrize("generated_maxlen", [10])
def test_stepped_round_iterator(
    episode_size,
    n_episodes,
    observation_size,
    action_size,
    batch_size,
    steps,
    shuffle,
    real_ratio,
    generated_maxlen,
):
    episodes = []
    for _ in range(n_episodes):
        observations = np.random.random((episode_size, observation_size))
        actions = np.random.random((episode_size, action_size))
        rewards = np.random.random(episode_size)
        episode = Episode(
            (observation_size,),
            action_size,
            observations,
            actions,
            rewards,
            terminal=False,
        )
        episodes.append(episode)

    orig_transitions = []
    for episode in episodes:
        orig_transitions += episode.transitions

    iterator = SteppedRoundIterator(
        orig_transitions,
        batch_size,
        steps=steps,
        real_ratio=real_ratio,
        generated_maxlen=generated_maxlen,
        shuffle=shuffle
    )

    # check without generated transitions
    n_frames = max(len(steps), 1)
    count = 0
    for batch in iterator:
        assert batch.observations.shape == (batch_size, observation_size * n_frames)
        assert batch.actions.shape == (batch_size, action_size)
        assert batch.rewards.shape == (batch_size, 1)
        count += 1
    assert count == episode_size * n_episodes // batch_size
    assert len(iterator) == episode_size * n_episodes // batch_size

    # check adding generated transitions
    transitions = []
    for _ in range(episode_size):
        transition = Transition(
            (observation_size,),
            action_size,
            np.random.random(observation_size),
            np.random.random(action_size),
            np.random.random(),
            np.random.random(observation_size),
            terminal=True,
        )
        transitions.append(transition)
    iterator.add_generated_transitions(transitions)
    assert len(iterator.generated_transitions) == generated_maxlen

    # check with generated transitions
    count = 0
    for batch in iterator:
        assert batch.observations.shape == (batch_size, observation_size * n_frames)
        assert batch.actions.shape == (batch_size, action_size)
        assert batch.rewards.shape == (batch_size, 1)
        assert batch.terminals.sum() == int(batch_size * (1 - real_ratio))
        count += 1
    real_batch_size = real_ratio * batch_size
    assert count == episode_size * n_episodes // real_batch_size
    assert len(iterator) == episode_size * n_episodes // real_batch_size