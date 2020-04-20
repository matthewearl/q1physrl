__all__ = (
    'MadWrapper',
    'GaussianSquashedGaussian',
)


import functools

import gym
import numpy as np
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.tf.tf_action_dist import (
    TFActionDistribution, MultiActionDistribution
)
from ray.rllib.utils import (
    try_import_tf, try_import_tfp, SMALL_NUMBER, MIN_LOG_NN_OUTPUT,
    MAX_LOG_NN_OUTPUT
)


tf = try_import_tf()
tfp = try_import_tfp()


class _SquashedGaussianBase(TFActionDistribution):
    """A diagonal gaussian distribution, squashed into bounded support."""

    def __init__(self, inputs, model, low=-1.0, high=1.0):
        """Parameterizes the distribution via `inputs`.

        Args:
            low (float): The lowest possible sampling value
                (excluding this value).
            high (float): The highest possible sampling value
                (excluding this value).
        """
        assert tfp is not None
        mean, log_std = tf.split(inputs, 2, axis=-1)
        self._num_vars = mean.shape[1]
        assert log_std.shape[1] == self._num_vars
        # Clip `std` values (coming from NN) to reasonable values.
        self.log_std = tf.clip_by_value(log_std, MIN_LOG_NN_OUTPUT,
                                        MAX_LOG_NN_OUTPUT)
        # Clip loc too, for numerical stability reasons.
        mean = tf.clip_by_value(mean, -3, 3)
        std = tf.exp(self.log_std)
        self.distr = tfp.distributions.Normal(loc=mean, scale=std)
        assert len(self.distr.loc.shape) == 2
        assert len(self.distr.scale.shape) == 2
        assert np.all(np.less(low, high))
        self.low = low
        self.high = high
        super().__init__(inputs, model)

    def deterministic_sample(self):
        mean = self.distr.mean()
        assert len(mean.shape) == 2
        s = self._squash(mean)
        assert len(s.shape) == 2
        return s

    def logp(self, x):
        assert len(x.shape) >= 2, "First dim batch, second dim variable"
        unsquashed_values = self._unsquash(x)
        log_prob = self.distr.log_prob(value=unsquashed_values)
        return tf.reduce_sum(log_prob -
                             self._log_squash_grad(unsquashed_values), axis=1)

    def _build_sample_op(self):
        s = self._squash(self.distr.sample())
        assert len(s.shape) == 2
        return s

    def _squash(self, unsquashed_values):
        """Squash an array element-wise into the (high, low) range

        Arguments:
            unsquashed_values: values to be squashed

        Returns:
            The squashed values.  The output shape is `unsquashed_values.shape`

        """
        raise NotImplementedError

    def _unsquash(self, values):
        """Unsquash an array element-wise from the (high, low) range

        Arguments:
            squashed_values: values to be unsquashed

        Returns:
            The unsquashed values.  The output shape is `squashed_values.shape`

        """
        raise NotImplementedError

    def _log_squash_grad(self, unsquashed_values):
        """Log gradient of _squash with respect to its argument.

        Arguments:
            squashed_values:  Point at which to measure the gradient.

        Returns:
            The gradient at the given point.  The output shape is
            `squashed_values.shape`.

        """
        raise NotImplementedError


class GaussianSquashedGaussian(_SquashedGaussianBase):
    """A gaussian CDF-squashed Gaussian distribution.

    The distribution will never return low or high exactly, but
    `low`+SMALL_NUMBER or `high`-SMALL_NUMBER respectively.
    """
    # Chosen to match the standard logistic variance, so that:
    #   Var(N(0, 2 * _SCALE)) = Var(Logistic(0, 1))
    _SCALE = 0.5 * 1.8137

    def kl(self, other):
        # KL(self || other) is just the KL of the two unsquashed distributions.
        assert isinstance(other, GaussianSquashedGaussian)

        mean = self.distr.loc
        std = self.distr.scale

        other_mean = other.distr.loc
        other_std = other.distr.scale

        return tf.reduce_sum((other.log_std - self.log_std +
                             (tf.square(std) + tf.square(mean - other_mean)) /
                             (2.0 * tf.square(other_std)) - 0.5), axis=1)

    def entropy(self):
        # Entropy is:
        #   -KL(self.distr || N(0, _SCALE)) + log(high - low)
        # where the latter distribution's CDF is used to do the squashing.

        mean = self.distr.loc
        std = self.distr.scale

        return tf.reduce_sum(tf.log(self.high - self.low) -
                             (tf.log(self._SCALE) - self.log_std +
                              (tf.square(std) + tf.square(mean)) /
                              (2.0 * tf.square(self._SCALE)) - 0.5), axis=1)

    def _log_squash_grad(self, unsquashed_values):
        squash_dist = tfp.distributions.Normal(loc=0, scale=self._SCALE)
        log_grad = squash_dist.log_prob(value=unsquashed_values)
        log_grad += tf.log(self.high - self.low)
        return log_grad

    def _squash(self, raw_values):
        # Make sure raw_values are not too high/low (such that tanh would
        # return exactly 1.0/-1.0, which would lead to +/-inf log-probs).

        values = tfp.bijectors.NormalCDF().forward(raw_values / self._SCALE)
        return (tf.clip_by_value(values, SMALL_NUMBER, 1.0 - SMALL_NUMBER) *
                (self.high - self.low) + self.low)

    def _unsquash(self, values):
        return self._SCALE * tfp.bijectors.NormalCDF().inverse(
            (values - self.low) / (self.high - self.low))


class Q1PhysActionDist(MultiActionDistribution):
    @staticmethod
    def _get_child_dists(action_space):
        child_dist = []
        input_lens = []
        for action in action_space.spaces:
            if isinstance(action, gym.spaces.Box):
                low = action.low
                high = action.high
                assert low.shape == (1,) and high.shape == (1,)

                dist = functools.partial(GaussianSquashedGaussian,
                                         low=low[0], high=high[0])
                action_space = 2
            else:
                dist, action_size = ModelCatalog.get_action_dist(
                    action, config=None)

            child_dist.append(dist)
            input_lens.append(action_size)

        return child_dist, input_lens

    def __init__(self, inputs, model):
        assert isinstance(model.action_space, gym.spaces.Tuple)

        child_dist, input_lens = self._get_child_dists(model.action_space)
        super().__init__(inputs, model, action_space=model.action_space,
                         child_distributions=child_dist,
                         input_lens=input_lens)

    @staticmethod
    def required_model_output_shape(action_space, model_config):
        child_dist, input_lens = Q1PhysActionDist._get_child_dists(
            action_space
        )
        return sum(input_lens)


ModelCatalog.register_custom_action_dist("q1_phys_action_dist",
                                         Q1PhysActionDist)
