from __future__ import division
import math
import warnings

import numpy

from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import optimizer


_default_hyperparam = optimizer.Hyperparameter()
_default_hyperparam.alpha = 0.01
_default_hyperparam.beta1 = 0.9
_default_hyperparam.beta2 = 0.999
_default_hyperparam.eps = 1e-3
_default_hyperparam.eta = 1.0
_default_hyperparam.weight_decay_rate = 0
_default_hyperparam.amsgrad = False


def _learning_rate(hp, t):
    if t == 0:
        raise RuntimeError(
            'Can\'t determine the learning rate of Yogi optimizer '
            'because the update steps have not been started.')
    fix1 = 1. - math.pow(hp.beta1, t)
    fix2 = 1. - math.pow(hp.beta2, t)
    return hp.alpha * math.sqrt(fix2) / fix1


class YogiRule(optimizer.UpdateRule):

    """Update rule of Yogi optimization algorithm.

    See: `Adaptive Methods for Nonconvex Optimization \
          <https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization>`_

    See :class:`~chainer.optimizers.Adam` for weight decay and AMSGrad options.

    Args:
        parent_hyperparam (~chainer.optimizer.Hyperparameter): Hyperparameter
            that provides the default values.
        alpha (float): Coefficient of learning rate.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        eta (float): Schedule multiplier, can be used for warm restarts.
        weight_decay_rate (float): Weight decay rate.
        amsgrad (bool): Whether to use the AMSGrad variant of Yogi.

    """
    _kernel = None
    _amsgrad_kernel = None

    def __init__(self, parent_hyperparam=None,
                 alpha=None, beta1=None, beta2=None, eps=None,
                 eta=None, weight_decay_rate=None, amsgrad=None):
        super(YogiRule, self).__init__(
            parent_hyperparam or _default_hyperparam)
        if alpha is not None:
            self.hyperparam.alpha = alpha
        if beta1 is not None:
            self.hyperparam.beta1 = beta1
        if beta2 is not None:
            self.hyperparam.beta2 = beta2
        if eps is not None:
            self.hyperparam.eps = eps
        if eta is not None:
            self.hyperparam.eta = eta
        if weight_decay_rate is not None:
            self.hyperparam.weight_decay_rate = weight_decay_rate
        if amsgrad is not None:
            self.hyperparam.amsgrad = amsgrad

    def init_state(self, param):
        xp = backend.get_array_module(param.data)
        with cuda.get_device_from_array(param.data):
            self.state['m'] = xp.zeros_like(param.data)
            self.state['v'] = xp.zeros_like(param.data)
            if self.hyperparam.amsgrad:
                self.state['vhat'] = xp.zeros_like(param.data)

        # For iDeep
        if (isinstance(param.data, intel64.mdarray)
                and intel64.inputs_all_ready((self.state['m'],))
                and intel64.inputs_all_ready((self.state['v'],))):
            self.state['m'] = intel64.ideep.array(
                self.state['m'], itype=intel64.ideep.wgt_array)
            self.state['v'] = intel64.ideep.array(
                self.state['v'], itype=intel64.ideep.wgt_array)

    def update_core_cpu(self, param):
        grad = param.grad
        if grad is None:
            return
        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Yogi optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        m, v = self.state['m'], self.state['v']
        sqgrad = grad * grad
        if (isinstance(m, intel64.mdarray)
                and isinstance(v, intel64.mdarray)):
            m.inplace_axpby(1.0, 1.0 - hp.beta1, grad - m)
            v.inplace_axpby(
                1.0, 1.0 - hp.beta2, numpy.sign(sqgrad - v) * sqgrad)
            if hp.amsgrad:
                vhat = self.state['vhat']
                numpy.maximum(vhat, v, out=vhat)
            else:
                vhat = v
            param.data.inplace_axpby(
                1.0 - hp.weight_decay_rate, -hp.eta,
                self.alpha_t * m / (numpy.sqrt(vhat) + hp.eps))
        else:
            m += (1 - hp.beta1) * (grad - m)
            v += (1 - hp.beta2) * numpy.sign(sqgrad - v) * sqgrad
            if hp.amsgrad:
                vhat = self.state['vhat']
                numpy.maximum(vhat, v, out=vhat)
            else:
                vhat = v
            param.data -= hp.eta * (
                self.alpha_t * m / (numpy.sqrt(vhat) + hp.eps) +
                hp.weight_decay_rate * param.data)

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return

        hp = self.hyperparam
        eps = grad.dtype.type(hp.eps)
        if hp.eps != 0 and eps == 0:
            raise ValueError(
                'eps of Yogi optimizer is too small for {} ({})'.format(
                    grad.dtype.name, hp.eps))
        if hp.amsgrad:
            if YogiRule._amsgrad_kernel is None:
                YogiRule._amsgrad_kernel = cuda.elementwise(
                    'T grad, T alpha_t, T one_minus_beta1, T one_minus_beta2, '
                    'T eps, T eta, T weight_decay_rate',
                    'T param, T m, T v, T vhat',
                    '''m += one_minus_beta1 * (grad - m);
                       T sqgrad = grad * grad;
                       v += one_minus_beta2 * sign(sqgrad - v) * sqgrad;
                       vhat = max(vhat, v);
                       param -= eta * (alpha_t * m / (sqrt(vhat) + eps) +
                                       weight_decay_rate * param);''',
                    'adam')
            YogiRule._amsgrad_kernel(
                grad, self.alpha_t, 1 - hp.beta1,
                1 - hp.beta2, hp.eps,
                hp.eta, hp.weight_decay_rate,
                param.data, self.state['m'], self.state['v'],
                self.state['vhat'])
        else:
            if YogiRule._kernel is None:
                YogiRule._kernel = cuda.elementwise(
                    'T grad, T alpha_t, T one_minus_beta1, T one_minus_beta2, '
                    'T eps, T eta, T weight_decay_rate',
                    'T param, T m, T v',
                    '''m += one_minus_beta1 * (grad - m);
                       T sqgrad = grad * grad;
                       v += one_minus_beta2 * sign(sqgrad - v) * sqgrad;
                       param -= eta * (alpha_t * m / (sqrt(v) + eps) +
                                       weight_decay_rate * param);''',
                    'adam')
            YogiRule._kernel(grad, self.alpha_t, 1 - hp.beta1,
                             1 - hp.beta2, hp.eps,
                             hp.eta, hp.weight_decay_rate,
                             param.data, self.state['m'], self.state['v'])

    @property
    def alpha_t(self):
        return _learning_rate(self.hyperparam, self.t)

    @property
    def lr(self):
        warnings.warn(
            'YogiRule.lr has been renamed to YogiRule.alpha_t. '
            'Use of YogiRule.lr is deprecated in Chainer v6.',
            DeprecationWarning)
        return self.alpha_t


class Yogi(optimizer.GradientMethod):

    """Yogi optimizer.

    See: `Adaptive Methods for Nonconvex Optimization \
          <https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization>`_


    See :class:`~chainer.optimizers.Adam` for weight decay and AMSGrad options.

    Args:
        alpha (float): Coefficient of learning rate.
        beta1 (float): Exponential decay rate of the first order moment.
        beta2 (float): Exponential decay rate of the second order moment.
        eps (float): Small value for the numerical stability.
        eta (float): Schedule multiplier, can be used for warm restarts.
        weight_decay_rate (float): Weight decay rate.
        amsgrad (bool): Whether to use AMSGrad variant of Yogi.

    """

    def __init__(self,
                 alpha=_default_hyperparam.alpha,
                 beta1=_default_hyperparam.beta1,
                 beta2=_default_hyperparam.beta2,
                 eps=_default_hyperparam.eps,
                 eta=_default_hyperparam.eta,
                 weight_decay_rate=_default_hyperparam.weight_decay_rate,
                 amsgrad=_default_hyperparam.amsgrad):
        super(Yogi, self).__init__()
        self.hyperparam.alpha = alpha
        self.hyperparam.beta1 = beta1
        self.hyperparam.beta2 = beta2
        self.hyperparam.eps = eps
        self.hyperparam.eta = eta
        self.hyperparam.weight_decay_rate = weight_decay_rate
        self.hyperparam.amsgrad = amsgrad

    alpha = optimizer.HyperparameterProxy('alpha')
    beta1 = optimizer.HyperparameterProxy('beta1')
    beta2 = optimizer.HyperparameterProxy('beta2')
    eps = optimizer.HyperparameterProxy('eps')
    eta = optimizer.HyperparameterProxy('eta')
    weight_decay_rate = optimizer.HyperparameterProxy('weight_decay_rate')
    amsgrad = optimizer.HyperparameterProxy('amsgrad')

    def create_update_rule(self):
        return YogiRule(self.hyperparam)

    @property
    def alpha_t(self):
        return _learning_rate(self.hyperparam, self.t)

    @property
    def lr(self):
        warnings.warn(
            'Yogi.lr has been renamed to YogiRule.alpha_t. '
            'Use of Yogi.lr is deprecated in Chainer v6.',
            DeprecationWarning)
        return self.alpha_t
