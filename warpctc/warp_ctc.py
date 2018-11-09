import torch
from . import _warp_ctc
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn import Module


def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


class _CTC(Function):
    @staticmethod
    def forward(ctx, preds, labels, preds_lens, label_lens):
        preds = preds.contiguous()
        grads = torch.zeros(preds.size()).type_as(preds)
        minibatch_size = preds.size(1)
        costs = torch.zeros(minibatch_size)
        _warp_ctc.ctc(preds,
                      grads,
                      labels,
                      label_lens,
                      preds_lens,
                      minibatch_size,
                      costs)
        ctx.save_for_backward(grads)

        ctx.costs = torch.FloatTensor(costs)
        if preds.is_cuda:
            ctx.costs = ctx.costs.cuda()

        return ctx.costs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grads, = ctx.saved_tensors
        T, B, F = grads.size()
        grad_output = torch.cat([grad_output.unsqueeze(0)] * T, dim=0)
        grad_output = torch.cat([grad_output.unsqueeze(-1)] * F, dim=-1)
        return grads * grad_output, None, None, None


_C = _CTC.apply


class CTCLoss(Module):
    r"""
    Args:
        size_average (bool, optional): By default,
                the losses are averaged by minibatch.
                If the field :attr:`size_average`
                is set to ``False``, the losses are instead
                summed for each minibatch. Ignored
                when reduces is ``False``. Default: ``True``
        reduce (bool, optional): By default, the losses are averaged
                or summed over observations for each minibatch
                depending on :attr:`size_average`. When :attr:`reduce`
                is ``False``, returns a loss per batch element instead
                and ignores :attr:`size_average`. Default: ``True``
    Shape:
        preds: :math:`(seqLength, batch, outputDim)`. Tensor contains output from network
        labels: :math:`(X,)`. Tensor contains all the targets of the batch in one sequence
        preds_lens: :math:`(batch,)`. Tensor contains size of each output sequence from the network
        label_lens: :math:`(batch,)`. Tensor contains label length of each example
    """
    def __init__(self, size_average=True, reduces=True):
        super(CTCLoss, self).__init__()
        self.size_average = size_average
        self.reduces = reduces

    def forward(self, preds, labels, preds_lens, label_lens):

        assert len(labels.size()) == 1  # labels must be 1 dimensional
        _assert_no_grad(labels)
        _assert_no_grad(preds_lens)
        _assert_no_grad(label_lens)

        loss = _C(preds, labels, preds_lens, label_lens)

        if self.reduces:
            if self.size_average:
                loss = loss.mean()
            else:
                loss = loss.sum()

        return loss
