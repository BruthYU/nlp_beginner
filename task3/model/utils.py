import torch
import torch.nn as nn

def get_mask(seq_batch,seq_lengths):
    #pos value =1,pos padding =0
    batch_size  = seq_batch.size()[0]
    max_len = torch.max(seq_lengths) #Why here is torch.max?
    mask = torch.ones(batch_size,max_len,dtype=torch.float)
    mask[seq_batch[:,:max_len]==0] = 0.0
    return mask

def masked_softmax(tensor,mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, sequence_len2, sequence_len2).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1,tensor_shape[-1])

    #reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor*reshaped_mask,dim=-1)
    result = result*reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)

def weighted_sum(tensor,weights,mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask

def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.

    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.

    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add


def correct_predictions(output_probabilities, targets):
    """
    Compute the number of predictions that match some target classes in the
    output of a model.

    Args:
        output_probabilities: A tensor of probabilities for different output
            classes.
        targets: The indices of the actual target classes.

    Returns:
        The number of correct predictions in 'output_probabilities'.
    """
    _, out_classes = output_probabilities.max(dim=1)
    correct = (out_classes == targets).sum()
    return correct.item()


if __name__ == "__main__":
    tensor = torch.randn((3,2))
    mask = torch.randint(0,2,(3,2))
    result = tensor*mask
    result = nn.functional.softmax(result, dim=-1)

    result = result*mask
    a = result.sum(dim=-1, keepdim=True) + 1e-13
    result = result/a
    pass


    # mask = torch.randint(0,2,(5,4,3))
    #
    # tensor_shape = tensor.size()
    # reshaped_tensor = tensor.view(-1, tensor_shape[-1])
    #
    # while mask.dim() < tensor.dim():
    #     mask = mask.unsqueeze(1)
    # mask = mask.expand_as(tensor).contiguous().float()
    # reshaped_mask = mask.view(-1, mask.size()[-1])


    pass