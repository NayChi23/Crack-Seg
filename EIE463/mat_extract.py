import numpy as np


def descriptor_mat(image_tensor, extractor, layer, facet, bin: bool = False, include_cls: bool = False, device='cuda'):
    """
    Extract descriptors from pretrained DINO model; Create an adj matrix from descriptors
    @param image_tensor: Tensor of size (batch, height, width)
    @param extractor: Initialized model to extract descriptors from
    @param layer: Layer to extract the descriptors from
    @param facet: Facet to extract the descriptors from (key, value, query)
    @param bin: apply log binning to the descriptor. default is False.
    @param include_cls: To include CLS token in extracted descriptor
    @param device: Training device
    @return: W: adjacency matrix, F: feature matrix, D: row wise diagonal of W
    """

    # images to descriptors.
    # input is a tensor of size batch X height X width,
    # output is size: batch X 1 X height/patchsize * width/patchsize X descriptor feature size
    descriptor = extractor.extract_descriptors(image_tensor.to(device), layer, facet, bin, include_cls).cpu().numpy()

    # batch X height/patchsize * width/patchsize X descriptor feature size
    descriptor = np.squeeze(descriptor, axis=1)

    # batch * height/patchsize * width/patchsize X descriptor feature size
    descriptor = descriptor.reshape((descriptor.shape[0] * descriptor.shape[1], descriptor.shape[2]))

    # descriptor feature size X batch * (height* width/(patchsize ^2))
    F = descriptor

    # descriptors affinities matrix
    W = F @ F.T
    W = W * (W > 0)
    # norm
    W = W / W.max()

    # D is row wise sum diagonal of W
    D = np.diag(np.sum(W, axis=-1))
    D[D < 1e-12] = 1.0  # Prevent division by zero. defualt threshold==1e-12

    return W, F, D
