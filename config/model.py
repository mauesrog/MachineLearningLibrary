"""Model Configuration File.

Alters the behavior of all models.

Attributes:
    model_defaults (:obj:`*`): Defines default values for all optional values
        within models.

"""
model_defaults = {
    "gradient_checker_shape": (200, 10),  #: Matrix dimensions to use with
                                          #: gradient.
    "num_grad_eps": 1e-7,  #: Step size for numerical gradient computation.
    "regularization": 1e-5  #: L2 regularization constant.
}
