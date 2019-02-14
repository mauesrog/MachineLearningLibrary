"""Learner Configuration File.

Alters the behavior of `Learner`.

Attributes:
    learner_defaults (:obj:`*`): Defines default values for all optional values
        within the learner.

"""
learner_defaults = {
    "decay_rate": 5e-2,  #: Default positive rate at which SGD's update rule
                        #: should descend.
    "learning_rate": 1e-4,  #: Default positive rate at which SGD's learning
                            #: rate should decline.
    "max_epoches": 1e4,  #: Default maximum number of epoches before
                         #: interrupting the descent.
    "min_feature_correlation": 0.5,  #: Smallest correlation of a feature to
                                     #: observation set accepted.
    "min_delta_epoch": 1e-6,  #: Smallest accepted difference in model loss
                              #: from one epoch to the next.
    "sgd_k": 30  #: Default number of buckets to use in k-bucket SGD.
}
