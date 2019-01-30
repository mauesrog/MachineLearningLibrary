"""Learner Configuration File.

Alters the behavior of `Learner`.

Attributes:
    learner_defaults (:obj:`*`): Defines default values for all optional values
        within the learner.

"""
learner_defaults = {
    "decay_rate": 0.5,  #: Default positive rate at which SGD's update rule
                        #: should descend.
    "learning_rate": 1e-3,  #: Default positive rate at which SGD's learning
                            #: rate should decline.
    "max_epoches": 1e4,  #: Default maximum number of epoches before
                         #: interrupting the descent.
    "min_delta_epoch": 1e-6,  #: Smallest accepted difference in model loss
                              #: from one epoch to the next.
    "sgd_k": 50  #: Default number of buckets to use in k-bucket SGD.
}
