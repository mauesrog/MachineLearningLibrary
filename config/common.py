"""Common Configuration File.

Alters the behavior of shared modules.

Attributes:
    exception_defaults (:obj:`*`): Defines default values for all optional
        values in Exceptions.

"""
exception_defaults = {
    "error_message": "Unkown reason.",  #: Default message of all general
                                        #: exceptions.
    "incompatible_message": "anything"  #: Default message of all exceptions
                                        #: related to dataset incompatibilty.
}
