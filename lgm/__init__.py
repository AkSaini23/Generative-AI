"""This module contains functionalities for building, training and running various generative models.

A focus is on reusing code to highlight the similarities between the many different types.
As such, there is a common training loop that is used by pretty much all models.
Each individual model only specifies the "core" of the training function (i.e. how to compute a loss), as well as
various other specific components, such as how to generate samples.

Currently, the focus is on generating images.
However, support for some audio and text datasets is planned for the future.
"""
