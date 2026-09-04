"""Turning somebody else's data into ours.

What lives here is run once, by hand, to prepare something the application then
reads for the rest of its life: a model grid downloaded from the group that
computed it, converted into the cube and the spectra the SED fitter wants.

That is a different thing from ``processing``, which runs per target, in a
worker, every time somebody asks a question. Nothing here is on that path -
it is invoked from a management command and writes files, and if it is slow or
wants three gigabytes of memory for a minute that is nobody's problem.
"""
