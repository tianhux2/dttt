# coding=utf-8
"""
Memory probes for the TTT / GDN masked-diffusion LM, built on the read forward of ``core``.

* ``core``: the read forward with injection points, the memory's per-layer contributions, the write settings of a
  checkpoint.
* ``_cli``: the shared command-line arguments (checkpoint, sources, platform check).
* ``memory_distance``: recall of a planted span by distance, as a share of the copy ceiling.
* ``context_learning``: facts taught early and asked later (passkeys, a new lexicon), masked or left-to-right reads.
* ``key_transplant``: whether one-off recall is lost at the read's keys or in what the memory holds (the write's
  keys transplanted into the read).
* ``doc_switch``: a document switch, the next document read with the previous one's memory carried, reset or off.

The Jacobian lens itself (``lens``, ``calibrate``, ``memory_readout``) stayed in the research repository.  Imports
are left to the modules so that the command lines check ``JAX_PLATFORMS`` before jax is loaded.
"""
