"""Define exceptions specific to NeuroR."""


class NeuroRError(Exception):
    """Exceptions raised by NeuroR."""


class CorruptedMorphology(NeuroRError):
    """Exception for morphologies that should not be used."""


class ZeroLengthRootSection(NeuroRError):
    """Exception for morphologies that have zero length root sections."""
