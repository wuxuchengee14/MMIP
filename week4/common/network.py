"""Use the operating system trust store, including Windows-managed roots.

Certificate validation stays enabled. This also covers torchvision's urllib
pretrained-weight downloads after configure() has initialized networking.
"""
import ssl


def enable_system_trust():
    import truststore
    truststore.inject_into_ssl()
