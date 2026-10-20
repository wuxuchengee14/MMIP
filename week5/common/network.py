"""Use the operating system trust store, including Windows-managed roots.

Certificate validation stays enabled. This also covers torchvision's urllib
pretrained-weight downloads after configure() has initialized networking.
"""
import ssl
import os


def enable_system_trust():
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import truststore
    truststore.inject_into_ssl()
