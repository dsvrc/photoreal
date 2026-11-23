"""Stand-in for libigl: visualize/ca_body imports it at module level but forward rendering never calls it."""


def __getattr__(name):
    def missing(*args, **kwargs):
        raise NotImplementedError(f"igl.{name} is not available in the local render environment")

    return missing
