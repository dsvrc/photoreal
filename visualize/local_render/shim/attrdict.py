"""Stand-in for the attrdict package (broken on Python >= 3.10): a dict with recursive attribute access."""


class AttrDict(dict):
    def __getattr__(self, name):
        try:
            value = self[name]
        except KeyError as e:
            raise AttributeError(name) from e
        return AttrDict(value) if isinstance(value, dict) and not isinstance(value, AttrDict) else value

    def __setattr__(self, name, value):
        self[name] = value
