# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/jik876/hifi-gan under the MIT license.

from typing import Any


class AttrDict(dict):
    def __init__(self: "AttrDict", *args: Any, **kwargs: Any) -> None:
        values = dict(*args, **kwargs)
        super().__init__({key: self._convert(value) for key, value in values.items()})
        self.__dict__ = self

    @classmethod
    def _convert(cls, value: Any) -> Any:
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            return cls(value)
        if isinstance(value, list):
            return [cls._convert(item) for item in value]
        return value
