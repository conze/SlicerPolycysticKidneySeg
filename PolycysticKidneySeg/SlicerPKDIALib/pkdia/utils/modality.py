from enum import Enum, EnumMeta


class ModalityEnumMeta(EnumMeta):
    def __contains__(cls, item):
        try:
            cls(item)
        except ValueError:
            return False
        return True


class ModalityEnum(str, Enum, metaclass=ModalityEnumMeta):
    T2 = "MRI T2"
    CT = "CT"
