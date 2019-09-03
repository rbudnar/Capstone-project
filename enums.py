import enum


class TrainingMode(enum.Enum):
    SINGLE = 1
    MULTI = 2
    CELL_ONLY = 3
    CELL_MULTI = 4


class CellType(enum.Enum):
    HUVEC = "HUVEC"
    HEPG2 = "HEPG2"
    U2OS = "U2OS"
    RPE = "RPE"
