"""Read and write .lhe files."""
from dataclasses import dataclass
from xml.etree import ElementTree


# data structures
@dataclass(frozen=True)
class EventInfo:
    """Contain the info line of an LHE event"""

    nparticles: int
    pid: int
    weight: float
    scale: float
    aqed: float
    aqcd: float


@dataclass(frozen=True)
class Particle:
    """Contain a particle line of an LHE event"""

    pdg_id: int
    status: int
    mother1: int
    mother2: int
    color1: int
    color2: int
    px: float
    py: float
    pz: float
    e: float
    m: float
    lifetime: float
    helicity: float


@dataclass(frozen=True)
class Event:
    info: EventInfo
    particles: list


# lhe parsing
def read_lhe(file_name_or_object):
    """Generate all Event objects from given lhe file."""
    for _, element in ElementTree.iterparse(file_name_or_object):
        if element.tag == "event":
            yield parse_event(element)


def parse_event(element):
    """Return an Event from the given lhe event element."""
    assert element.tag == "event"
    # first and last two lines are blank
    lines = element.text.split("\n")[1:-1]
    info = dataclass_from_string(EventInfo, lines[0])
    particles = [dataclass_from_string(Particle, line) for line in lines[1:]]
    return Event(info, particles)


# utility
def dataclass_from_string(cls, line):
    """Construct an instance of given dataclass from a string line."""
    return cls(
        *(
            type_(field)
            for type_, field in zip(cls.__annotations__.values(), line.split())
        )
    )
