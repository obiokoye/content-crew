import warnings

import pytest

from content_crew.knowledge import SourceLibrary

warnings.filterwarnings("ignore", message="A function callback cannot be saved in a checkpoint")


@pytest.fixture
def library() -> SourceLibrary:
    return SourceLibrary.from_dir()
