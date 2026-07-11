import pytest


@pytest.fixture(scope="session")
def model():
    from musclemimic.utils.retarget.msk_metrics import load_model

    return load_model()
