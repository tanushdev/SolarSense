import logging
import pytest
from loguru import logger

@pytest.fixture(autouse=True)
def propagate_loguru_to_caplog(caplog):
    # Loguru sink that forwards to standard logging, so pytest's caplog works
    class Propagator(logging.Handler):
        def emit(self, record):
            logging.getLogger(record.name).handle(record)
            
    handler_id = logger.add(Propagator(), level="DEBUG")
    yield
    logger.remove(handler_id)
