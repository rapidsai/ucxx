import pytest

from ucxx._lib.tests_cython.test_cython import cython_test_context_getter


@pytest.mark.cython
def test_context_getter():
    return cython_test_context_getter()
