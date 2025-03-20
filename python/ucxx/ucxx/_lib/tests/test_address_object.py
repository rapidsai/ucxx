# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import pickle

import ucxx._lib.libucxx as ucx_api

mp = mp.get_context("spawn")


def test_ucx_address_string():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    org_address_bytes = bytes(org_address)
    new_address = ucx_api.UCXAddress.create_from_string(org_address_bytes)
    new_address_bytes = bytes(new_address)
    assert hash(org_address) == hash(new_address)
    assert org_address_bytes == new_address_bytes


def test_pickle_ucx_address():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.address
    org_address_bytes = bytes(org_address)
    org_address_hash = hash(org_address)
    dumped_address = pickle.dumps(org_address)
    new_address = pickle.loads(dumped_address)
    new_address_bytes = bytes(new_address)

    assert org_address_hash == hash(new_address)
    assert org_address_bytes == new_address_bytes
