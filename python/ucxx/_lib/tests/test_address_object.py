# SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause

import multiprocessing as mp
import pickle

import ucxx._lib.libucxx as ucx_api

mp = mp.get_context("spawn")


def test_ucx_address_string():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.get_address()
    org_address_str = org_address.string
    new_address = ucx_api.UCXAddress.create_from_string(org_address_str)
    new_address_str = new_address.string
    assert hash(org_address) == hash(new_address)
    assert bytes(org_address_str) == bytes(new_address_str)


def test_pickle_ucx_address():
    ctx = ucx_api.UCXContext()
    worker = ucx_api.UCXWorker(ctx)
    org_address = worker.get_address()
    org_address_str = org_address.string
    org_address_hash = hash(org_address)
    dumped_address = pickle.dumps(org_address)
    org_address = bytes(org_address)
    new_address = pickle.loads(dumped_address)
    new_address_str = new_address.string

    assert org_address_hash == hash(new_address)
    assert bytes(org_address_str) == bytes(new_address_str)
    assert bytes(org_address) == bytes(new_address)
