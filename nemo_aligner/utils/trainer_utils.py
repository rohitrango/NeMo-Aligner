# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

from nemo.collections.nlp.data.language_modeling.megatron.data_samplers import MegatronPretrainingRandomSampler
from nemo.collections.nlp.data.language_modeling.megatron.megatron_batch_samplers import (
    MegatronPretrainingRandomBatchSampler,
)


def compute_num_steps_per_epoch(
    sampler: Union[MegatronPretrainingRandomSampler, MegatronPretrainingRandomBatchSampler],
    limit_train_batches: Union[int, float] = 1.0,
):
    if not sampler.drop_last:
        raise NotImplementedError("`drop_last=False` is not currently supported")

    num_steps_per_epoch = sampler.total_samples // sampler.global_batch_size

    if limit_train_batches is None or (isinstance(limit_train_batches, float) and limit_train_batches > 1.0):
        limit_train_batches = 1.0

    if limit_train_batches >= 0:
        return compute_limit_batches(num_steps_per_epoch, limit_train_batches)
    else:
        return num_steps_per_epoch


def compute_limit_batches(number_of_batches: int, limit_batches: Union[int, float, None]):
    if limit_batches is None:
        limit_batches = 1.0

    if isinstance(limit_batches, float):
        limit_batches = int(number_of_batches * limit_batches)
    elif isinstance(limit_batches, int):
        limit_batches = min(number_of_batches, limit_batches)
    else:
        raise TypeError(f"Invalid data type of {type(limit_batches)} cannot compute limit batches")

    return limit_batches


def safe_is_divisible(a, b):
    """a safe divisible check to allow b to be 0
    """
    if b == 0:
        return False
    return a % b == 0


def check_progress(
    step: int,
    max_steps: int,
    val_check_interval: int,
    save_interval: int,
    limit_val_batches: Union[int, float, None],
    run_time_exceeded: bool = False,
):
    is_validation_enabled = limit_val_batches != 0 and val_check_interval > 0
    is_save_enabled = save_interval > 0
    is_train_end = step == max_steps

    if is_validation_enabled:
        assert save_interval % val_check_interval == 0, f"{save_interval=} must be divisible by {val_check_interval=}"

    # run validation on the last step
    # or when we hit the val check interval
    run_val = safe_is_divisible(step, val_check_interval) or is_train_end or run_time_exceeded
    run_val &= is_validation_enabled

    # save model at save intervals or last step
    save_model = safe_is_divisible(step, save_interval) or is_train_end or run_time_exceeded
    # sometimes the user will provide a validation metric
    # to save against, so we need to run val when we save
    save_model &= is_save_enabled

    return run_val, save_model, is_train_end
