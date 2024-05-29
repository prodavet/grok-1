# Copyright 2024 X.AI Corp.
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

import bisect
import functools
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, NamedTuple, Optional, Tuple

import haiku as hk
import jax
import jax.experimental.pjit as pjit
import jax.numpy as jnp
import numpy as np
import sentencepiece
from jax.experimental import mesh_utils
from jax.sharding import PartitionSpec as P
from jax.typing import ArrayLike

import checkpoint as xai_checkpoint
from model import (
    LanguageModelConfig,
    LanguageModelOutput,
    TrainingState,
    apply_rules,
    Memory,
    KVMemory,
)

logger = logging.getLogger(__name__)
rank_logger = logging.getLogger("rank")

TOP_K = 8


class SampleSettings(NamedTuple):
    temperature: ArrayLike
    nucleus_p: ArrayLike
    mask: ArrayLike
    active: ArrayLike


class SampleOutput(NamedTuple):
    token_id: ArrayLike
    prob: ArrayLike
    top_k_token_ids: ArrayLike
    top_k_probs: ArrayLike


def insert_slice(memory: Memory, slice, length, i):
    updated_layers = [
        KVMemory(layer.k, layer.v, step=jnp.array([length])) for layer in slice.layers
    ]
    return jax.tree_map(lambda m, u: jax.lax.dynamic_update_index_in_dim(m, u[0], i, axis=0),
                        memory, Memory(layers=updated_layers))


def pad_to_size(x, size):
    if len(x) > size:
        x = x[-size:]
    return np.pad(x, [0, size - len(x)], mode="constant", constant_values=0)


def top_p_filter(logits: jax.Array, top_p: jax.Array) -> jax.Array:
    sorted_logits = jax.lax.sort(logits, is_stable=False)
    sorted_probs = jax.nn.softmax(sorted_logits)
    threshold_idx = jnp.argmax(jnp.cumsum(sorted_probs, -1) >= 1 - top_p, axis=-1)
    threshold_largest_logits = jnp.take_along_axis(
        sorted_logits, threshold_idx[..., jnp.newaxis], axis=-1
    )
    mask = logits >= threshold_largest_logits
    logits = jnp.where(mask, logits, -1e10)
    return logits


def sample_token(rngs: jax.random.PRNGKey,
                 lm_outputs: LanguageModelOutput,
                 settings: SampleSettings) -> SampleOutput:
    settings = SampleSettings(
        temperature=jnp.expand_dims(settings.temperature, (1, 2)),
        nucleus_p=jnp.expand_dims(settings.nucleus_p, (1, 2)),
        mask=jnp.expand_dims(settings.mask, 1),
        active=settings.active,
    )
    logits = lm_outputs.logits / settings.temperature.astype(lm_outputs.logits.dtype)
    logits = jnp.where(settings.mask, logits, -1e10)
    logits = top_p_filter(logits, settings.nucleus_p.astype(logits.dtype))

    new_token = jax.vmap(jax.random.categorical)(rngs, logits)

    probabilities = jax.nn.softmax(logits)
    token_prob = jnp.take_along_axis(probabilities, jnp.expand_dims(new_token, 1), axis=2)
    token_prob = jnp.squeeze(token_prob, 1)

    top_k_probs, top_k_token_ids = jax.lax.top_k(probabilities, TOP_K)
    top_k_probs = jnp.squeeze(top_k_probs, 1)
    top_k_token_ids = jnp.squeeze(top_k_token_ids, 1)
    return SampleOutput(
        new_token,
        token_prob,
        top_k_token_ids,
        top_k_probs,
    )


@dataclass
class ModelRunner:
    model: LanguageModelConfig

    bs_per_device: float = 2.0
    load_rename_rules: Optional[list[tuple[str, str]]] = None
    load_exclude_rules: Optional[list[str]] = None
    rng_seed: int = 42
    transform_forward: bool = False
    checkpoint_path: str = ""

    def make_forward_fn(self, mesh: Any):
        def forward(tokens):
            out = self.model.make(mesh=mesh)(tokens)
            return out, None

        if self.transform_forward:
            forward = hk.transform(forward)
        return forward

    def initialize(self,
                   init_data,
                   local_mesh_config: tuple[int, int],
                   between_hosts_config: tuple[int, int]):
        num_replicas = math.prod(between_hosts_config)
        self.model.initialize()
        self.model.fprop_dtype = jnp.bfloat16
        num_local_gpus = len(jax.local_devices())

        self.batch_size = int(self.bs_per_device * num_local_gpus * num_replicas)
        self.local_batch_size = self.batch_size // jax.process_count()

        self.local_mesh_config = local_mesh_config
        self.between_hosts_config = between_hosts_config
        self.mesh = make_mesh(self.local_mesh_config, self.between_hosts_config)
        self.forward = self.make_forward_fn(mesh=self.mesh)
        self.logits_fn = hk.transform(lambda tokens: self.forward(tokens)[0])

        self.eval_forward = self.make_forward_fn(mesh=self.mesh)
        self.logits_eval_fn = hk.transform(lambda tokens: self.eval_forward(tokens)[0])

        if self.transform_forward:
            self.state_sharding = self.get_state_sharding(init_data)
            self.init_fn = pjit.pjit(self.init, out_shardings=self.state_sharding)

    def init(self, rng: jax.Array, data) -> TrainingState:
        rng, init_rng = jax.random.split(rng)
        params = self.forward.init(init_rng, data["inputs"])
        return TrainingState(params=params)

    def get_state_sharding(self, init_data):
        with self.mesh:
            shapes = jax.eval_shape(self.init, rng, init_data)
            return jax.tree_util.tree_map_with_path(
                apply_rules(self.model.partition_rules()),
                shapes,
            )

    def load_or_init(self,
                     init_data: Any,
                     from_checkpoint: bool = True,
                     init_fn: Optional[Callable] = None):
        rng = jax.random.PRNGKey(self.rng_seed)

        if not self.checkpoint_path or not from_checkpoint:
            if init_fn is not None:
                state = init_fn(rng, init_data)
            else:
                state = self.init_fn(rng, init_data)
        else:
            with self.mesh:
                if init_fn:
                    state_shapes = jax.eval_shape(init_fn, rng, init_data)
                else:
                    state_shapes = jax.eval_shape(self.init_fn, rng, init_data)
            init_state = None

            state = xai_checkpoint.restore(
                checkpoint_path=self.checkpoint_path,
                state_shapes=state_shapes,
                mesh=self.mesh,
                between_hosts_config=self.between_hosts_config,
                state_sharding=self.state_sharding,
                init_state=init_state,
                params_only=True,
            )

            del init_state
        return state


@dataclass
class Request:
    prompt: str
    temperature: float
    nucleus_p: float
    rng_seed: int
    max_len: int


@dataclass
class InferenceRunner:
    name: str
    runner: Any
    load: str
    tokenizer_path: str = "/tmp/xai_data/tokenizer.model"
    local_mesh_config: Tuple[int, int] = (1, 1)
    between_hosts_config: Tuple[int, int] = (1, 1)
    pad_sizes: tuple[int] = (1024,)

    def get_pad_bucket(self, size):
        i = bisect.bisect_left(self.pad_sizes, size)
        return self.pad_sizes[min(i, len(self.pad_sizes) - 1)]

    def initialize(self):
        self.runner.transform_forward = True
        dummy_data = dict(
            inputs=np.zeros((1, 256), dtype=np.int32),
            targets=np.zeros((1, 256), dtype=np.int32),
        )
        self.runner.initialize(
            dummy_data,
            local_mesh_config=self.local_mesh_config,
            between_hosts_config=self.between_hosts_config,
        )

        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.tokenizer_path)
        self.vocab_size = self.runner.model.vocab_size
        params = self.runner.load_or_init(dummy_data)
        self.params = params

        @functools.lru_cache
        def lm():
            return self.runner.model.make(mesh=self.runner.mesh)

        # Define functions for model operations
        self.hk_forward = hk.without_apply_rng(hk.transform(lambda tokens: lm()(tokens)))
        self.hk_sample_step = hk.without_apply_rng(hk.transform(
            lambda rngs, last_output: sample_token(rngs, self.hk_forward(last_output), settings)))
        self.hk_new_memory = hk.without_apply_rng(hk.transform(lambda bs, sl: lm().init_memory(bs, sl)))
        self.hk_prefill_memory = hk.without_apply_rng(hk.transform(
            lambda rngs, memory, settings, last_output, prompt, length, rng_seed, new_settings, i:
            prefill_memory(rngs, memory, settings, last_output, prompt, length, rng_seed, new_settings, i)))

        self.setup_pjit_functions()

    def setup_pjit_functions(self):
        ds = P("data")
        ms = self.runner.model.model.get_memory_sharding()
        self.sample_step = pjit.pjit(
            self.hk_sample_step.apply,
            in_shardings=(self.runner.params_sharding, None, ds, ms, None),
            out_shardings=(None, ds, ms),
            donate_argnums=3,
        )
        self.prefill_memory = pjit.pjit(
            functools.partial(self.hk_prefill_memory.apply),
            in_shardings=(
                self.runner.params_sharding,
                None,
                ms,
                None,
                ds,
                None,
                None,
                None,
                None,
                None,
            ),
            out_shardings=(None, ds, ms, None),
            donate_argnums=(2,),
        )
        self.new_memory = pjit.pjit(
            self.hk_new_memory.apply,
            static_argnums=(1, 2),
            out_shardings=ms,
        )

    def run(self):
        """Generator that accepts prompts."""
        batch_size = self.runner.batch_size
        params = self.params
        rngs = jax.random.split(jax.random.PRNGKey(1), batch_size)
        with self.runner.mesh:
            memory = self.new_memory(params, batch_size, self.runner.model.sequence_len)
            settings = SampleSettings(
                temperature=np.zeros((batch_size,), dtype=np.float32),
                nucleus_p=np.zeros((batch_size,), dtype=np.float32),
                mask=np.ones((batch_size, self.vocab_size), dtype=np.int32),
                active=np.zeros((batch_size), dtype=np.int32),
            )
            last_output = SampleOutput(
                token_id=np.zeros((batch_size, 1), dtype=np.int32),
                prob=np.zeros((batch_size, 1), dtype=jnp.bfloat16),
                top_k_token_ids=np.zeros((batch_size, TOP_K), dtype=np.int32),
                top_k_probs=np.zeros((batch_size, TOP_K), dtype=jnp.bfloat16),
            )
            prompt = np.array([300, 400, 500, 600, 600, 700, 800])
            for size in self.pad_sizes:
                if size > self.runner.model.sequence_len:
                    break
                prompt_len = len(prompt)
                prompt = pad_to_size(prompt, size)
                rngs, last_output, memory, settings = self.prefill_memory(
                    params,
                    rngs,
                    memory,
                    settings,
                    last_output,
                    prompt,
                    prompt_len,
                    np.uint64(1),
                    SampleSettings(
                        temperature=np.float32(1),
                        nucleus_p=np.float32(1),
                        mask=np.ones((self.vocab_size,), dtype=np.int32),
                        active=np.zeros((), dtype=np.int32),
                    ),
                    0,
                )
        with self.runner.mesh:
            rngs, last_output, memory = self.sample_step(
                params, rngs, last_output, memory, settings
            )
        all_tokens = []
        free_slots = list(range(batch_size))
        requests = [None] * batch_size
        first_output = [None] * batch_size
        jax.tree_map(lambda x: x.copy_to_host_async(), last_output)
        prev_token = last_output
        step = 0
        total_num_tokens = 0
        total_num_sequences = 0
        with self.runner.mesh:
            while True:
                while free_slots:
                    request: Optional[Request] = yield
                    tokens = self.tokenizer.encode(request.prompt)
                    temperature = request.temperature
                    nucleus_p = request.nucleus_p
                    rng_seed = request.rng_seed
                    i = free_slots.pop()
                    prompt = np.array(tokens, dtype=np.int32)
                    prompt_len = len(prompt)
                    prompt = pad_to_size(prompt, self.get_pad_bucket(prompt.shape[0]))
                    mask = np.ones((self.vocab_size,), dtype=np.int32)
                    new_settings = SampleSettings(
                        temperature=np.float32(temperature),
                        nucleus_p=np.float32(nucleus_p),
                        mask=mask,
                        active=np.ones((), dtype=np.int32),
                    )
                    rng_seed = np.uint64(rng_seed)
                    rngs, last_output, memory, settings = self.prefill_memory(
                        params,
                        rngs,
                        memory,
                        settings,
                        last_output,
                        prompt,
                        prompt_len,
                        rng_seed,
                        new_settings,
                        i,
                    )
                    jax.tree_map(lambda x: x.copy_to_host_async(), last_output)
                    first_output[i] = last_output
                    requests[i
