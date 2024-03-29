# Copyright 2020 Hieu Nguyen
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
# ==============================================================================

from modelling.layers.gdn import GDN
import torch


def test_output():
    inputs = torch.rand(2,3,4,5)
    outputs = GDN(3, inverse=False, relu=False)(inputs)
    expected_outputs = inputs/torch.sqrt(1+ 0.1*(inputs**2))
    diff = torch.abs(expected_outputs-outputs)
    error = diff.max()
    # print(error)
    assert (error <= 1e-6), "failed gdn output test"


def test_igdn_output():
    inputs = torch.rand(2,3,4,5)
    outputs = GDN(3, inverse=True, relu=False)(inputs)
    expected_outputs = inputs*torch.sqrt(1+ 0.1*(inputs**2))
    diff = torch.abs(expected_outputs-outputs)
    error = diff.max()
    # print(error)
    assert (error <= 1e-6), "failed igdn output test"


def test_rgdn_output():
    inputs = torch.rand(2,3,4,5)-0.5
    outputs = GDN(3, inverse=False, relu=True)(inputs)
    inputs = torch.max(inputs, torch.tensor(0.))
    expected_outputs = inputs/torch.sqrt(1+ 0.1*(inputs**2))
    diff = torch.abs(expected_outputs-outputs)
    error = diff.max()
    # print(error)
    assert (error <= 1e-6), "failed rgdn output test"


def test_has_grad():
    inputs = torch.rand(2,3,4,5)
    layer = GDN(3, inverse=False, relu=False)
    outputs = layer(inputs)
    x = outputs.mean()
    x.backward()
    for name, param in layer.named_parameters():
        # print(name, param.grad)
        assert param.grad is not None
        

if __name__ == "__main__":
    test_output()
    test_igdn_output()
    test_rgdn_output()
    test_has_grad()