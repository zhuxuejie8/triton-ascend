# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import ast
from dataclasses import dataclass


<<<<<<< HEAD:third_party/ascend/unittest/generalization_cases/conftest.py
@pytest.fixture(scope="session", autouse=True)
def assign_npu(worker_id):
    npu_count = torch.npu.device_count()
    if worker_id == "master":
        npu_id = 0
    else:
        idx = int(worker_id.replace("gw", ""))
        npu_id = idx % npu_count
    torch.npu.set_device(npu_id)
=======
@dataclass(frozen=True)
class SymbolicExpr:
    text: str

    @classmethod
    def from_ast(cls, node: ast.AST) -> "SymbolicExpr":
        if isinstance(node, ast.Name):
            return cls(text=node.id)
        if isinstance(node, ast.Constant):
            return cls(text=str(node.value))
        return cls(text=ast.unparse(node))

    def __str__(self) -> str:
        return self.text
>>>>>>> release-3.2.2-0625-b79d137:third_party/ascend/backend/runtime/dsl_analysis/symbolic_expr.py
