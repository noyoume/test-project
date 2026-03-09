# 02. Core GCL Module 분석 (`egnn/models/gcl.py`)

> 논문: E(n) Equivariant Graph Neural Networks (arXiv:2102.09844)
> 분석 대상 파일: `egnn/models/gcl.py` (351 lines)

---

## 1. 파일 구조 Overview

이 파일은 EGNN 논문에서 제안하는 그래프 레이어들을 모두 포함하며, 기본 GNN부터 equivariant 확장까지 계층적으로 구성되어 있다.

```
unsorted_segment_sum / unsorted_segment_mean   <-- aggregation utilities
        |
      MLP (L4-20)                              <-- 범용 4-layer MLP
        |
  GCL_basic (L23-46)                           <-- abstract base class
     /        \
  GCL (L50-105)   GCL_rf (L108-142)            <-- 일반 GNN / Radial Field
        |
  E_GCL (L145-251)                             <-- **핵심: EGNN layer** (Eq. 3-6)
     |        \
  E_GCL_vel    GCL_rf_vel                       <-- velocity 확장 (Eq. 7)
  (L254-284)   (L289-332)
```

**설계 원칙**: 일반 GCL에서 좌표 업데이트를 추가하여 E_GCL을 구성하고, 여기에 속도항을 더해 E_GCL_vel로 확장하는 점진적 구조이다. 비교 실험을 위해 Radial Field 변형도 같은 파일에 포함되어 있다.

---

## 2. Utility Functions

### 2.1 `unsorted_segment_sum` (L335-341)

```python
def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))       # (N, D)
    result = data.new_full(result_shape, 0)           # zero-initialized
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))  # (E,) -> (E, D)
    result.scatter_add_(0, segment_ids, data)         # in-place scatter add
    return result
```

**역할**: 논문 Eq. 5의 `mi = sum_j(mij)` 구현. edge message를 source node 기준으로 합산한다.

**Tensor Shape 추적**:
- `data`: `(E, D)` -- E개의 edge에 대한 D차원 feature
- `segment_ids`: `(E,)` -> `unsqueeze + expand` -> `(E, D)`
- `result`: `(N, D)` -- N개의 node에 대한 aggregated message

**핵심**: `scatter_add_`는 `result[segment_ids[i][j]] += data[i][j]`를 수행. 같은 node로 향하는 모든 edge message를 더한다.

### 2.2 `unsorted_segment_mean` (L344-351)

```python
def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)   # division-by-zero 방지
```

**역할**: sum 대신 mean aggregation. 좌표 업데이트(coord_model)에서 사용된다.

**왜 좌표 업데이트에는 mean을 쓰는가?**
좌표는 물리적 공간의 값이므로, node의 이웃 수(degree)에 따라 이동량이 비례하면 안 된다. degree가 큰 node가 과도하게 이동하는 것을 방지하기 위해 mean을 사용한다. 반면 node feature 업데이트(node_model)에서는 sum을 사용하는데, 이는 이웃 수가 많을수록 더 많은 정보를 반영하는 것이 자연스럽기 때문이다.

**`count.clamp(min=1)`**: 이웃이 없는 고립 node에서 0으로 나누는 것을 방지한다.

---

## 3. MLP (L4-20)

```python
class MLP(nn.Module):
    """ a simple 4-layer MLP """
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),      # (nin) -> (nh)
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),       # (nh) -> (nh)
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),       # (nh) -> (nh)
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),     # (nh) -> (nout)
        )

    def forward(self, x):
        return self.net(x)
```

**참고**: 이 MLP 클래스는 gcl.py 내부에서는 사용되지 않는다. 외부 모듈에서 범용적으로 사용하기 위한 유틸리티이다. E_GCL 등의 내부 MLP들은 `nn.Sequential`로 직접 구성한다.

**LeakyReLU(0.2)**: 음수 영역에서 기울기 0.2. 이 파일 내에서 GCL_rf 계열이 LeakyReLU를 쓰고, E_GCL 계열은 기본값으로 ReLU를 쓴다.

---

## 4. GCL_basic (L23-46) -- Abstract Base Class

```python
class GCL_basic(nn.Module):
    def __init__(self):
        super(GCL_basic, self).__init__()

    def edge_model(self, source, target, edge_attr):
        pass    # subclass에서 구현

    def node_model(self, h, edge_index, edge_attr):
        pass    # subclass에서 구현

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index          # row: source indices, col: target indices
        edge_feat = self.edge_model(x[row], x[col], edge_attr)   # 논문 Eq. 2의 phi_e
        x = self.node_model(x, edge_index, edge_feat)            # 논문 Eq. 2의 phi_h
        return x, edge_feat
```

**Message Passing 패러다임의 골격**:
- `edge_index`: `(2, E)` 텐서. `row`는 source node, `col`는 target node.
- `x[row]`로 source node feature를, `x[col]`로 target node feature를 gather하여 edge 단위 연산 수행.
- edge_model -> aggregation -> node_model 순서.

**`edge_index` 컨벤션 주의**: 이 코드에서 `row`는 message를 **받는** node이다 (aggregation에서 `row` 기준으로 scatter). PyG의 컨벤션과 동일하다.

---

## 5. GCL (L50-105) -- Standard Graph Convolutional Layer

논문 Eq. 2 구현: `mij = phi_e(hi, hj, aij)`, `mi = sum(mij)`, `h_i^{l+1} = phi_h(hi, mi)`

### 5.1 `__init__` (L59-84)

```python
def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0,
             act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
    super(GCL, self).__init__()
    self.attention = attention
    self.t_eq = t_eq              # time equivariance flag (미사용)
    self.recurrent = recurrent    # residual connection 여부
    input_edge_nf = input_nf * 2  # source + target concatenation
```

**edge_mlp** (phi_e):
```python
    # Input: [h_i || h_j || edge_attr] -> hidden -> hidden
    self.edge_mlp = nn.Sequential(
        nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),  # (2*input_nf + edges_in_nf) -> hidden_nf
        act_fn,
        nn.Linear(hidden_nf, hidden_nf, bias=bias),                    # hidden_nf -> hidden_nf
        act_fn)
```

**att_mlp** (optional attention):
```python
    if self.attention:
        self.att_mlp = nn.Sequential(
            nn.Linear(input_nf, hidden_nf, bias=bias),   # |h_i - h_j| -> hidden
            act_fn,
            nn.Linear(hidden_nf, 1, bias=bias),          # hidden -> scalar
            nn.Sigmoid())                                  # [0, 1] attention weight
```
**주의**: attention 입력이 `|source - target|`이다 (L93). 이는 node 간 차이의 절대값으로 유사도를 측정하는 방식이다.

**node_mlp** (phi_h):
```python
    self.node_mlp = nn.Sequential(
        nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),  # [h_i || agg] -> hidden
        act_fn,
        nn.Linear(hidden_nf, output_nf, bias=bias))             # hidden -> output
```

### 5.2 `edge_model` (L87-95) -- 논문 Eq. 2: phi_e(hi, hj, aij)

```python
def edge_model(self, source, target, edge_attr):
    edge_in = torch.cat([source, target], dim=1)      # [h_i || h_j]: (E, 2*input_nf)
    if edge_attr is not None:
        edge_in = torch.cat([edge_in, edge_attr], dim=1)  # [h_i || h_j || a_ij]: (E, 2*input_nf + edges_in_nf)
    out = self.edge_mlp(edge_in)                       # (E, hidden_nf)
    if self.attention:
        att = self.att_mlp(torch.abs(source - target)) # (E, 1) -- soft attention
        out = out * att                                # element-wise scaling
    return out
```

**Tensor Shape**: `(E, input_nf)` x 2 -> concat -> `(E, 2*input_nf [+ edges_in_nf])` -> MLP -> `(E, hidden_nf)`

### 5.3 `node_model` (L97-105) -- 논문 Eq. 2: mi = sum(mij), hi+1 = phi_h(hi, mi)

```python
def node_model(self, h, edge_index, edge_attr):
    row, col = edge_index
    agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))  # (N, hidden_nf) -- Eq. 5
    out = torch.cat([h, agg], dim=1)       # [h_i || m_i]: (N, input_nf + hidden_nf)
    out = self.node_mlp(out)               # (N, output_nf) -- Eq. 6
    if self.recurrent:
        out = out + h                      # residual connection (requires output_nf == input_nf)
    return out
```

**`recurrent` 이름의 의미**: GRU 등의 recurrent 구조를 쓰려 했으나(L83-84 주석 참고), 단순 residual connection으로 대체되었다. 이름만 남아있다.

---

## 6. GCL_rf (L108-142) -- Radial Field Variant

논문 Table 1에서 비교 대상으로 사용되는 Radial Field 방식. 좌표 공간에서 직접 동작하며, node feature 없이 좌표만으로 메시지를 구성한다.

### 6.1 `__init__` (L117-126)

```python
def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
    super(GCL_rf, self).__init__()
    self.clamp = clamp
    layer = nn.Linear(nf, 1, bias=False)
    torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)   # *** 매우 작은 gain ***
    self.phi = nn.Sequential(
        nn.Linear(edge_attr_nf + 1, nf),   # radial distance + edge_attr -> hidden
        act_fn,
        layer)                              # hidden -> scalar (1)
    self.reg = reg   # regularization coefficient for position decay
```

**`gain=0.001`의 의미**: coord_mlp의 마지막 linear layer 가중치를 매우 작게 초기화한다. 이는 학습 초기에 좌표 이동량을 극도로 작게 만들어 훈련 안정성을 확보하기 위함이다. 좌표가 크게 변하면 거리 계산이 급변하여 gradient explosion이 발생할 수 있다. 이 패턴은 E_GCL(L176-177)에서도 동일하게 사용된다.

### 6.2 `edge_model` (L128-136)

```python
def edge_model(self, source, target, edge_attr):
    x_diff = source - target                                      # (E, D_coord) -- 좌표 차이 벡터
    radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)  # (E, 1) -- 유클리드 거리
    e_input = torch.cat([radial, edge_attr], dim=1)               # (E, 1 + edge_attr_nf)
    e_out = self.phi(e_input)                                     # (E, 1) -- scalar weight
    m_ij = x_diff * e_out                                         # (E, D_coord) -- 방향 * 크기
    if self.clamp:
        m_ij = torch.clamp(m_ij, min=-100, max=100)
    return m_ij
```

**E_GCL과의 차이**: GCL_rf는 유클리드 거리(sqrt)를 쓰고, E_GCL은 제곱 거리를 쓴다. GCL_rf는 source/target이 좌표 자체이며 별도의 node embedding이 없다.

### 6.3 `node_model` (L138-142)

```python
def node_model(self, x, edge_index, edge_attr):
    row, col = edge_index
    agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))  # mean aggregation
    x_out = x + agg - x * self.reg   # position update + L2 regularization toward origin
    return x_out
```

**`-x * self.reg`**: position decay term. 좌표가 원점에서 무한히 발산하는 것을 방지하는 정규화 항이다.

---

## 7. E_GCL (L145-251) -- 핵심 EGNN Layer

**논문의 핵심 기여**. Eq. 3-6을 구현하며, node feature 업데이트와 좌표 업데이트를 동시에 수행한다.

### 7.1 `__init__` (L154-196)

```python
def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0,
             act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0,
             attention=False, clamp=False, norm_diff=False, tanh=False):
    super(E_GCL, self).__init__()
    input_edge = input_nf * 2         # h_i, h_j concatenation
    self.coords_weight = coords_weight # C in Eq. 4: coordinate update scaling factor
    self.recurrent = recurrent         # residual connection
    self.attention = attention         # edge attention
    self.norm_diff = norm_diff         # normalize coordinate differences
    self.tanh = tanh                   # tanh activation on coord_mlp output
    edge_coords_nf = 1                # ||x_i - x_j||^2 is 1-dimensional
```

**edge_mlp** (phi_e, 논문 Eq. 3):
```python
    # Input dimension: 2*input_nf + 1 (radial) + edges_in_d (optional edge attributes)
    self.edge_mlp = nn.Sequential(
        nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),  # -> hidden
        act_fn,
        nn.Linear(hidden_nf, hidden_nf),                                  # -> hidden
        act_fn)
    # Output: (E, hidden_nf)
```

**node_mlp** (phi_h, 논문 Eq. 6):
```python
    self.node_mlp = nn.Sequential(
        nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),  # [h_i || m_i || node_attr]
        act_fn,
        nn.Linear(hidden_nf, output_nf))                             # -> output
```

**coord_mlp** (phi_x, 논문 Eq. 4의 scalar weight):
```python
    layer = nn.Linear(hidden_nf, 1, bias=False)          # output is scalar
    torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)  # *** 안정적 초기화 ***

    self.clamp = clamp
    coord_mlp = []
    coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))    # hidden -> hidden
    coord_mlp.append(act_fn)
    coord_mlp.append(layer)                                # hidden -> 1
    if self.tanh:
        coord_mlp.append(nn.Tanh())                       # [-1, 1] 범위 제한
        self.coords_range = nn.Parameter(torch.ones(1)) * 3  # learnable scaling factor
    self.coord_mlp = nn.Sequential(*coord_mlp)
```

**`bias=False`인 이유**: coord_mlp의 마지막 layer에 bias가 없다. bias가 있으면 모든 edge에 동일한 상수가 더해져, aggregation 후 좌표가 한 방향으로 치우치게 된다. 이는 translation equivariance를 깨뜨릴 수 있다. (단, mean aggregation에서는 이론적으로 상쇄되지 않음)

**`gain=0.001` 재등장**: GCL_rf와 동일한 이유. 학습 초기 좌표 변동을 억제하여 안정적인 훈련을 보장한다.

**`tanh` 옵션**: 좌표 이동량을 [-1, 1]로 제한하고, `coords_range`(학습 가능)를 곱하여 최대 이동 범위를 제어한다. 안정성이 더 필요한 경우 활성화한다.

**att_mlp** (optional):
```python
    if self.attention:
        self.att_mlp = nn.Sequential(
            nn.Linear(hidden_nf, 1),
            nn.Sigmoid())    # [0, 1] soft gating
```
GCL과 달리 E_GCL의 attention은 edge_mlp **출력**에 적용된다 (GCL은 `|source - target|`를 입력으로 사용).

### 7.2 `coord2radial` (L231-240) -- 좌표 -> 거리 변환

```python
def coord2radial(self, edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]                         # (E, D_coord): x_i - x_j
    radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)          # (E, 1): ||x_i - x_j||^2

    if self.norm_diff:
        norm = torch.sqrt(radial) + 1     # +1 to avoid division by zero
        coord_diff = coord_diff / (norm)   # unit direction vector (approximately)

    return radial, coord_diff
```

**왜 제곱 거리(squared distance)를 쓰는가?**

1. **Equivariance 보존**: `||x_i - x_j||^2`는 rotation/reflection에 대해 invariant한 스칼라이다. sqrt를 취해도 invariant하지만, 제곱 거리를 그대로 쓰면 sqrt 연산을 피할 수 있다.
2. **미분 안정성**: `sqrt(x)`는 `x=0` 근처에서 기울기가 무한대로 발산한다 (`d/dx sqrt(x) = 1/(2*sqrt(x))`). 제곱 거리를 그대로 쓰면 이 문제를 피할 수 있다. 두 node가 같은 위치에 있을 때 특히 중요하다.
3. **정보 보존**: phi_e가 충분히 expressive한 MLP라면, `d^2`를 입력으로 받아도 `d`에 대한 모든 smooth function을 근사할 수 있다 (MLP의 universal approximation).

**`norm_diff` 옵션**: coord_diff를 정규화하여 방향 벡터만 남긴다. 이 경우 이동 방향은 유지하되 거리 정보를 분리한다. `+1`은 zero-division 방지이면서 동시에 매우 가까운 node 사이의 force를 약화시키는 효과가 있다.

### 7.3 `edge_model` (L199-208) -- 논문 Eq. 3: mij = phi_e(hi, hj, ||xi-xj||^2, aij)

```python
def edge_model(self, source, target, radial, edge_attr):
    if edge_attr is None:
        out = torch.cat([source, target, radial], dim=1)
        # (E, 2*input_nf + 1)
    else:
        out = torch.cat([source, target, radial, edge_attr], dim=1)
        # (E, 2*input_nf + 1 + edges_in_d)
    out = self.edge_mlp(out)       # (E, hidden_nf) -- 논문의 phi_e
    if self.attention:
        att_val = self.att_mlp(out) # (E, 1)
        out = out * att_val         # soft gating
    return out
```

**GCL의 edge_model과의 핵심 차이**: `radial` (||xi-xj||^2)이 입력에 추가된다. 이것이 좌표 정보를 node feature와 함께 edge message에 반영하는 방법이다.

### 7.4 `coord_model` (L222-228) -- 논문 Eq. 4: xi+1 = xi + C * sum((xi-xj) * phi_x(mij))

```python
def coord_model(self, coord, edge_index, coord_diff, edge_feat):
    row, col = edge_index
    trans = coord_diff * self.coord_mlp(edge_feat)
    # coord_diff: (E, D_coord) -- (x_i - x_j) direction vectors
    # self.coord_mlp(edge_feat): (E, 1) -- scalar weight phi_x(m_ij)
    # trans: (E, D_coord) -- broadcasting: each edge gets a weighted direction

    trans = torch.clamp(trans, min=-100, max=100)
    # Safety clamp: 거의 활성화되지 않지만, 혹시 모를 explosion 방지

    agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
    # (N, D_coord) -- mean aggregation over neighbors

    coord += agg * self.coords_weight
    # coord += C * mean_j((x_i - x_j) * phi_x(m_ij))
    return coord
```

**논문 Eq. 4와의 대응** (line-by-line):
| Code (Line) | Paper Equation |
|---|---|
| L233 `coord_diff = coord[row] - coord[col]` | `(x_i - x_j)` in Eq. 4 |
| L224 `self.coord_mlp(edge_feat)` | `phi_x(m_ij)` in Eq. 4 |
| L224 `coord_diff * self.coord_mlp(edge_feat)` | `(x_i - x_j) * phi_x(m_ij)` in Eq. 4 |
| L226 `unsorted_segment_mean(trans, row, ...)` | `sum_j(...)` in Eq. 4 (mean 사용, 아래 설명 참고) |
| L227 `coord += agg * self.coords_weight` | `x_i^{l+1} = x_i^l + C * ...` in Eq. 4 |

**논문은 sum, 코드는 mean**: 논문 Eq. 4에서는 `1/(M-1) * sum`으로 정규화하지만, 실제 구현에서는 `unsorted_segment_mean`을 사용하여 이웃 수로 나눈다. fully-connected graph에서는 `M-1`이 이웃 수와 같으므로 동일하다.

**`self.coords_weight` (C)**: 좌표 업데이트 스케일링 상수. 기본값 1.0이며, 하이퍼파라미터로 조절 가능하다. 좌표 변화량의 전체 크기를 제어한다.

**Equivariance 보장 메커니즘**:
- `coord_diff = x_i - x_j`는 translation equivariant (평행이동해도 차이는 동일)
- `self.coord_mlp(edge_feat)`는 scalar (1차원) -> rotation에 무관
- scalar * vector = equivariant vector
- 이들의 합/평균도 equivariant
- 따라서 `coord += equivariant_vector`는 equivariance를 보존

### 7.5 `node_model` (L210-220) -- 논문 Eq. 5-6: mi = sum(mij), hi+1 = phi_h(hi, mi)

```python
def node_model(self, x, edge_index, edge_attr, node_attr):
    row, col = edge_index
    agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
    # (N, hidden_nf) -- Eq. 5: m_i = sum_j(m_ij)

    if node_attr is not None:
        agg = torch.cat([x, agg, node_attr], dim=1)   # (N, input_nf + hidden_nf + nodes_att_dim)
    else:
        agg = torch.cat([x, agg], dim=1)              # (N, input_nf + hidden_nf)

    out = self.node_mlp(agg)    # (N, output_nf) -- Eq. 6: phi_h(h_i, m_i)

    if self.recurrent:
        out = x + out           # residual connection (h_i + phi_h(...))
    return out, agg
```

**`agg` 반환**: node_model은 `(out, agg)`를 반환하지만, `agg`는 forward에서 사용되지 않는다. 디버깅/분석 용도로 보인다.

**node_attr**: 시간 불변 node 속성 (예: 원자 번호, 전하량 등). edge_attr와 별개로 node 업데이트 시 추가 정보를 제공한다.

### 7.6 `forward` (L242-251) -- 전체 순전파

```python
def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
    row, col = edge_index                                          # (2, E)
    radial, coord_diff = self.coord2radial(edge_index, coord)      # (E,1), (E, D_coord)

    edge_feat = self.edge_model(h[row], h[col], radial, edge_attr) # Eq. 3: (E, hidden_nf)
    coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)  # Eq. 4: (N, D_coord)
    h, agg = self.node_model(h, edge_index, edge_feat, node_attr)  # Eq. 5-6: (N, output_nf)

    return h, coord, edge_attr
```

**실행 순서가 중요하다**:
1. `coord2radial`: 현재 좌표로 거리 계산
2. `edge_model`: 거리 + node feature로 edge message 생성 (Eq. 3)
3. `coord_model`: edge message로 좌표 업데이트 (Eq. 4) -- **좌표가 먼저 업데이트됨**
4. `node_model`: edge message로 node feature 업데이트 (Eq. 5-6)

좌표 업데이트가 node feature 업데이트보다 먼저 일어나지만, 둘 다 같은 `edge_feat`를 사용하므로 순서가 결과에 영향을 주지 않는다. node_model의 입력은 업데이트 전의 `h`이다.

**`edge_attr` 반환**: edge_attr는 이 layer에서 변경되지 않고 그대로 반환된다. 다음 layer에서 동일한 edge attribute를 재사용할 수 있도록 한다.

### 7.7 전체 Tensor Shape 추적 (E_GCL forward pass)

```
입력:
  h:          (N, input_nf)      -- node features
  edge_index: (2, E)             -- edge connectivity
  coord:      (N, D_coord)       -- node coordinates (e.g., D_coord=3 for 3D)
  edge_attr:  (E, edges_in_d)    -- optional edge attributes

coord2radial:
  coord[row]:   (E, D_coord)
  coord[col]:   (E, D_coord)
  coord_diff:   (E, D_coord)     -- x_i - x_j
  radial:       (E, 1)           -- ||x_i - x_j||^2

edge_model:
  input:        (E, 2*input_nf + 1 + edges_in_d)  -- [h_i || h_j || radial || edge_attr]
  edge_feat:    (E, hidden_nf)

coord_model:
  coord_mlp(edge_feat):  (E, 1)              -- phi_x output (scalar per edge)
  trans:                 (E, D_coord)         -- (x_i-x_j) * phi_x(m_ij)
  agg (mean):            (N, D_coord)         -- aggregated translation
  coord (updated):       (N, D_coord)         -- x + C * agg

node_model:
  agg (sum):    (N, hidden_nf)               -- sum of edge messages
  cat:          (N, input_nf + hidden_nf [+ nodes_att_dim])
  h (updated):  (N, output_nf)

출력:
  h:          (N, output_nf)
  coord:      (N, D_coord)
  edge_attr:  (E, edges_in_d)     -- unchanged
```

---

## 8. E_GCL_vel (L254-284) -- Velocity Extension

논문 Eq. 7 구현: `v_i^{l+1} = phi_v(h_i) * v_i + C * sum((x_i - x_j) * phi_x(m_ij))`, `x_i^{l+1} = x_i + v_i^{l+1}`

E_GCL을 상속하며, velocity 항만 추가한다.

### 8.1 `__init__` (L264-270)

```python
def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0,
             act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0,
             attention=False, norm_diff=False, tanh=False):
    E_GCL.__init__(self, ...)     # 부모 클래스 초기화 (모든 E_GCL 구성요소 생성)
    self.norm_diff = norm_diff
    self.coord_mlp_vel = nn.Sequential(
        nn.Linear(input_nf, hidden_nf),    # h_i -> hidden
        act_fn,
        nn.Linear(hidden_nf, 1))           # hidden -> scalar (phi_v)
```

**`coord_mlp_vel`이 phi_v**: node feature `h_i`를 입력받아 scalar를 출력한다. 이 scalar가 기존 velocity에 곱해져 velocity의 크기를 조절한다.

**`gain=0.001` 미적용**: coord_mlp_vel에는 작은 gain 초기화가 없다. 이는 velocity 항이 이미 물리적으로 의미있는 크기를 가지고 있으므로, 과도하게 억제할 필요가 없기 때문으로 보인다.

### 8.2 `forward` (L272-284) -- Eq. 7 구현

```python
def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None):
    row, col = edge_index
    radial, coord_diff = self.coord2radial(edge_index, coord)

    edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)  # Eq. 3
    coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
    # 여기까지: coord = x_i + C * mean_j((x_i-x_j) * phi_x(m_ij))
    # 즉 E_GCL의 좌표 업데이트가 먼저 적용됨

    coord += self.coord_mlp_vel(h) * vel
    # coord_mlp_vel(h): (N, 1) -- phi_v(h_i), scalar per node
    # vel: (N, D_coord) -- velocity vector
    # broadcasting: (N, 1) * (N, D_coord) = (N, D_coord)
    # 최종: x_i^{l+1} = x_i + C*mean((x_i-x_j)*phi_x(m_ij)) + phi_v(h_i)*v_i

    h, agg = self.node_model(h, edge_index, edge_feat, node_attr)  # Eq. 5-6
    return h, coord, edge_attr
```

**논문 Eq. 7과의 대응**:
| Code | Paper Eq. 7 |
|---|---|
| `self.coord_mlp_vel(h)` | `phi_v(h_i^l)` -- node feature 기반 velocity scaling |
| `self.coord_mlp_vel(h) * vel` | `phi_v(h_i^l) * v_i` -- scaled velocity |
| coord_model 결과 + velocity 항 | `v_i^{l+1} = phi_v(h_i)*v_i + C*sum(...)` |
| `coord += ...` | `x_i^{l+1} = x_i + v_i^{l+1}` |

**주의: velocity는 업데이트되지 않는다**: 코드에서 `vel`은 수정되지 않고 입력값 그대로 사용된다. 즉, 각 layer에서 초기 velocity를 scaling하여 좌표에 더하는 구조이다. Velocity 자체의 업데이트는 이 layer 밖에서 관리된다.

---

## 9. GCL_rf_vel (L289-332) -- Radial Field + Velocity

GCL_rf의 velocity 확장 버전. E_GCL_vel과 달리 node embedding 없이 좌표 + velocity만으로 동작한다.

### 9.1 `__init__` (L297-311)

```python
def __init__(self, nf=64, edge_attr_nf=0, act_fn=nn.LeakyReLU(0.2), coords_weight=1.0):
    super(GCL_rf_vel, self).__init__()
    self.coords_weight = coords_weight
    self.coord_mlp_vel = nn.Sequential(
        nn.Linear(1, nf),       # vel_norm (scalar) -> hidden
        act_fn,
        nn.Linear(nf, 1))      # hidden -> scalar

    layer = nn.Linear(nf, 1, bias=False)
    torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
    self.phi = nn.Sequential(
        nn.Linear(1 + edge_attr_nf, nf),
        act_fn,
        layer,
        nn.Tanh())    # *** Tanh 필수 *** -- 주석에 "we had to add the tanh to keep this method stable"
```

**Tanh 안정성**: GCL_rf_vel은 node feature 없이 좌표만 사용하므로 representation power가 제한적이다. Tanh로 출력을 [-1, 1]로 제한하여 coordinate explosion을 방지한다. E_GCL에서는 Tanh가 옵션이지만, 여기서는 항상 적용된다.

**`coord_mlp_vel` 입력이 `vel_norm`**: E_GCL_vel에서는 `h` (node feature)를 입력으로 받지만, GCL_rf_vel에서는 `vel_norm` (velocity의 크기, scalar)을 입력으로 받는다. Node feature가 없으므로 velocity norm이 유일한 node-level 정보이다.

### 9.2 `forward` (L313-318)

```python
def forward(self, x, vel_norm, vel, edge_index, edge_attr=None):
    row, col = edge_index
    edge_m = self.edge_model(x[row], x[col], edge_attr)    # (E, D_coord)
    x = self.node_model(x, edge_index, edge_m)              # x + mean(edge_m) * C
    x += vel * self.coord_mlp_vel(vel_norm)                  # + vel * f(vel_norm)
    return x, edge_attr
```

---

## 10. 클래스 비교 분석

### 10.1 GCL vs E_GCL

| 측면 | GCL | E_GCL |
|---|---|---|
| **입력** | `h, edge_index` | `h, edge_index, coord` |
| **출력** | `h, edge_feat` | `h, coord, edge_attr` |
| **좌표 인식** | 없음 | 있음 (radial distance) |
| **좌표 업데이트** | 없음 | coord_model (Eq. 4) |
| **Equivariance** | 해당 없음 | E(n) equivariant |
| **edge_model 입력** | `[h_i, h_j, a_ij]` | `[h_i, h_j, \|\|x_i-x_j\|\|^2, a_ij]` |
| **Aggregation (node)** | sum | sum |
| **Aggregation (coord)** | N/A | mean |
| **상속** | GCL_basic | nn.Module (직접) |
| **Base class** | GCL_basic | nn.Module |

**E_GCL이 GCL_basic을 상속하지 않는 이유**: E_GCL의 forward signature가 다르다 (`coord` 추가). edge_model도 `radial` 파라미터가 추가되므로, GCL_basic의 forward 로직을 재사용할 수 없다.

### 10.2 E_GCL vs E_GCL_vel

| 측면 | E_GCL | E_GCL_vel |
|---|---|---|
| **추가 입력** | -- | `vel` (velocity vector) |
| **좌표 업데이트** | `x + C*mean(trans)` | `x + C*mean(trans) + phi_v(h)*vel` |
| **추가 파라미터** | -- | `coord_mlp_vel` (phi_v) |
| **용도** | 정적 구조 (QM9 등) | 동적 시뮬레이션 (N-body 등) |
| **논문 equation** | Eq. 3-6 | Eq. 7 |

### 10.3 E_GCL vs GCL_rf

| 측면 | E_GCL | GCL_rf |
|---|---|---|
| **Node feature** | 있음 (h) | 없음 (좌표가 곧 상태) |
| **Edge message** | hidden_nf 차원 벡터 | D_coord 차원 벡터 (좌표 공간) |
| **거리 계산** | squared distance (안정적) | Euclidean distance (sqrt) |
| **Expressivity** | 높음 (node + coord) | 낮음 (coord만) |
| **Regularization** | clamp (safety) | reg term (position decay) |

---

## 11. 설계 결정 및 Insights 요약

### 11.1 `gain=0.001` (L122, L177)

coord_mlp / phi의 마지막 linear layer에 적용. Xavier 초기화의 gain을 0.001로 설정하면 가중치가 `O(0.001/sqrt(fan_in))`으로 매우 작아진다. 이는:
- 학습 초기에 좌표 이동량을 거의 0으로 만듦
- 좌표가 급변하면 radial distance가 급변 -> edge message 급변 -> gradient explosion
- 학습이 진행되면서 서서히 적절한 이동량을 학습

### 11.2 Squared distance vs Euclidean distance

E_GCL은 `||x_i-x_j||^2`를, GCL_rf는 `||x_i-x_j||`를 사용한다. Squared distance의 장점:
- sqrt 연산 불필요 (연산 비용 절감)
- `x_i = x_j`일 때 gradient 안정 (sqrt의 gradient는 0에서 발산)
- MLP가 충분히 expressive하면 정보 손실 없음

### 11.3 Mean vs Sum aggregation

- **Node feature (Eq. 5)**: `unsorted_segment_sum` -- 이웃이 많을수록 더 많은 정보 반영
- **Coordinate (Eq. 4)**: `unsorted_segment_mean` -- degree-independent한 이동량 보장
- 이 구분이 학습 안정성에 핵심적 역할을 한다

### 11.4 Residual connection (`recurrent=True`)

`out = x + out` (L103, L219). "Recurrent"라는 이름이지만 실제로는 단순 skip connection이다. GRU 기반 recurrent update가 주석으로 남아있어(L83-84, L195-196), 원래는 GRU를 사용하려 했으나 단순 residual이 더 효과적이었음을 시사한다.

### 11.5 Clamp (L225)

`torch.clamp(trans, min=-100, max=100)` -- 주석에 "This is never activated but just in case it exploded it may save the train"이라고 명시. 실전에서는 거의 활성화되지 않는 안전장치이다. `gain=0.001`과 mean aggregation이 이미 충분히 값을 작게 유지하기 때문이다.

### 11.6 `edge_attr` 불변 반환

E_GCL의 forward는 `edge_attr`를 수정하지 않고 그대로 반환한다. 이는 edge attribute가 그래프의 고정된 속성(결합 유형, 거리 임계값 등)이며, layer마다 변하지 않는다는 설계 결정을 반영한다.

---

## 12. 정리: 논문 Equation-to-Code 매핑 (E_GCL 기준)

| 논문 | 수식 | 코드 위치 | 함수/라인 |
|---|---|---|---|
| **Eq. 3** | `m_ij = phi_e(h_i, h_j, \|\|x_i-x_j\|\|^2, a_ij)` | `edge_model` | L199-208 |
| Eq. 3 내부 | `\|\|x_i - x_j\|\|^2` | `coord2radial` | L234: `torch.sum((coord_diff)**2, 1)` |
| Eq. 3 내부 | `phi_e(...)` | `edge_mlp` | L165-169 |
| **Eq. 4** | `x_i^{l+1} = x_i + C * sum((x_i-x_j) * phi_x(m_ij))` | `coord_model` | L222-228 |
| Eq. 4 내부 | `(x_i - x_j)` | `coord2radial` | L233: `coord[row] - coord[col]` |
| Eq. 4 내부 | `phi_x(m_ij)` | `coord_mlp` | L224: `self.coord_mlp(edge_feat)` |
| Eq. 4 내부 | `(x_i-x_j) * phi_x(m_ij)` | -- | L224: `coord_diff * self.coord_mlp(edge_feat)` |
| Eq. 4 내부 | `C * sum(...)` | -- | L226-227: `unsorted_segment_mean * coords_weight` |
| **Eq. 5** | `m_i = sum_j(m_ij)` | `node_model` | L212: `unsorted_segment_sum(edge_attr, row, ...)` |
| **Eq. 6** | `h_i^{l+1} = phi_h(h_i, m_i)` | `node_model` | L213-217: cat + `node_mlp` |
| **Eq. 7** | `v_i^{l+1} = phi_v(h_i)*v + C*sum(...)` | `E_GCL_vel.forward` | L277 (coord_model) + L280 (`coord_mlp_vel(h)*vel`) |
| Eq. 7 내부 | `phi_v(h_i)` | `coord_mlp_vel` | L267-270 |
