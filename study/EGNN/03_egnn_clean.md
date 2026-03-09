# 03. egnn_clean.py 코드 분석

> 파일: `egnn/models/egnn_clean/egnn_clean.py`
> 논문: "E(n) Equivariant Graph Neural Networks" (arXiv:2102.09844)

---

## 1. 개요 및 목적

`egnn_clean.py`는 EGNN의 **독립형(standalone) 클린 구현**이다. `gcl.py`가 여러 실험용 변형(GCL, GCL_basic, GCL_rf, E_GCL_vel, GCL_rf_vel)을 모두 포함하는 반면, 이 파일은 핵심 EGNN 기능만 깔끔하게 분리한다.

파일 구성:
- **E_GCL** (L5-103): 단일 EGCL 레이어 -- 논문의 Eq. 3-6 구현
- **EGNN** (L106-146): E_GCL을 여러 층 쌓아 완전한 모델을 구성하는 wrapper
- **unsorted_segment_sum/mean** (L149-164): scatter 기반 집계 유틸리티
- **get_edges / get_edges_batch** (L167-191): fully connected 그래프의 edge 생성
- **\_\_main\_\_** (L194-211): 사용 예시

외부 의존성이 `torch`뿐이므로, 새 프로젝트에 EGNN을 가져다 쓸 때 이 파일 하나만 복사하면 된다.

---

## 2. gcl.py의 E_GCL과의 차이점

| 항목 | egnn_clean.py | gcl.py |
|------|---------------|--------|
| **활성화 함수** | `nn.SiLU()` (default) | `nn.ReLU()` (default) |
| **Residual 파라미터명** | `residual` | `recurrent` (실제 동작은 동일: `out = x + out`) |
| **좌표 정규화** | `normalize` 옵션 (`coord_diff / (norm + epsilon)`, `detach()` 적용) | `norm_diff` 옵션 (`coord_diff / (norm + 1)`, detach 없음) |
| **좌표 집계** | `coords_agg` 옵션 (`'mean'` 또는 `'sum'` 선택) | 항상 `unsorted_segment_mean` 고정 |
| **좌표 가중치** | 없음 (1.0 고정) | `coords_weight` 파라미터로 스케일링 |
| **Clamp** | 없음 (제거됨) | `torch.clamp(trans, min=-100, max=100)` 안전장치 |
| **tanh + coords_range** | `tanh` 옵션만 있음 | `tanh` 사용 시 `coords_range` 학습 파라미터도 추가 |
| **nodes_att_dim** | 없음 (제거됨) | node_mlp 입력에 추가 차원 가능 |
| **상속 구조** | 독립 `nn.Module` | 독립 `nn.Module` (GCL_basic을 상속하지 않음) |
| **Epsilon** | `self.epsilon = 1e-8` | 없음 (대신 `+ 1`로 zero division 회피) |
| **GRU 흔적** | 없음 (완전 제거) | 주석 처리된 GRU 코드 잔존 |

핵심 차이 요약:
- **SiLU vs ReLU**: SiLU (Swish)는 smooth non-linearity로 gradient flow가 더 안정적
- **normalize의 detach()**: clean 버전에서는 norm 계산을 computation graph에서 분리하여, 정규화 자체는 gradient를 받지 않음
- **coords_agg 선택**: 논문 Eq. 4의 C = 1/(M-1) 정규화를 `'mean'`으로 구현 가능

---

## 3. E_GCL 클래스: 수식-코드 매핑

### 3.1 `__init__` (L11-47): 네트워크 구성

```python
def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0,
             act_fn=nn.SiLU(), residual=True, attention=False,
             normalize=False, coords_agg='mean', tanh=False):
```

**edge_mlp** -- 논문의 phi_e (Eq. 3):
```python
# Input: [h_i, h_j, ||x_i-x_j||^2, a_ij]
# Dimensions: input_nf*2 + 1 + edges_in_d -> hidden_nf
self.edge_mlp = nn.Sequential(
    nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),  # L23
    act_fn,                                                            # L24
    nn.Linear(hidden_nf, hidden_nf),                                   # L25
    act_fn)                                                            # L26
```

**node_mlp** -- 논문의 phi_h (Eq. 6):
```python
# Input: [h_i, mi]
# Dimensions: hidden_nf + input_nf -> output_nf
self.node_mlp = nn.Sequential(
    nn.Linear(hidden_nf + input_nf, hidden_nf),  # L29
    act_fn,                                        # L30
    nn.Linear(hidden_nf, output_nf))               # L31
```

**coord_mlp** -- 논문의 phi_x (Eq. 4):
```python
# Input: mij (edge feature)
# Dimensions: hidden_nf -> 1 (scalar output)
layer = nn.Linear(hidden_nf, 1, bias=False)                    # L33
torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)        # L34 -- small init for stability
coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]   # L37-39
if self.tanh:
    coord_mlp.append(nn.Tanh())                                 # L41 -- bounds output to [-1, 1]
```

`gain=0.001`의 의미: 좌표 업데이트를 처음에 거의 0에 가깝게 초기화한다. 학습 초기에 좌표가 급변하면 학습이 불안정해지므로, 작은 값에서 시작하여 점진적으로 학습하도록 유도한다.

**att_mlp** -- 논문의 attention mechanism (optional):
```python
# Input: edge feature (mij)
# Output: scalar attention weight in [0, 1]
self.att_mlp = nn.Sequential(
    nn.Linear(hidden_nf, 1),  # L46
    nn.Sigmoid())              # L47
```

### 3.2 `coord2radial` (L84-93): 좌표 -> 거리/방향 변환

```python
def coord2radial(self, edge_index, coord):
    row, col = edge_index
    coord_diff = coord[row] - coord[col]                       # (x_i - x_j), shape: [E, n_dim]
    radial = torch.sum(coord_diff**2, 1).unsqueeze(1)          # ||x_i - x_j||^2, shape: [E, 1]

    if self.normalize:
        norm = torch.sqrt(radial).detach() + self.epsilon       # ||x_i - x_j|| + eps, detached
        coord_diff = coord_diff / norm                          # unit direction vector

    return radial, coord_diff
```

이 함수는 두 가지를 계산한다:
1. `radial`: 제곱 거리 (E(n) invariant scalar) -- edge_model의 입력으로 사용
2. `coord_diff`: 방향 벡터 -- coord_model에서 좌표 업데이트 방향으로 사용

`normalize` 옵션이 켜지면, coord_diff를 단위 벡터로 정규화한다. 이 경우 Eq. 4가 다음과 같이 변한다:

```
# 기본: x_i += Sigma (x_i - x_j) * phi_x(mij)
# normalize: x_i += Sigma (x_i - x_j) / ||x_i - x_j|| * phi_x(mij)
```

`detach()`가 중요한 이유: norm 자체는 gradient를 받지 않아야 정규화가 gradient의 스케일을 왜곡하지 않는다. 학습 안정성을 위한 설계이다.

### 3.3 `edge_model` (L49-58): Eq. 3 구현

```
Eq. 3: mij = phi_e(h_i^l, h_j^l, ||x_i^l - x_j^l||^2, a_ij)
```

```python
def edge_model(self, source, target, radial, edge_attr):
    # source = h[row] = h_i, shape: [E, input_nf]
    # target = h[col] = h_j, shape: [E, input_nf]
    # radial = ||x_i - x_j||^2, shape: [E, 1]
    # edge_attr = a_ij, shape: [E, edges_in_d] or None

    if edge_attr is None:
        out = torch.cat([source, target, radial], dim=1)        # [E, input_nf*2 + 1]
    else:
        out = torch.cat([source, target, radial, edge_attr], dim=1)  # [E, input_nf*2 + 1 + edges_in_d]

    out = self.edge_mlp(out)                                     # [E, hidden_nf] -- this is mij

    if self.attention:
        att_val = self.att_mlp(out)                              # [E, 1] -- sigmoid attention
        out = out * att_val                                      # element-wise scaling

    return out  # mij, shape: [E, hidden_nf]
```

논문과 정확히 일치한다. `phi_e`가 MLP이고, 입력이 `[h_i, h_j, ||x_i-x_j||^2, a_ij]`의 concatenation이다.

attention 옵션은 논문 Section 3.3의 edge inference (Eq. 8)에 해당한다: `eij * mij` 형태로, edge feature에 soft gating을 적용한다.

### 3.4 `coord_model` (L72-82): Eq. 4 구현

```
Eq. 4: x_i^{l+1} = x_i^l + C * Sigma_{j!=i} (x_i^l - x_j^l) * phi_x(mij)
```

```python
def coord_model(self, coord, edge_index, coord_diff, edge_feat):
    row, col = edge_index
    trans = coord_diff * self.coord_mlp(edge_feat)
    # coord_diff: [E, n_dim]  -- direction (x_i - x_j)
    # self.coord_mlp(edge_feat): [E, 1]  -- phi_x(mij), scalar weight
    # trans: [E, n_dim]  -- weighted direction vector per edge

    if self.coords_agg == 'sum':
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))    # [N, n_dim]
    elif self.coords_agg == 'mean':
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))   # [N, n_dim]

    coord = coord + agg                                          # x_i^{l+1} = x_i^l + agg
    return coord  # [N, n_dim]
```

핵심 연산은 `coord_diff * self.coord_mlp(edge_feat)`이다:
- `coord_diff`는 `[E, n_dim]` (예: 3D라면 `[E, 3]`)
- `self.coord_mlp(edge_feat)`는 `[E, 1]` (스칼라)
- broadcasting으로 곱해져서 각 방향 벡터에 스칼라 가중치가 곱해진다

이것이 equivariance의 핵심이다. phi_x가 스칼라를 출력하므로 회전 행렬 Q를 밖으로 뺄 수 있다:
```
Q(x_i - x_j) * phi_x(mij) = Q * [(x_i - x_j) * phi_x(mij)]
```

`coords_agg` 선택:
- `'mean'`: 논문 Eq. 4의 C = 1/(M-1) 정규화에 해당. 이웃 수에 관계없이 일정한 스케일 유지.
- `'sum'`: 이웃이 많을수록 업데이트가 커짐. 특정 태스크에서 유리할 수 있음.

### 3.5 `node_model` (L60-70): Eq. 5 + Eq. 6 구현

```
Eq. 5: mi = Sigma_{j!=i} mij           (aggregation)
Eq. 6: h_i^{l+1} = phi_h(h_i^l, mi)   (node update)
```

```python
def node_model(self, x, edge_index, edge_attr, node_attr):
    row, col = edge_index

    # --- Eq. 5: aggregation ---
    agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
    # edge_attr (= mij): [E, hidden_nf]
    # agg (= mi): [N, hidden_nf]  -- sum of messages for each node

    # --- Eq. 6: node update ---
    if node_attr is not None:
        agg = torch.cat([x, agg, node_attr], dim=1)             # [N, input_nf + hidden_nf + node_attr_dim]
    else:
        agg = torch.cat([x, agg], dim=1)                        # [N, input_nf + hidden_nf]

    out = self.node_mlp(agg)                                     # [N, output_nf]  -- phi_h([h_i, mi])

    if self.residual:
        out = x + out                                            # residual connection: h + phi_h(h, m)

    return out, agg  # out: [N, output_nf], agg: [N, input_nf + hidden_nf]
```

주의: `node_attr`은 이 파일에서 실제로 사용되진 않지만(EGNN wrapper가 None으로 전달), 확장성을 위해 인터페이스를 유지한다. `node_attr`이 있으면 node_mlp의 입력 차원이 달라지므로 `__init__`에서 해당 차원을 맞춰야 한다.

Aggregation에서 `unsorted_segment_sum`을 사용하는 것은 논문 Eq. 5와 동일하다 (mean이 아닌 sum).

### 3.6 `forward` (L95-103): 전체 흐름

```python
def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
    row, col = edge_index

    # Step 1: Compute pairwise distances and direction vectors
    radial, coord_diff = self.coord2radial(edge_index, coord)
    # radial: [E, 1], coord_diff: [E, n_dim]

    # Step 2: Eq. 3 -- Edge model (message generation)
    edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
    # edge_feat (mij): [E, hidden_nf]

    # Step 3: Eq. 4 -- Coordinate update
    coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
    # coord: [N, n_dim]

    # Step 4: Eq. 5+6 -- Message aggregation + node update
    h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
    # h: [N, output_nf]

    return h, coord, edge_attr
```

**실행 순서가 중요하다**: coord_model이 node_model보다 먼저 실행된다. 즉, 좌표 업데이트는 현재 레이어의 edge feature(mij)를 사용하되, 노드 임베딩 업데이트 전에 수행된다. 이는 논문의 수식 순서(Eq. 3 -> Eq. 4 -> Eq. 5 -> Eq. 6)와 일치한다.

`edge_attr`은 변환 없이 그대로 반환된다 -- edge attribute는 레이어를 거쳐도 변하지 않는다.

---

## 4. EGNN 클래스: 전체 모델 (L106-146)

### 4.1 `__init__` (L107-139)

```python
class EGNN(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, in_edge_nf=0,
                 device='cpu', act_fn=nn.SiLU(), n_layers=4, residual=True,
                 attention=False, normalize=False, tanh=False):
```

구조:
```
embedding_in:  Linear(in_node_nf -> hidden_nf)
  |
gcl_0:  E_GCL(hidden_nf -> hidden_nf)
gcl_1:  E_GCL(hidden_nf -> hidden_nf)
  ...
gcl_{n-1}: E_GCL(hidden_nf -> hidden_nf)
  |
embedding_out: Linear(hidden_nf -> out_node_nf)
```

```python
self.embedding_in = nn.Linear(in_node_nf, self.hidden_nf)        # L133
self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)      # L134
for i in range(0, n_layers):
    self.add_module("gcl_%d" % i, E_GCL(                         # L136
        self.hidden_nf, self.hidden_nf, self.hidden_nf,          # all hidden_nf
        edges_in_d=in_edge_nf, act_fn=act_fn, residual=residual,
        attention=attention, normalize=normalize, tanh=tanh))
```

모든 E_GCL 레이어의 input_nf = output_nf = hidden_nf이다. 이는 residual connection(`out = x + out`)이 동작하기 위한 필수 조건이다. 차원이 다르면 덧셈이 불가능하다.

`add_module`로 등록하므로 `model.parameters()`에 자동 포함된다. `self._modules["gcl_%d" % i]`로 접근한다.

### 4.2 `forward` (L141-146)

```python
def forward(self, h, x, edges, edge_attr):
    # h: [N_total, in_node_nf]  -- node features
    # x: [N_total, n_dim]       -- coordinates (e.g., 3D positions)
    # edges: [2, E]             -- edge index (row, col)
    # edge_attr: [E, in_edge_nf] -- edge attributes

    h = self.embedding_in(h)                          # [N_total, hidden_nf]

    for i in range(0, self.n_layers):
        h, x, _ = self._modules["gcl_%d" % i](h, edges, x, edge_attr=edge_attr)
        # h: [N_total, hidden_nf]
        # x: [N_total, n_dim]  -- updated coordinates
        # _: edge_attr (unchanged, discarded)

    h = self.embedding_out(h)                         # [N_total, out_node_nf]
    return h, x
```

forward의 흐름:
1. `embedding_in`: 입력 노드 피처를 hidden 차원으로 projection
2. n개의 E_GCL 레이어를 순차 실행 -- h와 x가 매 레이어마다 업데이트됨
3. `embedding_out`: 최종 h를 출력 차원으로 projection

좌표 x에는 별도의 embedding이 없다 -- x는 실제 물리적 좌표이므로 차원 변환 없이 그대로 사용하며, E_GCL 내부에서만 업데이트된다.

---

## 5. 유틸리티 함수

### 5.1 unsorted_segment_sum (L149-154)

```python
def unsorted_segment_sum(data, segment_ids, num_segments):
    # data: [E, D]           -- per-edge values
    # segment_ids: [E]       -- which node each edge belongs to (row indices)
    # num_segments: N         -- number of nodes

    result_shape = (num_segments, data.size(1))                     # (N, D)
    result = data.new_full(result_shape, 0)                         # [N, D] zeros
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))  # [E] -> [E, D]
    result.scatter_add_(0, segment_ids, data)                       # in-place scatter add
    return result  # [N, D]
```

`scatter_add_`는 `result[segment_ids[i]][j] += data[i][j]`를 수행한다. 즉 같은 노드로 향하는 모든 edge의 값을 합산한다.

TensorFlow의 `tf.math.unsorted_segment_sum`과 동일한 기능을 PyTorch로 구현한 것이다.

### 5.2 unsorted_segment_mean (L157-164)

```python
def unsorted_segment_mean(data, segment_ids, num_segments):
    # Same as sum, but divides by count
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)   # clamp to avoid division by zero
```

sum과 동일하되, 각 segment에 속한 원소의 개수로 나눈다. `count.clamp(min=1)`로 이웃이 없는 노드에서의 0 나눗셈을 방지한다.

### 5.3 get_edges / get_edges_batch (L167-191)

```python
def get_edges(n_nodes):
    # Generates fully connected graph edges (excluding self-loops)
    # For n_nodes=4: returns all (i,j) pairs where i != j
    # Number of edges: n_nodes * (n_nodes - 1)
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                rows.append(i)
                cols.append(j)
    return [rows, cols]
```

```python
def get_edges_batch(n_nodes, batch_size):
    edges = get_edges(n_nodes)                                    # base edges for single graph
    edge_attr = torch.ones(len(edges[0]) * batch_size, 1)        # [E_total, 1] all-ones edge attr

    edges = [torch.LongTensor(edges[0]), torch.LongTensor(edges[1])]
    if batch_size == 1:
        return edges, edge_attr

    # For batch_size > 1: offset node indices per graph
    rows, cols = [], []
    for i in range(batch_size):
        rows.append(edges[0] + n_nodes * i)                      # offset by n_nodes * batch_idx
        cols.append(edges[1] + n_nodes * i)
    edges = [torch.cat(rows), torch.cat(cols)]
    return edges, edge_attr
```

배치 처리 방식: PyG 스타일의 "하나의 큰 그래프"로 합친다. 예를 들어 batch_size=2, n_nodes=4이면:
- 그래프 0의 노드: 0, 1, 2, 3
- 그래프 1의 노드: 4, 5, 6, 7
- 각 그래프 내에서만 fully connected, 그래프 간에는 연결 없음

edge_attr는 모두 1.0인 더미 값으로, 실제 응용에서는 원자 간 결합 종류 등으로 대체한다.

---

## 6. Tensor Shape 추적 (예시)

`__main__` 블록의 예시를 따라 전체 shape 변화를 추적한다:

```
Hyperparameters:
  batch_size = 8, n_nodes = 4, n_feat = 1, x_dim = 3
  hidden_nf = 32, n_layers = 4 (default)
  N_total = batch_size * n_nodes = 32
  E = n_nodes * (n_nodes - 1) * batch_size = 4 * 3 * 8 = 96
```

```
Input:
  h:         [32, 1]     -- 1-dimensional node features
  x:         [32, 3]     -- 3D coordinates
  edges:     [2, 96]     -- edge index (row, col tensors each of length 96)
  edge_attr: [96, 1]     -- dummy edge attributes

embedding_in(h):  [32, 1] -> [32, 32]

--- E_GCL layer (repeated 4 times) ---

  coord2radial:
    coord_diff = coord[row] - coord[col]:     [96, 3]
    radial = sum(coord_diff^2, dim=1):        [96, 1]

  edge_model:
    cat([h_i, h_j, radial, edge_attr]):       [96, 32+32+1+1] = [96, 66]
    edge_mlp output:                          [96, 32]          -- mij

  coord_model:
    coord_mlp(edge_feat):                     [96, 1]           -- phi_x(mij)
    trans = coord_diff * phi_x:               [96, 3]
    agg (mean over edges per node):           [32, 3]
    coord = coord + agg:                      [32, 3]

  node_model:
    agg = segment_sum(mij, row):              [32, 32]          -- mi (Eq. 5)
    cat([h, agg]):                            [32, 32+32] = [32, 64]
    node_mlp output:                          [32, 32]
    residual: out = h + out:                  [32, 32]

--- End E_GCL ---

embedding_out(h): [32, 32] -> [32, 1]

Output:
  h: [32, 1]     -- updated node features
  x: [32, 3]     -- updated coordinates
```

---

## 7. egnn_clean.py vs gcl.py: 어느 것을 사용할 것인가

| 상황 | 선택 | 이유 |
|------|------|------|
| 새 프로젝트에 EGNN 도입 | **egnn_clean.py** | 단일 파일, 의존성 없음, 코드가 깔끔 |
| N-body velocity 실험 재현 | **gcl.py** | E_GCL_vel 클래스가 필요 (Eq. 7 구현) |
| 논문 결과 정확히 재현 | **gcl.py** | 원래 실험에 사용된 파일 (ReLU, coords_weight 등 하이퍼파라미터 일치) |
| Radial Field baseline 비교 | **gcl.py** | GCL_rf, GCL_rf_vel 포함 |
| 학습 안정성이 중요 | **egnn_clean.py** | SiLU, normalize, coords_agg 옵션으로 세밀한 제어 가능 |
| 커스텀 모델 구축 | **egnn_clean.py** | 불필요한 코드 없이 E_GCL만 깔끔하게 확장 가능 |

정리하면, `gcl.py`는 논문의 모든 실험을 재현하기 위한 "연구용 코드"이고, `egnn_clean.py`는 핵심만 추출한 "프로덕션/재사용 코드"이다. 후자가 개선된 default (SiLU, normalize 옵션)를 포함하고 있으므로, 새로 시작하는 프로젝트에서는 egnn_clean.py를 기반으로 하는 것이 권장된다.
