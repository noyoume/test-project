# 04. N-body Experiment Pipeline 분석

> EGNN 논문 (arXiv:2102.09844), Section 5.1 "N-body system" 실험의 전체 파이프라인을 분석한다.
> 관련 코드: `egnn/` 디렉토리 내 파일들.

---

## 1. 실험 개요 (Experiment Overview)

N-body 실험의 목표는 **5개의 charged particle로 구성된 시스템에서 미래 위치를 예측**하는 것이다.

- 입력: 특정 시점(frame_0)에서의 위치(`loc`), 속도(`vel`), 전하(`charges`), 입자 간 상호작용(`edges`)
- 출력: 미래 시점(frame_T)에서의 위치(`loc`)
- 손실 함수: MSE loss between predicted positions and ground truth positions
- 3D 공간에서의 시뮬레이션 (`dim=3`)

핵심 아이디어는 이 시스템이 **회전/병진 등변(equivariant)**하다는 것이다. 물리 법칙은 좌표계에 의존하지 않으므로, 모델도 이를 반영해야 한다. EGNN은 좌표를 직접 업데이트하면서 E(n) equivariance를 유지한다.

---

## 2. 데이터 생성 파이프라인 (Data Generation)

> 파일: `egnn/n_body_system/dataset/synthetic_sim.py`, `generate_dataset.py`

### 2.1 ChargedParticlesSim (주요 실험에 사용)

Coulomb-like force를 받는 charged particle 시스템을 시뮬레이션한다.

**전하(Charge) 초기화:**

```python
self._charge_types = np.array([-1., 0., 1.])
charges = np.random.choice(self._charge_types, size=(n_balls, 1), p=[0.5, 0, 0.5])
# charge_prob의 기본값이 [1/2, 0, 1/2]이므로 0 전하는 사용되지 않음.
# 각 입자는 +1 또는 -1 전하를 가진다.
```

**Edge (상호작용 행렬) 생성:**

```python
edges = charges.dot(charges.transpose())
# edges[i,j] = charges[i] * charges[j]
# 같은 부호: +1 (repulsive), 다른 부호: -1 (attractive)
```

이 `edges` 행렬은 입자 간 상호작용의 부호와 크기를 인코딩한다. 학습 시 모델에 edge attribute로 제공된다.

**Coulomb-like 힘 계산:**

```python
l2_dist_power3 = np.power(self._l2(loc_next.T, loc_next.T), 3./2.)
forces_size = interaction_strength * edges / l2_dist_power3
# F ~ q_i * q_j / |r|^3 (unnormalized direction vector와 곱해지므로 결과적으로 ~ 1/|r|^2)
```

힘의 크기는 `interaction_strength * edges[i,j] / |r_ij|^3`이다. 이후 방향 벡터 `(r_i - r_j)`와 곱해지므로, 실제 힘은 Coulomb의 법칙 `F ~ q_i*q_j * (r_i - r_j) / |r_ij|^3`을 따른다. 결과적으로 `|F| ~ 1/|r|^2`이 된다.

힘은 `_max_F = 0.1 / _delta_T = 100`으로 clamping되어 수치 안정성을 보장한다.

**Leapfrog integration:**

```python
self._delta_T = 0.001

# Half-step velocity update (initial)
vel_next += self._delta_T * F

# Main loop: leapfrog
for i in range(1, T):
    loc_next += self._delta_T * vel_next           # position update
    # ... force recalculation ...
    vel_next += self._delta_T * F                   # velocity update
```

Leapfrog integrator는 symplectic integrator로서 에너지를 장기적으로 보존하는 특성이 있다. 위치와 속도를 반 스텝씩 엇갈려 업데이트한다.

**샘플링:**

```python
T_save = int(T / sample_freq - 1)
# T=5000, sample_freq=100 (default for small) -> T_save=49 frames
# 매 sample_freq step마다 하나의 frame을 저장
```

### 2.2 SpringSim

스프링으로 연결된 입자 시스템이다. N-body 실험의 주된 대상은 charged particles이지만, 코드에 함께 구현되어 있다.

```python
self._spring_types = np.array([0., 0.5, 1.])
edges = np.random.choice(self._spring_types, size=(n_balls, n_balls), p=spring_prob)
# spring_prob 기본값: [1/2, 0, 1/2] -> 스프링 강도 0 또는 1
```

SpringSim에서의 힘: `F = -interaction_strength * edges[i,j] * (r_i - r_j)` (Hooke's law). ChargedParticlesSim과 달리 거리에 비례하는 복원력이다.

SpringSim에는 box boundary에서의 탄성 충돌(`_clamp`)이 있지만, 실제 시뮬레이션 루프에서는 주석 처리되어 있다.

### 2.3 generate_dataset.py: 데이터 저장 형식

```bash
# nbody (large dataset):
python generate_dataset.py --num-train 50000 --sample-freq 500

# nbody_small:
python generate_dataset.py --num-train 10000 --seed 43 --sufix small
```

생성되는 `.npy` 파일들:

| 파일 | Shape | 설명 |
|------|-------|------|
| `loc_{split}.npy` | `(num_sims, T_save, dim, n_balls)` | 위치 trajectory |
| `vel_{split}.npy` | `(num_sims, T_save, dim, n_balls)` | 속도 trajectory |
| `edges_{split}.npy` | `(num_sims, n_balls, n_balls)` | 상호작용 행렬 (charge product) |
| `charges_{split}.npy` | `(num_sims, n_balls, 1)` | 각 입자의 전하 |

`nbody_small` 기준: `T=5000`, `sample_freq=100` -> `T_save=49` frames.
`nbody` (large) 기준: `T=5000`, `sample_freq=500` -> `T_save=9` frames.

---

## 3. NBodyDataset: 로딩과 전처리

> 파일: `egnn/n_body_system/dataset_nbody.py`

### 3.1 데이터 로딩

```python
class NBodyDataset():
    def __init__(self, partition='train', max_samples=1e8, dataset_name="se3_transformer"):
        # dataset_name에 따라 suffix 결정
        # "nbody"       -> suffix = "{split}_charged5_initvel1"
        # "nbody_small" -> suffix = "{split}_charged5_initvel1small"
```

### 3.2 전처리 (preprocess)

```python
def preprocess(self, loc, vel, edges, charges):
    loc, vel = torch.Tensor(loc).transpose(2, 3)  # (N, T, dim, n_balls) -> (N, T, n_balls, dim)
    # max_samples로 데이터 수 제한
    loc = loc[0:self.max_samples, :, :, :]
```

Edge 처리 -- fully connected graph (self-loop 제외):

```python
rows, cols = [], []
for i in range(n_nodes):      # n_nodes = 5
    for j in range(n_nodes):
        if i != j:
            edge_attr.append(edges[:, i, j])  # charge product for each pair
            rows.append(i)
            cols.append(j)
# 결과: 5*4 = 20 edges (방향 포함)
# edge_attr shape: (batch_size, 20, 1) -- charge product 값
```

### 3.3 Frame 선택 (__getitem__)

```python
def __getitem__(self, i):
    if self.dataset_name == "nbody":
        frame_0, frame_T = 6, 8       # large dataset: 짧은 시간 간격
    elif self.dataset_name == "nbody_small":
        frame_0, frame_T = 30, 40     # small dataset: 긴 시간 간격
    elif self.dataset_name == "nbody_small_out_dist":
        frame_0, frame_T = 20, 30     # out-of-distribution 테스트

    return loc[frame_0], vel[frame_0], edge_attr, charges, loc[frame_T]
    # input: frame_0 시점의 (loc, vel, edge_attr, charges)
    # target: frame_T 시점의 loc
```

`nbody_small`의 경우 `sample_freq=100`, `_delta_T=0.001`이므로:
- frame 30은 시뮬레이션 시간 `(30+1)*100*0.001 = 3.1`초 시점
- frame 40은 시뮬레이션 시간 `(40+1)*100*0.001 = 4.1`초 시점
- 모델은 **1.0초 후의 위치를 예측**하는 셈이다 (10 sample intervals)

### 3.4 Batch용 Edge 생성 (get_edges)

```python
def get_edges(self, batch_size, n_nodes):
    # 배치 내 각 그래프의 노드 인덱스를 offset하여 연결
    for i in range(batch_size):
        rows.append(edges[0] + n_nodes * i)  # node index offset
        cols.append(edges[1] + n_nodes * i)
    edges = [torch.cat(rows), torch.cat(cols)]
```

배치 내의 여러 그래프를 하나의 큰 disconnected graph로 합치는 방식이다. PyG의 `Batch` 개념과 동일하다.

---

## 4. Model Wrappers (model.py)

> 파일: `egnn/n_body_system/model.py`
> GCL 구현: `egnn/models/gcl.py`

### 4.1 GNN (Standard Graph Neural Network)

```python
class GNN(nn.Module):
    # input_dim=6: [loc(3) ; vel(3)] concatenation
    # Embedding: Linear(6, hidden_nf)
    # n_layers x GCL(hidden_nf, hidden_nf, edges_in_nf=1)
    # Decoder: Linear(hidden_nf, hidden_nf) -> SiLU -> Linear(hidden_nf, 3)
```

- **Node feature**: 위치와 속도를 concat한 6차원 벡터
- **Edge attribute**: charge product (1차원)
- **출력**: decoder를 통해 3D 좌표 직접 예측
- **Recurrent**: residual connection (`out = x + out`)
- 좌표를 node feature에 직접 넣기 때문에 **equivariant하지 않다**

GCL의 edge_model: `[h_i; h_j; edge_attr] -> MLP -> m_ij`
GCL의 node_model: `[h_i; sum(m_ij)] -> MLP -> h_i'`

### 4.2 EGNN (E(n) Equivariant GNN - 기본 버전)

```python
class EGNN(nn.Module):
    # in_node_nf=1 (constant), in_edge_nf depends on preprocessing
    # Embedding: Linear(1, hidden_nf)
    # n_layers x E_GCL(hidden_nf, hidden_nf, edges_in_d=in_edge_nf)
```

- **Node feature**: 모든 노드에 상수 1 (위치 정보를 feature에 넣지 않음)
- **좌표(x)**: 별도의 좌표 채널로 처리 -> equivariance 보장
- **Edge attribute**: `[charge_product; loc_dist; vel_attr]` (3차원)
  - `loc_dist`: 입자 간 squared distance
  - `vel_attr`: `get_velocity_attr`로 계산한 projected velocity

E_GCL의 핵심 연산:

```python
# Edge model: m_ij = MLP([h_i; h_j; ||x_i - x_j||^2; edge_attr])
# Coord model: x_i += mean_j( (x_i - x_j) * MLP(m_ij) )  -- equivariant update
# Node model: h_i = h_i + MLP([h_i; sum_j(m_ij)])         -- invariant update
```

`coord_mlp`의 마지막 layer는 xavier init with `gain=0.001`로 초기화되어, 좌표 업데이트가 처음에 매우 작게 시작한다.

### 4.3 EGNN_vel (EGNN with Velocity - 주요 모델)

```python
class EGNN_vel(nn.Module):
    # in_node_nf=1: velocity norm (scalar)
    # in_edge_nf=2: [charge_product; loc_dist]
    # Uses E_GCL_vel layers
```

- **Node feature**: 속도의 L2 norm (`sqrt(sum(vel^2))`) -- scalar이므로 invariant
- **좌표(x)**: 별도 채널
- **속도(vel)**: 별도 입력으로 전달
- **Edge attribute**: `[charge_product; loc_dist]` (2차원) -- `get_velocity_attr` 사용하지 않음

E_GCL_vel의 핵심 차이점 (E_GCL 대비):

```python
# E_GCL과 동일한 edge_model, node_model, coord_model 수행 후:
coord += self.coord_mlp_vel(h) * vel
# coord_mlp_vel: Linear(hidden_nf, hidden_nf) -> act_fn -> Linear(hidden_nf, 1)
# h를 기반으로 scalar weight를 학습하여 velocity를 좌표 업데이트에 반영
```

이 velocity 항은 물리적으로 자연스럽다: 현재 속도 방향으로 얼마나 이동할지를 학습한다. `vel`은 3D 벡터이므로 이 업데이트도 equivariant하다 (scalar * vector).

### 4.4 RF_vel (Radial Field with Velocity)

```python
class RF_vel(nn.Module):
    # GCL_rf_vel layers 사용
    # hidden feature 없이 좌표 공간에서 직접 연산
```

- **입력**: `vel_norm` (scalar), `loc`, `vel`, `edge_attr`
- Radial Field 방식: edge message가 `(x_i - x_j) * phi(||x_i - x_j||, edge_attr)`
  - `phi`의 출력에 `Tanh`가 적용됨 (안정성을 위해)
- 좌표 업데이트: `x_i += mean_j(m_ij) + vel * MLP(vel_norm)`
- Hidden node feature `h`를 유지하지 않는 simpler baseline

### 4.5 Baseline, Linear, Linear_dynamics

```python
class Baseline(nn.Module):
    def forward(self, loc): return loc           # 현재 위치를 그대로 반환

class Linear(nn.Module):
    def forward(self, input): return self.linear(input)  # input=[loc;vel], Linear(6,3)

class Linear_dynamics(nn.Module):
    def forward(self, x, v): return x + v * self.time   # time은 학습 파라미터 (초기값 0.7)
```

- `Baseline`: 아무것도 하지 않음 -- "움직이지 않는다"는 가정의 하한선
- `Linear`: [loc; vel]을 입력으로 받는 단순 선형 모델
- `Linear_dynamics`: `x' = x + v*t` -- 등속 운동 가정, `t`만 학습

### 4.6 get_velocity_attr 함수

```python
def get_velocity_attr(loc, vel, rows, cols):
    diff = loc[cols] - loc[rows]                  # edge direction vector
    norm = torch.norm(diff, p=2, dim=1).unsqueeze(1)
    u = diff / norm                                # unit direction vector
    va = vel[rows] * u                             # velocity projected onto edge direction
    va = torch.sum(va, dim=1).unsqueeze(1)         # dot product -> scalar
    return va
```

이 함수는 source 노드의 속도를 edge 방향으로 projection한 **scalar** 값을 반환한다. 이는 rotation invariant한 양이다 (내적 결과). EGNN 모델(velocity 없는 버전)에서 edge attribute로 사용된다.

---

## 5. Training Pipeline (main_nbody.py)

### 5.1 데이터 전처리 (모델별)

`train()` 함수 내에서 모델 종류에 따라 입력 데이터를 다르게 구성한다.

**배치 데이터 reshape:**

```python
batch_size, n_nodes, _ = data[0].size()         # data[0] = loc: (B, 5, 3)
data = [d.view(-1, d.size(2)) for d in data]    # (B, 5, 3) -> (B*5, 3)
loc, vel, edge_attr, charges, loc_end = data
# 모든 배치의 노드를 하나로 합침 (B*5 nodes)
```

**GNN:**

```python
nodes = torch.cat([loc, vel], dim=1)   # (B*5, 6) -- loc과 vel concat
loc_pred = model(nodes, edges, edge_attr)
# edge_attr: charge product만 사용 (1차원)
```

**EGNN (velocity 없는 버전):**

```python
nodes = torch.ones(loc.size(0), 1)               # (B*5, 1) -- 상수 feature
loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # squared distance
vel_attr = get_velocity_attr(loc, vel, rows, cols).detach()        # projected velocity (scalar)
edge_attr = torch.cat([edge_attr, loc_dist, vel_attr], 1).detach()
# edge_attr: [charge_product; squared_distance; projected_velocity] (3차원)
loc_pred = model(nodes, loc.detach(), edges, edge_attr)
```

**EGNN_vel (주요 모델):**

```python
nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()  # velocity norm (B*5, 1)
loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)
edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
# edge_attr: [charge_product; squared_distance] (2차원)
loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
# vel을 별도 인자로 전달
```

**RF_vel:**

```python
vel_norm = torch.sqrt(torch.sum(vel ** 2, dim=1).unsqueeze(1)).detach()
loc_dist = torch.sum((loc[rows] - loc[cols]) ** 2, 1).unsqueeze(1)
edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()
loc_pred = model(vel_norm, loc.detach(), edges, vel, edge_attr)
```

주목할 점: EGNN 계열 모델에서 `loc.detach()`와 `.detach()`를 사용한다. 이는 좌표 자체를 통한 gradient flow를 차단하고, 좌표 업데이트 함수(`coord_mlp`)만을 통해 학습하도록 한다.

### 5.2 손실 함수와 학습

```python
loss_mse = nn.MSELoss()
loss = loss_mse(loc_pred, loc_end)     # predicted position vs ground truth position at frame_T
```

Optimizer: Adam with `lr=5e-4`, `weight_decay=1e-12`

Validation 기반 early stopping 방식:
```python
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_test_loss = test_loss    # validation best일 때의 test loss를 기록
    best_epoch = epoch
```

### 5.3 Sweep Experiment

`main_sweep()`는 training sample 수를 변화시키며 성능을 비교한다:

```python
training_samples = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25000, 50000]
n_epochs =         [2000, 2000, 4000, 5000, 8000, 10000, 8000, 6000, 4000, 2000]
```

데이터가 적을수록 더 많은 epoch을 돌리고, 데이터가 많을수록 적은 epoch으로 충분하다. `egnn_vel`은 별도의 (더 적은) epoch 스케줄을 가진다 -- 수렴이 빠르다는 것을 시사한다.

Test interval도 동적으로 조정된다:
```python
args.test_interval = max(int(10000 / tr_samples), 1)
# 데이터가 적으면 더 자주 validation 체크
```

---

## 6. Hyperparameters 정리

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `n_layers` | 4 | GCL/E_GCL 레이어 수 |
| `nf` (hidden_nf) | 64 | Hidden feature dimension |
| `lr` | 5e-4 | Learning rate (Adam) |
| `batch_size` | 100 | 배치 크기 |
| `epochs` | 10000 | 최대 epoch 수 |
| `weight_decay` | 1e-12 | L2 regularization (사실상 없음) |
| `max_training_samples` | 3000 | 기본 학습 샘플 수 |
| `test_interval` | 5 | Validation/test 수행 간격 (epoch) |
| `n_balls` | 5 | 입자 수 |
| `dataset` | nbody_small | 기본 데이터셋 |
| `recurrent` | True | EGNN_vel에서 residual connection 사용 |
| `norm_diff` | False | 좌표 차이 정규화 여부 |
| `tanh` | False | coord_mlp에 Tanh 적용 여부 |

EGNN_vel 모델 구체 설정:
- `in_node_nf=1` (velocity norm)
- `in_edge_nf=2` (charge product + squared distance)
- `act_fn=SiLU` (Swish activation)
- `coords_weight=1.0`

---

## 7. 데이터 흐름 정리 (Key Insights)

### 7.1 전체 데이터 흐름

```
[Physics Simulation]
ChargedParticlesSim.sample_trajectory()
  -> loc: (49, 3, 5), vel: (49, 3, 5), edges: (5, 5), charges: (5, 1)

[generate_dataset.py]
10000 simulations -> .npy files
  -> loc: (10000, 49, 3, 5)

[NBodyDataset.preprocess()]
transpose: (10000, 49, 3, 5) -> (10000, 49, 5, 3)
edges: (5, 5) matrix -> (20,) edge list + edge_attr

[NBodyDataset.__getitem__()]
frame selection: frame_0=30, frame_T=40
  -> loc_0: (5, 3), vel_0: (5, 3), edge_attr: (20, 1), charges: (5, 1), loc_T: (5, 3)

[DataLoader + train()]
batch reshape: (B, 5, 3) -> (B*5, 3)
get_edges(): node index offset for batched graph

[Model-specific preprocessing]
EGNN_vel:
  nodes = ||vel||          : (B*5, 1)   -- invariant scalar
  edge_attr += loc_dist    : (B*20, 2)  -- [charge_product; squared_distance]
  vel (separate input)     : (B*5, 3)   -- equivariant vector

[EGNN_vel forward]
h = embedding(nodes)       : (B*5, 64)
for each E_GCL_vel layer:
  radial, coord_diff = coord2radial(edges, x)
  m_ij = edge_model(h_i, h_j, radial, edge_attr)
  x += mean_j( coord_diff * coord_mlp(m_ij) )    -- equivariant position update
  x += coord_mlp_vel(h) * vel                     -- velocity-based update
  h = h + node_mlp([h; sum_j(m_ij)])              -- invariant feature update

[Loss]
MSE(x_pred, loc_T)
```

### 7.2 Equivariance가 유지되는 이유

1. **Node feature `h`는 invariant**: velocity norm, aggregated messages 등 모두 scalar/invariant 연산의 결과
2. **좌표 업데이트는 equivariant**: `coord_diff * scalar`는 equivariant (방향 벡터에 scalar를 곱함)
3. **Edge attribute는 invariant**: squared distance, charge product 모두 rotation에 불변
4. **Velocity 항**: `scalar * vel`은 equivariant (scalar가 h로부터 계산되므로 invariant, vel은 equivariant vector)

### 7.3 EGNN_vel vs EGNN 차이

| 측면 | EGNN | EGNN_vel |
|------|------|----------|
| Node feature | 상수 1 | velocity norm |
| Velocity 정보 | edge attr에 projected velocity로 포함 | 별도 벡터 입력, 좌표 업데이트에 직접 사용 |
| Edge attr | [charge; dist; vel_attr] (3D) | [charge; dist] (2D) |
| 좌표 업데이트 | interaction term만 | interaction + velocity term |
| Layer class | E_GCL | E_GCL_vel |

EGNN_vel이 velocity를 vector로 직접 다루는 것이 핵심이다. Projected velocity (scalar)로 정보를 잃는 것이 아니라, 전체 velocity vector를 equivariant하게 좌표 업데이트에 반영한다.

### 7.4 Aggregation 방식의 차이

- **Node update**: `unsorted_segment_sum` -- 이웃 메시지의 합
- **Coord update**: `unsorted_segment_mean` -- 이웃으로부터의 좌표 변위의 평균

좌표 업데이트에 mean을 쓰는 이유는, 이웃 수에 따라 업데이트 크기가 과도하게 커지는 것을 방지하기 위함이다.

### 7.5 .detach() 사용 패턴

```python
nodes = torch.sqrt(torch.sum(vel ** 2, dim=1)).unsqueeze(1).detach()
loc_pred = model(nodes, loc.detach(), edges, vel, edge_attr)
```

`loc`과 `nodes`에 `.detach()`를 사용하여 입력 좌표/feature를 통한 gradient를 차단한다. 이는 모델 내부의 좌표 업데이트 메커니즘(`coord_mlp`, `coord_mlp_vel`)만이 gradient를 받도록 하여, 학습을 안정화한다. `vel`에는 detach를 하지 않는 점이 주목할 만하다 -- `vel`을 통한 gradient flow는 허용되지만, 실제로 `vel`은 입력 데이터이므로 gradient가 모델 파라미터로만 흐른다.
