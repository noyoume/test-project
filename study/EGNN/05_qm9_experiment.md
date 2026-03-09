# 05. QM9 Molecular Property Prediction Experiment

> **Paper**: E(n) Equivariant Graph Neural Networks (arXiv:2102.09844), Section 5.3
> **Task**: 분자의 양자역학적 성질 예측 (invariant scalar property prediction)

---

## 1. 실험 개요

QM9 데이터셋은 최대 29개의 원자로 구성된 약 134k개의 소분자에 대해 DFT(밀도범함수이론)로 계산한 양자역학적 성질을 담고 있다.

### 예측 대상: 12가지 분자 성질

```
--property 옵션으로 선택: alpha | gap | homo | lumo | mu | Cv | G | H | r2 | U | U0 | zpve
```

| Property | 설명 | 단위 |
|----------|------|------|
| alpha | Isotropic polarizability | bohr^3 |
| gap | HOMO-LUMO gap | eV |
| homo | HOMO energy | eV |
| lumo | LUMO energy | eV |
| mu | Dipole moment | Debye |
| Cv | Heat capacity | cal/(mol K) |
| G | Free energy | eV |
| H | Enthalpy | eV |
| r2 | Electronic spatial extent | bohr^2 |
| U | Internal energy | eV |
| U0 | Internal energy at 0K | eV |
| zpve | Zero-point vibrational energy | eV |

### 평가 지표: MAE (Mean Absolute Error)

```python
# L1 loss = MAE
loss_l1 = nn.L1Loss()
```

각 property별로 개별 모델을 학습하며, test set에서의 MAE를 보고한다.

---

## 2. Data Pipeline

### 2.1 QM9 Dataset 기본 구조

- 최대 29개의 원자 (분자마다 원자 수가 다름)
- 5가지 원자 종류: H(1), C(6), N(7), O(8), F(9) -- 원자번호가 charges
- 각 원자의 3D 좌표 (positions)

### 2.2 ProcessedDataset (`qm9/data/dataset.py`)

`ProcessedDataset`은 PyTorch `Dataset`을 상속하며 다음을 수행한다:

#### (a) One-hot encoding 생성

```python
# charges: [N_molecules, max_atoms] -- atomic number of each atom (e.g., 6 for Carbon)
# included_species: [1, 6, 7, 8, 9] (H, C, N, O, F)
self.data['one_hot'] = self.data['charges'].unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0)
# Result: [N_molecules, max_atoms, 5] boolean tensor
```

Broadcasting으로 각 원자의 atomic number를 5종의 원소와 비교하여 one-hot 벡터를 생성한다.

#### (b) Thermochemical energy subtraction

```python
if subtract_thermo:
    thermo_targets = [key.split('_')[0] for key in data.keys() if key.endswith('_thermo')]
    for key in thermo_targets:
        data[key] -= data[key + '_thermo'].to(data[key].dtype)
```

QM9의 에너지 관련 성질(U0, U, G, H 등)에서 개별 원자의 thermochemical energy 기여분을 빼준다. 이렇게 하면 모델이 **분자 구조에서 오는 에너지 차이**만 학습하게 된다. 이는 QM9 벤치마크에서 표준적으로 사용하는 전처리다.

#### (c) Shuffling

```python
if shuffle:
    self.perm = torch.randperm(len(data['charges']))[:self.num_pts]
```

학습 데이터를 랜덤 순서로 셔플한다. `__getitem__`에서 `self.perm[idx]`로 인덱싱.

#### (d) 단위 변환 (Unit Conversion)

```python
# dataset.py (retrieve_dataloaders)
qm9_to_eV = {
    'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114,
    'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114
}
for dataset in datasets.values():
    dataset.convert_units(qm9_to_eV)
```

원래 QM9 데이터는 Hartree 단위로 저장되어 있다. `convert_units`는 각 property에 변환 계수를 곱하여 eV 단위로 바꾼다.

- 1 Hartree = 27.2114 eV
- zpve는 meV 수준이므로 27211.4를 곱해서 meV 단위로 변환

mu, alpha, r2, Cv 등은 `qm9_to_eV` dict에 포함되지 않으므로 원래 단위 그대로 사용한다.

### 2.3 collate_fn (`qm9/data/collate.py`)

DataLoader에서 배치를 구성할 때 호출되는 함수다. 분자마다 원자 수가 다르므로 padding과 masking이 필수적이다.

#### (a) batch_stack: padding

```python
def batch_stack(props):
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)       # scalar properties (labels)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)
        # Variable-length tensors -> padded to max length in batch
```

`pad_sequence`로 배치 내 가장 큰 분자 크기에 맞춰 0으로 padding한다.

#### (b) drop_zeros: 불필요한 padding 제거

```python
to_keep = (batch['charges'].sum(0) > 0)  # which atom positions have at least one real atom
batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}
```

배치 전체에서 모든 분자가 padding인 column을 제거하여 불필요한 계산을 줄인다.

#### (c) atom_mask 생성

```python
atom_mask = batch['charges'] > 0  # [batch_size, n_nodes]
batch['atom_mask'] = atom_mask
```

실제 원자인 위치는 `True`, padding 위치는 `False`. charges가 0인 곳이 padding이다.

#### (d) edge_mask 생성

```python
batch_size, n_nodes = atom_mask.size()

# Both source and target must be real atoms
edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
# [B, n_nodes, 1] * [B, 1, n_nodes] -> [B, n_nodes, n_nodes]

# Mask diagonal (self-loops removed from edge_mask)
diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
edge_mask *= diag_mask

# Flatten for use in message passing
batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)
```

**중요**: `edge_mask`는 collate 단계에서 self-loop을 제거하지만, `get_adj_matrix`는 self-loop을 포함한 fully connected graph를 생성한다. 결과적으로 edge_mask가 self-loop edge의 feature를 0으로 만들어 **실질적으로 self-loop이 무효화**된다. 즉, adjacency 구조상 self-loop이 존재하지만 edge_mask에 의해 그 기여가 차단된다.

### 2.4 preprocess_input (`qm9/utils.py`)

Atomic charges를 polynomial 방식으로 확장하여 node feature를 구성한다.

```python
def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    # charges: [B, n_nodes] -- atomic numbers
    # charge_scale: scalar -- normalization factor from dataset
    # charge_power: 2 (default)

    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    # charge_tensor: [B, n_nodes, 1, charge_power+1]
    # For charge_power=2: [z^0, z^1, z^2] = [1, z/scale, (z/scale)^2]

    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    # one_hot: [B, n_nodes, 5, 1]
    # charge_tensor: [B, n_nodes, 1, 3]
    # outer product -> [B, n_nodes, 5, 3]
    # flatten -> [B, n_nodes, 15]
    return atom_scalars
```

**결과 차원**: 5 atom types * (charge_power + 1) = 5 * 3 = **15차원** node feature

이 방식은 one-hot에 charge의 polynomial 정보를 곱하여 원소별로 다른 가중치를 부여하는 효과가 있다. 예를 들어 Carbon(z=6)의 경우 `[1, 6/scale, 36/scale^2]`이 Carbon의 one-hot 위치에만 들어간다. 이는 Cormorant 논문에서 가져온 전처리 방식이다.

---

## 3. E_GCL_mask (`qm9/models.py`)

### 3.1 E_GCL과의 관계

`E_GCL_mask`는 `E_GCL` (`models/gcl.py`)을 상속하되, QM9 task에 맞게 두 가지를 변경한다:

#### (a) coord_mlp 삭제

```python
def __init__(self, ...):
    E_GCL.__init__(self, ...)
    del self.coord_mlp   # <-- coord update MLP removed
    self.act_fn = act_fn
```

부모 클래스 `E_GCL.__init__`에서 `self.coord_mlp`가 생성되지만, 즉시 `del`로 제거한다. 이 MLP는 좌표 업데이트에 사용되는데, QM9에서는 필요하지 않기 때문이다.

#### (b) coord_model은 정의되어 있지만 호출되지 않음

```python
def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
    # This method exists but is NEVER called in forward()
    row, col = edge_index
    trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask  # would crash: coord_mlp deleted
    ...
```

`forward`에서 `coord_model` 호출이 **주석 처리**되어 있다:

```python
def forward(self, h, edge_index, coord, node_mask, edge_mask, ...):
    row, col = edge_index
    radial, coord_diff = self.coord2radial(edge_index, coord)

    edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
    edge_feat = edge_feat * edge_mask       # <-- mask padded edges

    #coord = self.coord_model(...)           # <-- COMMENTED OUT: no coordinate update

    h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
    return h, coord, edge_attr              # coord is returned unchanged
```

#### 왜 좌표 업데이트를 하지 않는가?

**QM9는 invariant task**이다. 분자 에너지, HOMO-LUMO gap 등은 스칼라 값으로, 분자를 회전/이동시켜도 바뀌지 않는다. 따라서:

1. 좌표로부터 **거리 정보**만 추출하면 충분하다 (`coord2radial`로 pairwise distance 계산)
2. 좌표를 업데이트할 이유가 없다 -- 예측 대상이 좌표가 아니라 스칼라 property
3. 좌표 업데이트를 제거함으로써 모델은 순수한 **invariant GNN**으로 동작한다

이것이 N-body 실험과의 가장 큰 구조적 차이점이다. N-body에서는 미래 좌표를 예측해야 하므로 equivariant한 좌표 업데이트가 핵심이었지만, QM9에서는 불필요하다.

#### (c) edge_mask 적용

```python
edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
edge_feat = edge_feat * edge_mask   # Mask out edges involving padding atoms
```

`edge_mask`는 `[B*n_nodes*n_nodes, 1]` 크기로, padding 원자가 관여하는 edge를 0으로 만든다. 이렇게 하면 padding 원자로부터의 메시지가 전파되지 않는다.

---

## 4. EGNN Wrapper (`qm9/models.py`)

### 4.1 전체 구조

```
Input: h0 [B*n_nodes, 15], x [B*n_nodes, 3]
  |
  v
embedding (Linear: 15 -> 128)
  |
  v
E_GCL_mask x 7 layers (default n_layers=7)
  |
  v
node_dec (Linear -> SiLU -> Linear: 128 -> 128)
  |
  v
node_mask 적용 (padding 노드 제거)
  |
  v
Sum pooling over nodes (graph-level representation)
  |
  v
graph_dec (Linear -> SiLU -> Linear: 128 -> 1)
  |
  v
Output: scalar prediction [B]
```

### 4.2 핵심 구현 사항

#### (a) Embedding Layer

```python
self.embedding = nn.Linear(in_node_nf, hidden_nf)  # 15 -> 128
```

15차원 node feature를 128차원 hidden space로 매핑한다.

#### (b) node_attr: 원본 feature를 매 layer에 공급

```python
# in forward():
for i in range(0, self.n_layers):
    if self.node_attr:
        h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask,
                                                edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
```

`node_attr=h0`로 원본 15차원 feature를 **매 layer의 node_model에 추가 입력**으로 전달한다. `E_GCL.node_model`에서 `[h, agg, node_attr]`을 concat하여 MLP에 넣는다:

```python
# In E_GCL.node_model (gcl.py):
if node_attr is not None:
    agg = torch.cat([x, agg, node_attr], dim=1)  # [h, aggregated_messages, original_features]
```

이는 deep network에서 원소 정보가 소실되지 않도록 하는 skip-connection 역할을 한다. `node_attr=1`이 기본값이므로, node_mlp의 입력 차원은 `hidden_nf + hidden_nf + in_node_nf = 128 + 128 + 15 = 271`이 된다.

#### (c) Sum Pooling & node_mask

```python
h = self.node_dec(h)         # [B*n_nodes, 128] -> [B*n_nodes, 128]
h = h * node_mask            # Zero out padding nodes
h = h.view(-1, n_nodes, self.hidden_nf)  # [B, n_nodes, 128]
h = torch.sum(h, dim=1)     # [B, 128] -- sum over all nodes
pred = self.graph_dec(h)     # [B, 128] -> [B, 1]
return pred.squeeze(1)       # [B]
```

**Sum pooling**은 node-level representation을 graph-level로 aggregation한다.

- `node_mask`를 곱해서 padding 노드의 기여를 완전히 제거
- `atom_mask`는 `[B*n_nodes, 1]` shape이므로 broadcasting으로 128차원 전체에 적용
- Sum pooling은 분자 크기에 대해 extensive한 성질(에너지 등)을 예측하기에 자연스럽다

#### (d) Activation Function 차이

```python
# EGNN wrapper uses SiLU (Swish)
act_fn=nn.SiLU()

# E_GCL base class defaults to ReLU, but EGNN passes SiLU to E_GCL_mask
```

QM9 모델에서는 `nn.SiLU()`를 사용한다. SiLU는 smooth하고 non-monotonic한 activation으로, molecular property 예측에서 더 좋은 성능을 보이는 경향이 있다.

---

## 5. Training Pipeline (`main_qm9.py`)

### 5.1 Mean/MAD Normalization

```python
meann, mad = qm9_utils.compute_mean_mad(dataloaders, args.property)
```

```python
def compute_mean_mad(dataloaders, label_property):
    values = dataloaders['train'].dataset.data[label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)    # MAD = Mean Absolute Deviation
    return meann, mad
```

Training set의 label에 대해 **mean**과 **MAD** (Mean Absolute Deviation)을 계산한다.
MAD는 standard deviation과 유사하지만 outlier에 덜 민감하다.

### 5.2 Train vs Test의 Loss 계산 차이

```python
if partition == 'train':
    # Normalized label space에서 loss 계산
    loss = loss_l1(pred, (label - meann) / mad)
    loss.backward()
    optimizer.step()
else:
    # Original scale로 denormalize하여 loss 계산
    loss = loss_l1(mad * pred + meann, label)
```

**핵심**: 모델은 정규화된 target `(y - mean) / mad`을 예측하도록 학습된다.
- **Train**: label을 정규화하여 loss 계산 -> gradient가 안정적
- **Test**: 모델 출력을 역변환 `mad * pred + meann`하여 원래 스케일의 MAE 보고

이 normalization 전략은 property마다 스케일이 크게 다르기 때문에 (예: zpve는 meV, U0는 eV) 학습 안정성에 중요하다.

### 5.3 Optimizer & Scheduler

```python
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-16)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
```

- **Adam** optimizer: lr=1e-3, weight_decay=1e-16 (사실상 no regularization)
- **CosineAnnealingLR**: 1000 epochs에 걸쳐 cosine 스케줄로 lr 감소
- 매 epoch 시작 시 `lr_scheduler.step()` 호출

### 5.4 Model Selection

```python
if val_loss < res['best_val']:
    res['best_val'] = val_loss
    res['best_test'] = test_loss
    res['best_epoch'] = epoch
```

Validation set에서 가장 좋은 loss를 기록한 시점의 test loss를 최종 결과로 보고한다. 단, 모델 checkpoint를 저장하지는 않고 test loss 값만 기록한다.

### 5.5 Hyperparameters 정리

| Parameter | Default Value |
|-----------|---------------|
| batch_size | 96 |
| epochs | 1000 |
| learning_rate | 1e-3 |
| hidden_nf | 128 |
| n_layers | 7 |
| attention | 1 (enabled) |
| node_attr | 0 (disabled by default, but paper likely uses 1) |
| charge_power | 2 |
| weight_decay | 1e-16 |
| in_node_nf | 15 (= 5 atom types * 3 charge powers) |
| in_edge_nf | 0 (no explicit edge features) |

---

## 6. get_adj_matrix (`qm9/utils.py`)

```python
edges_dic = {}  # Cache for reuse

def get_adj_matrix(n_nodes, batch_size, device):
    # Creates fully connected graph INCLUDING self-loops
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):       # i == j included (self-loops)
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)
    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges
```

### N-body 실험과의 차이점

| | QM9 | N-body |
|---|---|---|
| Graph 구성 | Fully connected (모든 원자 쌍) | Fully connected (모든 입자 쌍) |
| Self-loops | **포함** (i==j 허용) | 제외 |
| Edge masking | edge_mask로 padding edges 제거 + **self-loop 무효화** | 없음 (고정 크기) |
| 노드 수 | 가변 (분자마다 다름, 최대 29) | 고정 (5개) |

`get_adj_matrix`는 self-loop을 포함하지만, `collate_fn`의 `edge_mask`가 대각선을 mask하므로 실질적으로 self-loop 메시지는 0이 된다. 결국 **self-loop 없는 fully connected graph**와 동일하게 동작한다.

또한, `edges_dic`에 결과를 캐싱하여 같은 `(n_nodes, batch_size)` 조합에 대해 반복 생성을 피한다. 단, 현재 코드에 캐시 저장 로직 (`edges_dic[n_nodes][batch_size] = edges`)이 빠져 있어 실제로는 매번 재생성되는 버그가 있다.

---

## 7. N-body EGNN과의 핵심 설계 차이점

| 측면 | N-body EGNN | QM9 EGNN |
|------|-------------|----------|
| **Task type** | Equivariant (좌표 예측) | Invariant (스칼라 예측) |
| **Coordinate update** | 활성화 (coord_mlp 사용) | **비활성화** (coord_mlp 삭제) |
| **Output** | 업데이트된 좌표 [N, 3] | 그래프 수준 스칼라 [B, 1] |
| **Pooling** | 없음 (node-level output) | Sum pooling + graph_dec |
| **Node features** | 위치만 (feature 없음) | 15차원 (one-hot * charge polynomial) |
| **Edge features** | 없음 | 없음 (in_edge_nf=0) |
| **Masking** | 불필요 (고정 크기) | atom_mask + edge_mask (가변 크기) |
| **Label normalization** | 없음 | Mean/MAD normalization |
| **GCL class** | E_GCL (coord update 포함) | E_GCL_mask (coord update 제거) |
| **Activation** | 미지정 (default ReLU) | SiLU |
| **Aggregation** | unsorted_segment_mean (coord) | unsorted_segment_sum (message) |
| **Layers** | 4 | 7 |
| **Hidden dim** | 64 | 128 |

### 구조적 차이의 핵심 원인

N-body는 **equivariant output** (미래 좌표)을 필요로 하므로, EGNN의 equivariant coordinate update가 핵심이다. 반면 QM9는 **invariant output** (분자 에너지 등)을 필요로 하므로:

1. 좌표 업데이트가 불필요하다 -- 입력 좌표에서 pairwise distance만 추출하면 된다
2. Sum pooling이 필요하다 -- node-level features를 graph-level로 집계해야 한다
3. Masking이 필요하다 -- 분자마다 원자 수가 다르므로 padding 처리가 필수

즉, QM9에서 EGNN은 **equivariant coordinate update를 제거한 invariant GNN**으로 작동하며, pairwise distance를 edge feature로 사용하여 SE(3) invariance를 보장한다. `coord2radial`이 `||x_i - x_j||^2`을 계산하므로 회전/병진에 불변인 feature만 사용한다.

---

## 부록: 전체 Data Flow 요약

```
[Raw QM9 Data]
    |
    v
ProcessedDataset:
    - one_hot encoding (5 atom types)
    - thermochemical subtraction
    - unit conversion (Hartree -> eV)
    - shuffle
    |
    v
collate_fn (per batch):
    - pad_sequence (max atom count in batch)
    - drop_zeros (remove all-padding columns)
    - atom_mask: [B, n_nodes] (real atoms = 1)
    - edge_mask: [B*n*n, 1] (real edges = 1, self-loops = 0, padding = 0)
    |
    v
preprocess_input:
    - one_hot [B, n, 5] x charge_polynomial [B, n, 1, 3]
    - -> node features [B*n, 15]
    |
    v
get_adj_matrix:
    - fully connected edges [2, B*n*n] (including self-loops)
    |
    v
EGNN Forward:
    - embedding: 15 -> 128
    - 7x E_GCL_mask:
        - coord2radial: ||x_i - x_j||^2 (invariant distance)
        - edge_model: [h_i, h_j, radial] -> edge_feat
        - edge_feat *= edge_mask (mask padding + self-loops)
        - node_model: [h_i, agg(edge_feat), node_attr] -> h_i'  (+ residual)
        - coord is NOT updated
    - node_dec: 128 -> 128
    - *= node_mask (zero out padding)
    - sum pooling: [B, n, 128] -> [B, 128]
    - graph_dec: 128 -> 1
    |
    v
Loss:
    - Train: L1(pred, (label - mean) / mad)
    - Test:  L1(mad * pred + mean, label)
```
