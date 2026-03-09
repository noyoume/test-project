# 06. Graph Autoencoder Experiment (Section 5.2)

논문의 Section 5.2에서는 Graph Autoencoder 실험을 통해 EGNN이 graph reconstruction task에서
기존 GNN 대비 우수한 성능을 보임을 검증한다.

---

## 1. Experiment Overview

이 실험의 목표는 **그래프의 adjacency matrix를 node embedding으로부터 복원**하는 것이다.

```
Input Graph --> Encoder --> Node Embeddings z_i --> Decoder --> Reconstructed Adjacency Matrix
```

주어진 그래프 G = (V, E)에 대해:
1. **Encoder**: GNN 레이어를 통해 각 노드 i에 대한 embedding z_i를 생성
2. **Decoder**: 두 노드 embedding 간의 거리로부터 edge 존재 여부를 예측

핵심은 encoder가 그래프 구조를 충분히 반영하는 embedding을 학습해야
decoder가 원래의 adjacency matrix를 정확히 복원할 수 있다는 점이다.

---

## 2. The Symmetry Problem: Why Featureless Graphs Need Noise Injection

### 문제 정의

이 실험에서 사용하는 그래프는 **featureless graph**이다. 즉 모든 노드의 feature가 동일하다:

```python
# graph.py - networkx2graph()
nodes = torch.ones((G_nx.number_of_nodes(), 1))  # all nodes have feature = 1.0
```

이 경우 표준 GNN은 심각한 문제에 직면한다: **모든 노드가 동일한 입력 feature를 가지므로,
같은 degree를 가진 노드들은 구별 불가능한 동일한 embedding을 갖게 된다.**

예를 들어 5개 노드의 그래프에서 노드 A와 B가 모두 degree 3이라면,
message passing 후에도 z_A = z_B가 되어 decoder가 이 두 노드를 구별할 수 없다.
이것이 **symmetry problem**이다.

### 해결: Noise Injection

AE 모델은 이 문제를 noise injection으로 해결한다:

```python
# models/ae.py - AE.encode()
def encode(self, nodes, edges, edge_attr=None):
    if self.noise_dim:
        nodes = torch.randn(nodes.size(0), self.noise_dim).to(self.device)
    # ... proceed with GCL layers
```

각 노드에 random noise를 부여함으로써 symmetry를 깨뜨린다.
그러나 이 방법은 **모델이 noise에 대해 robust해야 한다**는 추가 부담을 준다.
같은 그래프에 대해 매번 다른 noise가 주어지므로 일관된 embedding을 학습하기 어렵다.

### EGNN의 접근: Equivariant Noise Processing

AE_rf와 AE_EGNN은 다른 전략을 취한다. Random noise를 **좌표 (coordinates)**로 사용한다:

```python
# models/ae.py - AE_EGNN.encode()
coords = torch.randn(h.size(0), self.K).to(self.device)  # random initial coords
for i in range(0, self.n_layers):
    h, coords, _ = self._modules["gcl_%d" % i](h, edges, coords, edge_attr=edge_attr)
    coords -= self.reg * coords  # regularization: pull coords toward origin
return coords  # <-- returns coords, not h!
```

EGNN는 이 random 좌표를 E(n) equivariant하게 처리한다. 핵심 차이점:
- **AE (GCL)**: noise를 node feature로 주입 -> feature space에서 symmetry를 깨뜨림
- **AE_rf / AE_EGNN**: noise를 좌표로 주입 -> coordinate space에서 equivariant하게 처리

EGNN의 equivariance 덕분에 noise의 회전/평행이동에 대해 불변인 표현을 학습할 수 있다.
이로 인해 noise에 대한 민감도가 줄어들어 더 안정적인 학습이 가능하다.

---

## 3. Decoder Equation (Paper Eq. 9)

모든 모델이 공유하는 decoder의 핵심 수식:

```
A_hat_ij = sigma(w * ||z_i - z_j||^2 + b)
```

여기서:
- `z_i, z_j`: 노드 i, j의 embedding (or coordinates)
- `w`: learnable weight (음수로 초기화: -0.1)
- `b`: learnable bias (1.0으로 초기화)
- `sigma`: sigmoid function

**직관**: 두 노드의 embedding이 가까우면 (||z_i - z_j||^2이 작으면) edge가 존재할 확률이 높다.
w가 음수이므로 거리가 작을수록 sigmoid 입력이 커져서 확률이 높아진다.

### 코드 구현 (AE_parent.decode_from_x)

```python
def decode_from_x(self, x, linear_layer=None, C=10, b=-1, remove_diagonal=True):
    n_nodes = x.size(0)
    x_a = x.unsqueeze(0)          # shape: (1, n_nodes, K)
    x_b = torch.transpose(x_a, 0, 1)  # shape: (n_nodes, 1, K)
    X = (x_a - x_b) ** 2          # shape: (n_nodes, n_nodes, K) -- broadcasting
    X = X.view(n_nodes ** 2, -1)  # shape: (n_nodes^2, K)

    if linear_layer is not None:
        # AE model: learnable linear layer on squared differences
        X = torch.sigmoid(linear_layer(X))     # nn.Linear(K, 1)
    else:
        # AE_rf / AE_EGNN: w * sum(squared_diff) + b
        X = torch.sigmoid(C * torch.sum(X, dim=1) + b)

    adj_pred = X.view(n_nodes, n_nodes)
    if remove_diagonal:
        adj_pred = adj_pred * (1 - torch.eye(n_nodes).to(self.device))  # no self-loops
    return adj_pred
```

**두 가지 decoder variant:**
1. **AE (learnable_dec=1)**: `nn.Linear(embedding_nf, 1)`로 squared difference의 각 차원에 다른 weight를 적용
2. **AE_rf / AE_EGNN**: 스칼라 `w`와 `b`로 squared difference의 합에 적용 (논문의 Eq. 9 그대로)

---

## 4. Dataset Generation

### 4.1 DatasetErdosRenyi

고정된 노드 수와 엣지 수를 가진 random graph를 생성한다.

```python
# d_creator.py
G = nx.gnm_random_graph(self.n_nodes, self.n_edges, directed=False)
```

- `nx.gnm_random_graph(n, m)`: n개 노드에서 m개 엣지를 균일 랜덤으로 선택
- Undirected로 생성 후 `to_directed()`로 변환 (양방향 edge)
- 사용 예: `erdosrenyi_100_16_20` -> 100 samples, 16 nodes, 20 edges

### 4.2 DatasetErdosRenyiNodes

**가변 노드 수**를 가진 random graph. 일반화 능력 테스트에 중요하다.

```python
# d_creator.py
self.n_nodes = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
# ...
G = nx.gnp_random_graph(n_nodes, self.p, directed=False)
```

- `nx.gnp_random_graph(n, p)`: n개 노드에서 각 edge가 확률 p로 존재 (Erdos-Renyi G(n,p) model)
- 노드 수 7~16으로 다양하게 생성
- 각 노드 수별로 `n_samples / len(n_nodes)`개씩 생성 후 shuffle
- `overfit=True` 시 노드 수별 10개만 생성 (디버깅/과적합 테스트용)
- Split: train 5000 / val 500 / test 500 (seed로 재현성 보장)

### 4.3 DatasetCommunity

두 개의 커뮤니티로 구성된 그래프. 더 복잡한 구조를 가진다.

```python
# d_creator.py - n_community()
def n_community(c_sizes, p_inter=0.01):
    # Each community: dense random graph with p=0.7
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    # Add sparse inter-community edges with p_inter=0.01
    for n1 in nodes1:
        for n2 in nodes2:
            if np.random.rand() < p_inter:
                G.add_edge(n1, n2)
    # Guarantee at least one inter-community edge
    if not has_inter_edge:
        G.add_edge(nodes1[0], nodes2[0])
```

- 각 커뮤니티 크기: [6, 7, 8, 9, 10] 중 랜덤 선택
- 커뮤니티 내부: 높은 연결 확률 (p=0.7) -> dense subgraph
- 커뮤니티 간: 낮은 연결 확률 (p_inter=0.01) -> sparse connection
- 최소 1개의 inter-community edge 보장 -> 전체 그래프 연결성 유지
- 총 노드 수: 12~20 (두 커뮤니티 합)

### 4.4 Dataset Selection (d_selector.py)

```python
# d_selector.py - retrieve_dataset()
# dataset_name format examples:
#   'erdosrenyinodes_0.25_none'    -> ErdosRenyiNodes, p=0.25, no overfit
#   'erdosrenyinodes_0.25_overfit' -> ErdosRenyiNodes, p=0.25, overfit mode
#   'community_ours'               -> DatasetCommunity, standard split
#   'community_overfit'             -> DatasetCommunity, 100 samples only
```

Dataset 이름을 문자열로 파싱하여 적절한 클래스를 instantiate한다.
기본값은 `community_ours`.

---

## 5. Graph Class

`graph.py`의 `Graph` 클래스는 그래프를 tensor 기반으로 표현한다.

### 5.1 Node Storage

```python
class Graph:
    def __init__(self, nodes, edges='fc', edge_attr=None):
        self.nodes = nodes       # (n_nodes, n_features) tensor
        self.edges = edges       # [LongTensor(rows), LongTensor(cols)] - sparse COO format
        self.edge_attr = edge_attr  # (n_edges, edge_features) or None
```

- `nodes`: 노드 feature matrix. 이 실험에서는 `torch.ones((n, 1))` (featureless)
- `edges`: COO format의 edge index. `[row_indices, col_indices]`
- `edge_attr`: edge feature. 이 실험에서는 `None` (생성 시)

### 5.2 Edge Creation

`edges='fc'`일 경우 fully-connected edges를 자동 생성:

```python
def _create_fc_edges(self):
    # Create undirected fully connected edges (without self-loops)
    for i in range(len(self.nodes)):
        for j in range(i + 1, len(self.nodes)):
            e_rows.append(i); e_cols.append(j)
    e_rows += e_cols              # add reverse direction
    e_cols += e_rows[0:len(e_rows) // 2]
```

### 5.3 Adjacency Operations

```python
def get_adjacency(self, loops=False):
    # sparse edges -> dense adjacency matrix
    adjacency = sparse2dense(n_nodes, self.edges)
    if loops:
        adjacency = adjacency + torch.eye(n_nodes)
    else:
        adjacency = adjacency * (1 - torch.eye(n_nodes))  # remove self-loops
    return adjacency
```

`sparse2dense()`는 COO format edge index를 `torch.sparse.FloatTensor`를 이용해 dense matrix로 변환한다.

### 5.4 Dense Graph Conversion (get_dense_graph)

학습 시 사용되는 핵심 메서드. 원래의 sparse graph를 **fully-connected dense graph**로 변환한다:

```python
def get_dense_graph(self, store=True, loops=False):
    adjacency = self.get_adjacency(loops)
    edges_dense, edge_attr_dense = self._dense2attributes(n_nodes, adjacency)
    return self.nodes, edges_dense, edge_attr_dense, adjacency
```

`_dense2attributes()`는:
- 모든 노드 쌍 (i != j)에 대해 edge를 생성 (fully connected, no self-loops)
- `edge_attr`에 adjacency 값 (0 or 1)을 저장

이렇게 하면 encoder는 **fully connected graph** 위에서 message passing을 수행하되,
edge attribute로 원래의 연결 정보를 전달받는다.

### 5.5 NetworkX Conversion

```python
def networkx2graph(G_nx):
    # networkx -> Graph: relabel nodes to 0-indexed, extract edges
    nodes = torch.ones((G_nx.number_of_nodes(), 1))
    rows = torch.LongTensor([edge[0] for edge in G_nx.edges])
    cols = torch.LongTensor([edge[1] for edge in G_nx.edges])
    return Graph(nodes, [rows, cols], edge_attr=None)
```

---

## 6. Model Architectures

### 6.1 AE_parent (Base Class)

모든 autoencoder의 부모 클래스. `encode()`, `decode()`, `decode_from_x()`, `forward()`를 정의한다.

```python
class AE_parent(nn.Module):
    def forward(self, nodes, edges, edge_attr=None):
        x = self.encode(nodes, edges, edge_attr)  # -> embeddings
        adj_pred = self.decode(x)                   # -> adjacency matrix
        return adj_pred, x
```

### 6.2 AE: GCL Encoder + Learnable Sigmoid Decoder

```
Architecture:
  Input (noise_dim or 1) -> GCL_0 (hidden_nf) -> GCL_1 -> ... -> GCL_{n-1} -> Linear(embedding_nf) -> decode
```

- **Encoder**: `n_layers`개의 GCL (Graph Convolutional Layer) 적용 후 linear projection
- **Decoder**: `nn.Linear(embedding_nf, 1)` -- squared difference의 각 차원에 별도 weight 학습
- **Symmetry 해결**: `noise_dim > 0`이면 입력 feature를 random noise로 대체

```python
# Encoder
self.add_module("gcl_0", GCL(max(1, self.noise_dim), self.hidden_nf, ...))
for i in range(1, n_layers):
    self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, ...))
self.fc_emb = nn.Linear(self.hidden_nf, self.embedding_nf)

# Decoder
self.fc_dec = nn.Linear(self.embedding_nf, 1)  # learnable decoder
```

### 6.3 AE_rf: Radial Field Encoder + w,b Decoder

```
Architecture:
  Random coords (embedding_nf) -> GCL_rf_0 -> GCL_rf_1 -> ... -> GCL_rf_{n-1} -> decode(w, b)
```

- **Encoder**: `GCL_rf` (Radial Field) 레이어 사용. 좌표 공간에서 직접 동작
- **Decoder**: 학습 가능한 스칼라 `w`, `b` 사용 (논문 Eq. 9)
- **Symmetry 해결**: random 좌표를 초기값으로 사용, radial field가 equivariant하게 처리

```python
# Initial random coordinates as input (not node features)
x = torch.randn(nodes.size(0), self.embedding_nf).to(self.device)
for i in range(0, self.n_layers):
    x, _ = self._modules["gcl_%d" % i](x, edges, edge_attr=edge_attr)
return x
```

### 6.4 AE_EGNN: E_GCL Encoder + Coordinate-based Decoder

```
Architecture:
  h (ones) + random coords (K) -> E_GCL_0 -> E_GCL_1 -> ... -> E_GCL_{n-1} -> decode(w, b)
                                              (coords regularized each step)
```

- **Encoder**: `E_GCL` (Equivariant Graph Convolutional Layer) 사용
- **핵심 차별점**: node feature h와 coordinates를 동시에 업데이트하되, **최종 출력은 coordinates**
- **Decoder**: w, b 스칼라로 좌표 간 거리 기반 복원
- **Regularization**: 매 레이어 후 `coords -= reg * coords`로 좌표를 origin 방향으로 당김

```python
def encode(self, h, edges, edge_attr=None):
    coords = torch.randn(h.size(0), self.K).to(self.device)
    for i in range(0, self.n_layers):
        h, coords, _ = self._modules["gcl_%d" % i](h, edges, coords, edge_attr=edge_attr)
        coords -= self.reg * coords  # regularization: prevent coord explosion
    return coords  # <-- returns coordinates, not features!
```

Coordinate regularization의 역할:
- 좌표가 발산하는 것을 방지
- `reg=1e-3`으로 매우 약한 regularization (학습을 방해하지 않으면서 안정성 확보)
- `clamp` 옵션도 있지만 실제로는 활성화되지 않는다고 코드 주석에 명시

### 6.5 Baseline

```python
class Baseline(nn.Module):
    def forward(self, nodes, b, c):
        n_nodes = nodes.size(0)
        return torch.zeros(n_nodes, n_nodes) * self.dummy, torch.ones(n_nodes)
```

항상 zero adjacency matrix를 반환. 다른 모델의 성능 하한선(lower bound)을 측정하기 위한 것.
`self.dummy`는 gradient가 흐르도록 하는 dummy parameter.

### 6.6 Symmetry Problem 해결 방식 비교

| Model   | Symmetry Breaking 방식 | Equivariance | Decoder |
|---------|------------------------|--------------|---------|
| AE      | noise를 node feature로 주입 | 없음 | Learnable linear |
| AE_rf   | random 좌표 초기화 | Radial field (translation equivariant) | w, b scalar |
| AE_EGNN | random 좌표 초기화 | E(n) equivariant (E_GCL) | w, b scalar |

---

## 7. Loss Function

### BCE Loss (Binary Cross Entropy)

Adjacency matrix 복원은 binary classification 문제로 취급한다:
각 (i, j) 위치에 대해 edge 존재 여부(0 or 1)를 예측.

```python
# losess.py
def adj_bce(pred, gt, reduce='mean', weight=None):
    return F.binary_cross_entropy(pred.view(-1, 1), gt.view(-1, 1),
                                   reduction=reduce, weight=weight)
```

- `pred`: decoder 출력 (sigmoid 이미 적용됨, 0~1 사이 값)
- `gt`: ground truth adjacency matrix (0 or 1)
- 모든 (i, j) 쌍에 대해 flatten 후 BCE 계산

### VAE Loss Wrapper

```python
def vae_loss(adj_rec, adj_gt, mu, logvar, reduce='sum'):
    BCE = adj_bce(adj_rec, adj_gt, reduce)
    if mu is None:
        KLD = torch.zeros(1)  # AE mode: no KL divergence
    else:
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD
```

이 실험에서는 `mu=None`, `logvar=None`으로 호출되므로 KLD = 0.
순수한 AE로서 BCE만 사용한다.

**참고**: `main_ae.py`에서 `reduce='sum'`으로 호출되므로 loss는 모든 edge 쌍의 BCE 합이다.

---

## 8. Evaluation Metrics

### 8.1 adjacency_error

```python
def adjacency_error(adj_pred, adj_gt):
    adj_pred = (adj_pred > 0.5).type(torch.float32)  # threshold at 0.5
    adj_errors = torch.abs(adj_pred - adj_gt)
    wrong_edges = torch.sum(adj_errors)               # total mismatches
    adj_error = wrong_edges / (n_nodes**2 - n_nodes)  # normalized by possible edges
    return wrong_edges.item(), adj_error.item()
```

- Threshold 0.5로 binary prediction 생성
- `wrong_edges`: 잘못 예측한 edge 수 (FP + FN)
- `adj_error`: wrong_edges를 가능한 edge 수 (self-loop 제외)로 나눈 비율

### 8.2 F1 Score (tp_fp_fn)

```python
def tp_fp_fn(adj_pred, adj_gt):
    adj_pred = (adj_pred > 0.5).type(torch.float32)
    tp = torch.sum(adj_pred * adj_gt).item()         # True Positive
    fp = torch.sum(adj_pred * (1 - adj_gt)).item()    # False Positive
    fn = torch.sum((1 - adj_pred) * adj_gt).item()    # False Negative
    return tp, fp, fn
```

F1 score는 test 시 계산:

```python
# main_ae.py - test()
f1_score = 1.0 * res['tp'] / (res['tp'] + 0.5 * (res['fp'] + res['fn']))
```

F1 = TP / (TP + 0.5 * (FP + FN)) -- edge의 precision과 recall의 조화 평균.

---

## 9. Training Pipeline

### 9.1 전체 흐름

```python
# main_ae.py
# 1. Dataset loading
dataset = d_selector.retrieve_dataset(args.dataset, partition="train", directed=True)
train_loader = Dataloader(dataset, batch_size=1)  # batch_size=1!

# 2. Model creation (ae | ae_rf | ae_egnn | baseline)

# 3. Optimizer & Scheduler
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-16)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

# 4. Training loop
for epoch in range(args.epochs):
    train(epoch, train_loader)
    if epoch % test_interval == 0:
        res_val = test(epoch, val_loader)
        res_test = test(epoch, test_loader)
        # Early stopping based on val BCE
        if res_val['bce'] < best_bce_val:
            best_bce_val = res_val['bce']
            best_res_test = res_test
```

### 9.2 Batch Size = 1

각 그래프의 노드 수가 다를 수 있으므로 (특히 ErdosRenyiNodes, Community)
**batch_size=1**로 한 번에 하나의 그래프만 처리한다.
이는 가변 크기 그래프를 padding 없이 처리하기 위한 것이다.

### 9.3 CosineAnnealingLR

```python
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
```

Learning rate를 cosine schedule로 감소시킨다. 초기 lr=1e-4에서 시작하여
epochs 동안 점진적으로 0까지 감소한다.

### 9.4 Train Step Details

```python
def train(epoch, loader):
    lr_scheduler.step(epoch)
    for batch_idx, data in enumerate(loader):
        graph = data[0]
        # Convert to dense fully-connected graph
        nodes, edges, edge_attr, adj_gt = graph.get_dense_graph(store=True, loops=False)
        # Forward pass
        adj_pred, z = model(nodes, edges, edge_attr)
        # Loss (BCE only, no KLD)
        bce, kl = losess.vae_loss(adj_pred, adj_gt, None, None)
        loss = bce
        loss.backward()
        optimizer.step()
```

중요한 점: `get_dense_graph()`는 원래 sparse graph를 **fully-connected graph**로 변환한다.
원래의 adjacency 정보는 `edge_attr`에 담기고, `adj_gt`가 ground truth가 된다.
모델은 fully-connected edges 위에서 message passing을 수행하며, 원래 어떤 edge가 존재했는지를
edge attribute를 통해 힌트로 받는다.

### 9.5 Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs |
| `lr` | 1e-4 | Initial learning rate |
| `nf` | 64 | Hidden feature dimension |
| `emb_nf` | 8 | Embedding dimension (AE only) |
| `K` | 8 | Coordinate dimension (AE_EGNN) |
| `n_layers` | 4 | Number of GCL layers |
| `reg` | 1e-3 | Coordinate regularization (AE_rf, AE_EGNN) |
| `noise_dim` | 0 | Noise dimension for AE (0 = no noise) |
| `weight_decay` | 1e-16 | Adam weight decay (effectively 0) |
| `no-cuda` | True | CPU only (GPU not used in this experiment) |

### 9.6 Custom Dataloader

```python
class Dataloader():
    def __init__(self, dataset, batch_size=1, shuffle=True, n_nodes=0):
        self.dataset = dataset
        self.batch_size = batch_size
        if self.shuffle:
            random.shuffle(self.dataset.graphs)

    def __next__(self):
        if self.idx > len(self.dataset) - self.batch_size:
            self.idx = 0
            random.shuffle(self.dataset.graphs)  # reshuffle at end of epoch
            raise StopIteration
        else:
            samples = self.dataset[self.idx:self.idx + self.batch_size]
            self.idx += self.batch_size
            return samples
```

PyTorch의 DataLoader를 사용하지 않고 직접 구현한 간단한 iterator.
Graph 객체를 직접 반환하며, epoch이 끝날 때마다 reshuffle한다.

---

## 10. Key Insight: EGNN's Advantage

### 왜 EGNN이 이 task에서 유리한가?

이 실험의 핵심 통찰:

**1. Symmetry breaking이 필수적이다**

Featureless graph에서 GNN이 의미 있는 node-level embedding을 만들려면
반드시 symmetry를 깨뜨려야 한다. 모든 모델이 random noise를 사용하지만,
그것을 처리하는 방식이 다르다.

**2. Noise에 대한 equivariance가 일반화를 돕는다**

- **AE (GCL)**: noise를 feature로 사용. 같은 그래프에 대해 매번 다른 noise가 주어지면
  완전히 다른 embedding이 나올 수 있다. 모델이 noise-invariant한 표현을 학습해야 하지만
  이를 위한 구조적 보장이 없다.

- **AE_EGNN (E_GCL)**: noise를 좌표로 사용. E(n) equivariance 덕분에 noise의
  회전/반사/평행이동에 대해 일관된 출력을 생성한다. Decoder가 좌표 간 **거리**만 사용하므로
  (||z_i - z_j||^2), 좌표의 절대 위치가 아닌 상대적 관계만 중요하다.

**3. Decoder와의 궁합**

EGNN의 출력은 좌표이고, decoder는 좌표 간 거리를 사용한다.
좌표 간 거리는 E(n) 변환에 대해 불변이므로, encoder의 equivariance와 decoder의 invariance가
자연스럽게 정합(consistent)된다.

```
Random coords x0 --[E_GCL layers]--> Updated coords z
                                       |
                                       v
            A_hat_ij = sigma(w * ||z_i - z_j||^2 + b)
            ||z_i - z_j||^2 is invariant to rotation/translation of z
            => prediction is stable regardless of initial noise orientation
```

이 구조적 정합성이 EGNN이 featureless graph autoencoder에서
더 적은 에러율과 높은 F1 score를 달성하는 핵심 이유이다.
