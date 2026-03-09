# 07. SE(3) Transformer & TFN Comparison 분석

> EGNN 논문 (arXiv:2102.09844)에서 N-body experiment의 baseline으로 사용된 SE(3) Transformer와 TFN 구현을 분석한다.

---

## 1. Overview: 왜 이 모델들이 포함되었는가

EGNN 논문의 핵심 주장은 **spherical harmonics 없이도 SE(3)-equivariant한 메시지 패싱이 가능**하다는 것이다.
이를 입증하기 위해 N-body charged particle simulation 실험에서 다음 두 가지 기존 모델을 baseline으로 비교한다:

- **TFN (Tensor Field Network)**: spherical harmonics 기반의 SE(3)-equivariant GCN
- **SE(3) Transformer**: TFN에 attention mechanism을 추가한 모델

이 두 모델은 `egnn/n_body_system/se3_dynamics/` 디렉토리에 구현되어 있으며, 원래 graph-level prediction용으로 설계된 것을 node-level dynamics prediction에 맞게 수정한 `OursTFN`, `OurSE3Transformer` 변형이 함께 제공된다.

**핵심 비교 포인트**: TFN/SE3 Transformer는 equivariance를 보장하기 위해 Wigner-D matrix와 spherical harmonics라는 무거운 수학적 장치를 사용하는 반면, EGNN은 단순히 **scalar distance**만으로 동일한 성질을 달성한다.

---

## 2. Architecture Comparison: EGNN vs TFN vs SE(3) Transformer

### 2.1 TFN (Tensor Field Network)

TFN은 SE(3)-equivariant convolution을 수행하는 GCN이다. 핵심 아이디어는 **spherical harmonics를 basis로 사용하여 equivariant kernel을 구성**하는 것이다.

#### 원리: Equivariant Basis Construction

```python
# modules.py - get_basis_and_r()
def get_basis_and_r(G, max_degree):
    # 1. Cartesian -> Spherical coordinates (r, alpha, beta)
    r_ij = utils_steerable.get_spherical_from_cartesian_torch(G.edata['d'])

    # 2. Precompute spherical harmonics Y_J for all orders up to 2*max_degree
    Y = utils_steerable.precompute_sh(r_ij, 2 * max_degree)

    # 3. Build equivariant basis from Y and Clebsch-Gordan coefficients
    basis = get_basis(Y, max_degree)

    # 4. Scalar distances for radial functions
    r = torch.sqrt(torch.sum(G.edata['d']**2, -1, keepdim=True))
    return basis, r
```

**Equivariant basis 구성 과정** (`get_basis()` in `modules.py`):

각 (d_in, d_out) type 쌍에 대해, Clebsch-Gordan 분해에 의해 J = |d_in - d_out|, ..., d_in + d_out 범위의 irreducible representation이 필요하다:

```python
# modules.py - get_basis()
for d_in in range(max_degree+1):
    for d_out in range(max_degree+1):
        K_Js = []
        for J in range(abs(d_in-d_out), d_in+d_out+1):
            # Q_J: basis transformation matrix (Clebsch-Gordan coefficients)
            Q_J = utils_steerable._basis_transformation_Q_J(J, d_in, d_out)
            # K_J: equivariant kernel component
            K_J = torch.matmul(Y[J], Q_J)
            K_Js.append(K_J)
        # Stack into basis tensor
        # shape: (-1, 1, 2*d_out+1, 1, 2*d_in+1, 2*min(d_in,d_out)+1)
        basis[f'{d_in},{d_out}'] = torch.stack(K_Js, -1).view(*size)
```

이 basis는 다음의 수학적 관계에 기반한다:

- **Spherical harmonics** Y_J^m(r_ij): 방향 정보를 encoding하는 angular basis function
- **Wigner-D matrices** D^J(R): SO(3) 회전의 irreducible representation
- **Clebsch-Gordan coefficients** (Q_J): tensor product를 irreducible representation으로 분해

`_basis_transformation_Q_J()` 함수는 Sylvester equation의 kernel을 SVD로 풀어서 Q_J를 구한다:

```python
# utils_steerable.py - _basis_transformation_Q_J()
def _R_tensor(a, b, c):
    return kron(irr_repr(order_out, a, b, c), irr_repr(order_in, a, b, c))

def _sylvester_submatrix(J, a, b, c):
    R_tensor = _R_tensor(a, b, c)        # [m_out*m_in, m_out*m_in]
    R_irrep_J = irr_repr(J, a, b, c)     # [2J+1, 2J+1]
    # Sylvester equation: R_tensor @ Q_J = Q_J @ R_irrep_J
    return kron(R_tensor, eye) - kron(eye, R_irrep_J.t())

# Solve for Q_J in the null space
null_space = get_matrices_kernel([_sylvester_submatrix(J, a, b, c) for a,b,c in random_angles])
Q_J = null_space[0].view((2*order_out+1)*(2*order_in+1), 2*J+1)
```

#### TFN Convolution Layer (GConvSE3)

TFN의 convolution은 **radial function**과 **equivariant basis**의 결합으로 이루어진다:

```python
# modules.py - PairwiseConv.forward()
def forward(self, feat, basis):
    R = self.rp(feat)  # Radial profile: NN(||r_ij||) -> weights
    # Combine radial weights with angular basis
    kernel = torch.sum(R * basis[f'{self.degree_in},{self.degree_out}'], -1)
    return kernel.view(kernel.shape[0], self.d_out*self.nc_out, -1)
```

여기서 `RadialFunc`은 거리 정보만을 입력으로 받는 MLP이다:

```python
# modules.py - RadialFunc
self.net = nn.Sequential(
    nn.Linear(edge_dim+1, mid_dim),      # input: scalar distance (+ optional edge features)
    BN(mid_dim), act_fn,
    nn.Linear(mid_dim, mid_dim),
    BN(mid_dim), act_fn,
    nn.Linear(mid_dim, num_freq*in_dim*out_dim)  # output: radial weights
)
```

즉, TFN kernel = (scalar distance로 학습되는 radial function) x (spherical harmonics로 계산되는 angular basis)

#### TFN Model Variants

**원본 TFN** (`TFN` class): graph-level prediction용

```
Input: type-0 features only
-> GConvSE3 + GNormSE3 layers (num_layers-1 times)
-> GConvSE3 (final)
-> GMaxPooling (graph-level)
-> Linear + ReLU + Linear (scalar output)
```

**N-body 적용 TFN** (`OursTFN` class): node-level dynamics prediction용

```
Input: type-0 (charges) + type-1 (velocity) features
-> GConvSE3 + GNormSE3 layers (num_layers-1 times)
-> GConvSE3 (final, output type-1 only)
# No pooling, no FC layers - direct node-level output
```

주요 차이점:
- `OursTFN`은 `in_types={0: 1, 1: 1}`, `out_types={1: 1}`을 사용하여 type-1 (3D vector) 출력을 직접 반환
- Pooling과 FC layer가 제거됨 (node-level prediction이므로)

### 2.2 SE(3) Transformer

SE(3) Transformer는 TFN의 convolution에 **multi-head self-attention**을 추가한 모델이다.

#### Attention Mechanism (GSE3Res)

`GSE3Res`는 SE(3)-equivariant attention block이다:

```python
# modules.py - GSE3Res
class GSE3Res(nn.Module):
    def __init__(self, f_in, f_out, edge_dim, div, n_heads, act_fn, learnable_skip):
        # Value projection: full equivariant convolution (node -> edge)
        self.GMAB['v'] = GConvSE3Partial(f_in, f_mid_out, edge_dim=edge_dim)
        # Key projection: equivariant convolution (node -> edge)
        self.GMAB['k'] = GConvSE3Partial(f_in, f_mid_in, edge_dim=edge_dim)
        # Query projection: 1x1 convolution (node -> node, self-interaction only)
        self.GMAB['q'] = G1x1SE3(f_in, f_mid_in)
        # Multi-head attention
        self.GMAB['attn'] = GMABSE3(f_mid_out, f_mid_in, n_heads=n_heads)
```

Attention weight 계산 과정 (`GMABSE3`):

```python
# modules.py - GMABSE3.forward()
# 1. key-query inner product
G.apply_edges(fn.e_dot_v('k', 'q', 'e'))

# 2. Scaled softmax
e = e / np.sqrt(self.f_key.n_features)
G.edata['a'] = edge_softmax(G, e)

# 3. Attention-weighted message passing
# msg = attention_weight * value
msg = attn.unsqueeze(-1).unsqueeze(-1) * value
```

**핵심**: attention weight는 **scalar** (type-0)이다. Key와 query의 inner product로 계산되므로 자연스럽게 invariant하다. 이 scalar weight를 equivariant한 value에 곱하면 결과도 equivariant하다.

#### SE(3) Transformer Model Variants

**원본 SE3Transformer** (`SE3Transformer` class):

```
Input: type-0 features only
-> GSE3Res + GNormSE3 layers (num_layers times)
-> GConvSE3 (final)
-> GAvgPooling or GMaxPooling
-> Linear + ReLU + Linear
```

**N-body 적용** (`OurSE3Transformer` class):

```
Input: type-0 (charges) + type-1 (velocity) features
-> GSE3Res + GNormSE3 layers (num_layers times)
-> GConvSE3 (final, output type-1)
-> scalar_trick scaling (learnable parameter, initialized to 0.01)
```

`OurSE3Transformer`의 특이점:
- `scalar_trick`: 출력을 학습 가능한 작은 scalar로 스케일링한다. 초기화 시 0.01로 설정되어 학습 초기에 출력이 너무 크지 않도록 한다.
- `learnable_skip=False`: skip connection의 projection을 학습하지 않는다 (실제로 skip connection 자체가 주석 처리되어 비활성화됨).

```python
# models.py - OurSE3Transformer
self.scalar_trick = nn.Parameter(torch.ones(1)*0.01)

def forward(self, G):
    basis, r = get_basis_and_r(G, self.num_degrees-1)
    h = {'0': G.ndata['f'], '1': G.ndata['f1']}
    for layer in self.Gblock:
        h = layer(h, G=G, r=r, basis=basis)
    # Scale all output types by scalar_trick
    for key in h:
        h[key] = h[key] * self.scalar_trick
    return h
```

### 2.3 EGNN (E(n) Equivariant Graph Neural Network)

EGNN의 접근 방식은 근본적으로 다르다:

- **Spherical harmonics를 사용하지 않는다**
- **Scalar distance** ||x_i - x_j||^2 만으로 equivariance를 보장한다
- **좌표를 직접 업데이트**한다: x_i^{l+1} = x_i^l + C * sum_j (x_i - x_j) * phi_x(m_ij)

EGNN의 equivariance 보장 메커니즘:
1. Message function은 scalar distance만 사용하므로 invariant
2. 좌표 업데이트는 상대 위치 벡터 (x_i - x_j)에 invariant scalar를 곱하므로 equivariant
3. 별도의 representation theory가 필요 없다

---

## 3. OurDynamics Wrapper (dynamics.py)

`OurDynamics`는 N-body simulation의 array 형태 입력을 SE(3) Transformer/TFN이 처리할 수 있는 DGL graph 형태로 변환하는 wrapper이다.

### 3.1 입력 변환 과정

```python
# dynamics.py - OurDynamics.f()
def f(self, xs, vs, charges):
    # xs: (batch_size, num_nodes, 3) - positions
    # vs: (batch_size, num_nodes, 3) - velocities
    # charges: (batch_size, num_nodes, 1) - charges
```

**Graph 생성 (최초 1회)**:

```python
if self.graph is None:
    self.graph = array_to_graph(xs)
    # Zero out positions and distances (will be recomputed each forward)
    self.graph.ndata['x'] = torch.zeros_like(self.graph.ndata['x'])
    self.graph.edata['d'] = torch.zeros_like(self.graph.edata['d'])
    # Precompute fully-connected edge indices
    indices_src, indices_dst, _w = connect_fully(xs.size(1))
```

이후 forward pass에서는 graph 구조를 재사용하고, edge distance만 업데이트한다:

```python
# Efficient batched distance computation (no loop over batch)
distance = xs[:, self.indices_dst] - xs[:, self.indices_src]
```

### 3.2 Feature Assignment

```python
# Type-1 feature (vectors): velocity
self.graph.ndata['vel'] = vs.view(B*N, 3).unsqueeze(1)  # shape: [B*N, 1, 3]
self.graph.ndata['f1'] = self.graph.ndata['vel']         # f1 = velocity

# Type-0 feature (scalars): charges
self.graph.ndata['f'] = charges.unsqueeze(2)             # shape: [B*N, 1, 1]

# Edge features: relative positions
self.graph.edata['d'] = distance.view(-1, 3)             # shape: [B*N*(N-1), 3]
```

- **type-0 feature ('f')**: charge 값 (scalar) -- SE(3) 변환에 불변
- **type-1 feature ('f1')**: velocity 벡터 -- SE(3) 변환에 따라 회전

### 3.3 출력 변환 및 Residual Connection

```python
G_out = self.se3(G)

# Extract type-1 output and reshape to original dimensions
out = G_out['1'].view(xs.size())  # shape: (batch_size, num_nodes, 3)

return out + xs  # Residual: predicted displacement + current position
```

모델의 출력 type-1 feature를 position 공간으로 해석하고, 현재 위치 `xs`에 더하는 residual connection을 사용한다.
이는 모델이 **위치의 변화량(displacement)**을 예측하도록 유도한다.

### 3.4 connect_fully: Fully Connected Graph

```python
def connect_fully(num_atoms):
    # Create all edges except self-loops
    adjacency = {}
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                adjacency[(i, j)] = 1
    # Returns (src_indices, dst_indices, weights) as numpy arrays
```

N개 노드에 대해 N*(N-1)개의 edge를 생성한다 (self-loop 제외). N-body problem에서는 모든 입자가 서로 상호작용하므로 fully connected graph가 자연스럽다.

---

## 4. Fiber Class: SE(3) Feature Type 표현

`Fiber`는 SE(3) representation theory에서의 feature type 구조를 표현하는 데이터 구조이다.

### 4.1 Type System

SE(3) group의 irreducible representation은 non-negative integer `l` (degree/order)로 인덱싱된다:

| Type (l) | Dimension (2l+1) | 물리적 의미 | 변환 규칙 |
|----------|------------------|------------|----------|
| 0 | 1 | Scalar | 불변 (invariant) |
| 1 | 3 | Vector | 3D 회전 |
| 2 | 5 | Traceless symmetric tensor | 5D 회전 |
| 3 | 7 | Higher-order tensor | 7D 회전 |

### 4.2 Fiber 생성 방식

```python
# 방법 1: num_degrees개의 type, 각각 num_channels개의 channel
Fiber(num_degrees=4, num_channels=16)
# -> [(16, 0), (16, 1), (16, 2), (16, 3)]
# -> type-0: 16*1=16, type-1: 16*3=48, type-2: 16*5=80, type-3: 16*7=112
# -> total: 256 features

# 방법 2: dictionary로 직접 지정
Fiber(dictionary={0: 1, 1: 1})
# -> [(1, 0), (1, 1)]
# -> type-0: 1*1=1 (charge), type-1: 1*3=3 (velocity)
# -> total: 4 features
```

### 4.3 주요 속성과 메서드

```python
fiber = Fiber(num_degrees=4, num_channels=16)

fiber.structure        # [(16, 0), (16, 1), (16, 2), (16, 3)] -- (multiplicity, degree) pairs
fiber.structure_dict   # {0: 16, 1: 16, 2: 16, 3: 16} -- degree -> multiplicity
fiber.degrees          # (0, 1, 2, 3)
fiber.multiplicities   # (16, 16, 16, 16)
fiber.n_features       # 16*1 + 16*3 + 16*5 + 16*7 = 256 (total scalar dimension)
fiber.feature_indices  # {0: (0,16), 1: (16,64), 2: (64,144), 3: (144,256)}
```

### 4.4 N-body 실험에서의 Fiber 구성

```python
# OursTFN / OurSE3Transformer에서 사용하는 fiber 구조:
fibers = {
    'in':  Fiber(dictionary={0: 1, 1: 1}),   # input: 1 charge (scalar) + 1 velocity (vector)
    'mid': Fiber(num_degrees=4, num_channels=nf),  # hidden: 4 types, nf channels each
    'out': Fiber(dictionary={1: 1})           # output: 1 vector (predicted displacement)
}
```

---

## 5. Key Architectural Differences Table

| 항목 | TFN | SE(3) Transformer | EGNN |
|------|-----|-------------------|------|
| **Equivariance 보장 방식** | Spherical harmonics + Wigner-D | Spherical harmonics + Wigner-D + Attention | Scalar distance + relative position vector |
| **Feature 표현** | Fiber (type-0, 1, 2, ...) | Fiber (type-0, 1, 2, ...) | Scalar node features + 3D coordinates |
| **Convolution kernel** | Radial function x Angular basis | Radial function x Angular basis | MLP on scalar distances |
| **Attention** | 없음 | Equivariant multi-head attention | 없음 (또는 optional) |
| **Message 계산** | O(L^6) per edge (L = max degree) | O(L^6) per edge + attention overhead | O(1) per edge (degree-independent) |
| **좌표 업데이트** | Feature space에서 간접적 | Feature space에서 간접적 | 직접 좌표 업데이트 |
| **Basis 사전 계산** | 매 forward pass마다 필요 | 매 forward pass마다 필요 | 불필요 |
| **Graph library 의존성** | DGL 필수 | DGL 필수 | 범용 (DGL 불필요) |
| **Higher-order features** | type-0, 1, 2, 3, ... | type-0, 1, 2, 3, ... | type-0 (scalar) + 좌표만 사용 |
| **Normalization** | GNormSE3 (norm-based) | GNormSE3 (norm-based) | Standard LayerNorm/BatchNorm |
| **출력 스케일링** | 없음 | scalar_trick (learnable) | 없음 |
| **Skip connection** | 없음 | 주석 처리됨 (비활성) | 좌표 residual (x + dx) |

---

## 6. Computational Complexity Comparison

### 6.1 TFN/SE(3) Transformer의 비용

#### Spherical Harmonics 사전 계산

매 forward pass의 시작에서 `get_basis_and_r()`가 호출된다:

```python
# Step 1: Cartesian -> Spherical conversion (O(E) where E = num_edges)
r_ij = get_spherical_from_cartesian_torch(G.edata['d'])

# Step 2: Spherical harmonics up to order 2*max_degree
# For each order J = 0, 1, ..., 2*L:
#   Compute Y_J with (2J+1) components for each edge
# Cost: O(E * L^2) where L = max_degree
Y = precompute_sh(r_ij, 2 * max_degree)

# Step 3: Equivariant basis construction
# For each (d_in, d_out) pair, combine Y with Q_J (Clebsch-Gordan)
# Number of pairs: (L+1)^2
# Each pair involves J = |d_in-d_out|, ..., d_in+d_out
# Cost: O(E * L^4) for all pairs (dominated by highest degrees)
basis = get_basis(Y, max_degree)
```

#### Convolution 비용

각 `GConvSE3` layer에서:

```python
# For each (d_in, d_out) pair:
#   RadialFunc: MLP on scalar distance -> O(E * mid_dim^2)
#   Kernel construction: R * basis -> O(E * m_in * m_out * (2*min(d_in,d_out)+1))
#   Message: kernel @ src_features -> O(E * m_out * (2*d_out+1) * m_in * (2*d_in+1))
```

type L까지 사용할 때, 가장 큰 비용은 high-order type간의 convolution에서 발생한다:
- Kernel size: (2L+1) x (2L+1) per frequency
- Number of frequencies: 2*min(d_in, d_out)+1
- 전체적으로 **O(E * C^2 * L^3)** per layer (C = num_channels)

#### Attention 추가 비용 (SE(3) Transformer)

```python
# Value: GConvSE3Partial - same cost as GConvSE3 but without aggregation
# Key: GConvSE3Partial
# Query: G1x1SE3 - simple linear (cheap)
# Attention: dot product + softmax + weighted sum

# Total: roughly 3x the cost of a single GConvSE3 layer
```

### 6.2 EGNN의 비용

EGNN의 한 layer에서:

```python
# 1. Distance computation: O(E)
# 2. Message MLP: O(E * hidden_dim^2) - standard MLP on scalars
# 3. Coordinate update: O(E * 3) - multiply relative position by scalar weight
# 4. Node update MLP: O(N * hidden_dim^2)
```

**전체 비용: O(E * hidden_dim^2)** per layer -- degree L에 무관

### 6.3 비교 요약

| 비용 항목 | TFN/SE3T | EGNN |
|----------|----------|------|
| Basis 사전 계산 | O(E * L^4) | 0 |
| Layer당 convolution | O(E * C^2 * L^3) | O(E * C^2) |
| Memory (per edge) | O(L^4 * C^2) (basis tensors) | O(C) |
| 전체 (K layers) | O(K * E * C^2 * L^3 + E * L^4) | O(K * E * C^2) |

L=3 (num_degrees=4)일 때, TFN의 convolution 비용은 EGNN 대비 대략 **L^3 = 27배** 이상 크다. 여기에 basis 사전 계산 비용까지 더해진다.

---

## 7. Spherical Harmonics가 비싼 이유와 차원 한계

### 7.1 계산 비용

**Spherical harmonics Y_l^m(theta, phi)**의 계산에는 다음이 필요하다:

1. **Associated Legendre polynomials** P_l^m(cos(theta)):
   - 재귀적 계산이 필요하다 (representations.py의 `SphericalHarmonics.lpmv()`)
   - 각 (l, m) 쌍에 대해 P_0^0부터 P_l^m까지 bottom-up으로 계산
   - 수치 안정성을 위해 특수 함수(semifactorial, pochhammer 등)를 사용

2. **Trigonometric functions**: cos(m*phi), sin(m*phi) 계산

3. **Normalization constants**: sqrt((2l+1)/(4*pi)) 등

```python
# representations.py - SphericalHarmonics.get()
def get(self, l, theta, phi, refresh=True):
    results = []
    for m in range(-l, l+1):          # 2l+1 components
        results.append(self.get_element(l, m, theta, phi))
    return torch.stack(results, -1)   # shape: [..., 2l+1]
```

order l의 spherical harmonics는 (2l+1)개의 component를 가지며, 각각이 Legendre polynomial 계산을 필요로 한다.

### 7.2 Wigner-D Matrix와 Basis Transformation

**Wigner-D matrix** D^l(R)은 SO(3) 회전 R의 order-l irreducible representation이다:

```python
# SO3.py - irr_repr()
def irr_repr(order, alpha, beta, gamma, dtype=None):
    # Depends on external library: lie_learn
    from lie_learn.representations.SO3.wigner_d import wigner_D_matrix
    return torch.tensor(wigner_D_matrix(order, alpha, beta, gamma), ...)
```

**Basis transformation Q_J**의 계산이 특히 비싸다:

```python
# utils_steerable.py - _basis_transformation_Q_J()
# For each (J, d_in, d_out):
#   1. Construct Sylvester equation matrices using Kronecker products
#      Size: (m_out*m_in)*(2J+1) x (m_out*m_in)*(2J+1)
#      where m_in = 2*d_in+1, m_out = 2*d_out+1
#   2. Solve via SVD to find null space
```

이 행렬의 크기는 degree가 올라갈수록 급격히 커진다:
- d_in = d_out = 3일 때: (7*7*7) x (7*7*7) = 343 x 343 행렬의 SVD

다행히 이 값은 캐싱된다 (`@cached_dirpklgz("cache/trans_Q")`), 하지만 spherical harmonics와 basis 자체는 매 forward pass마다 재계산해야 한다 (edge distance가 변하므로).

### 7.3 차원 한계 (Dimension Limitation)

TFN/SE(3) Transformer는 **3D 공간에 특화**되어 있다:

1. **Spherical harmonics는 3D sphere (S^2)에서 정의**된다:
   - Cartesian (x, y, z) -> Spherical (r, alpha, beta) 변환이 전제
   - 이는 본질적으로 R^3에서의 방향 정보를 encoding하는 것

2. **SO(3) representation theory에 기반**한다:
   - Wigner-D matrices는 3D 회전군 SO(3)의 representation
   - N차원으로의 일반화는 SO(N) representation theory를 필요로 하며, 이는 훨씬 복잡

3. **EGNN의 장점: E(n) equivariance**:
   - EGNN은 ||x_i - x_j||^2 (scalar distance)와 (x_i - x_j) (relative position vector)만 사용
   - 이 두 연산은 **임의의 차원 n**에서 E(n)-equivariant
   - 따라서 EGNN은 3D뿐만 아니라 **임의의 차원에서 동작**한다

```
TFN/SE3T:  R^3 전용 (spherical harmonics가 S^2에 기반)
EGNN:      R^n 범용 (scalar distance는 차원에 무관)
```

### 7.4 GNormSE3: Higher-order Feature의 Nonlinearity 문제

Higher-order type feature에 nonlinearity를 적용하는 것도 비자명하다. `GNormSE3`은 norm-based 방식을 사용한다:

```python
# modules.py - GNormSE3.forward()
def forward(self, features, **kwargs):
    for k, v in features.items():
        # v shape: [..., m, 2*k+1]
        norm = v.norm(2, -1, keepdim=True)    # norm is invariant
        phase = v / norm                       # unit vector (equivariant)
        transformed = self.transform[str(k)](norm[...,0])  # NN on norms (invariant)
        output[k] = transformed * phase        # scale equivariant phase by invariant magnitude
```

이 방식은:
1. Feature의 norm (invariant scalar)을 추출
2. Norm에 학습 가능한 함수를 적용 (invariant -> invariant)
3. 원래 방향(phase)에 변환된 norm을 곱함

일반적인 ReLU 같은 nonlinearity는 equivariance를 깨뜨리기 때문에, 이러한 우회적인 방법이 필요하다. 이는 추가적인 계산 비용과 표현력의 제약을 야기한다.

---

## 8. 정리

EGNN 논문이 SE(3) Transformer와 TFN을 baseline으로 비교하는 이유는 명확하다:

1. **성능 동등성**: EGNN은 spherical harmonics 없이도 N-body simulation에서 TFN/SE(3) Transformer와 비슷하거나 더 나은 성능을 달성한다.

2. **효율성**: Basis 사전 계산과 higher-order tensor convolution이 없으므로, EGNN은 훨씬 빠르고 메모리 효율적이다.

3. **일반성**: EGNN은 E(n)-equivariant하므로 임의 차원에 적용 가능하지만, TFN/SE(3) Transformer는 SO(3) (3D 회전)에 특화되어 있다.

4. **단순성**: EGNN의 구현은 표준 MLP와 간단한 기하학적 연산만으로 이루어지며, DGL 같은 특수 라이브러리나 Wigner-D matrix, Clebsch-Gordan coefficient 같은 수학적 장치가 불필요하다.

코드 구조상으로도, EGNN의 핵심 로직은 수십 줄로 구현 가능한 반면, TFN/SE(3) Transformer는 `modules.py` (718줄), `fibers.py` (153줄), `representations.py` (250줄), `SO3.py` (290줄), `utils_steerable.py` (329줄) 등 수천 줄의 코드가 필요하다.
