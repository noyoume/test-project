# 01. EGNN 이론 분석

> 논문: "E(n) Equivariant Graph Neural Networks" (Satorras, Hoogeboom, Welling, ICML 2021)
> arXiv:2102.09844v3

---

## 1. 핵심 동기: 왜 EGNN인가?

기존 E(3)/SE(3) equivariant 모델들(TFN, SE3-Transformer)의 문제점:
- **Spherical harmonics 계산이 비쌈**: 중간 레이어에서 higher-order representation이 필요
- **3차원에 제한**: 구면 조화 함수(spherical harmonics)가 3D에 특화되어 있어 고차원 확장이 어려움
- **구현 복잡도**: Wigner-D 행렬, Clebsch-Gordan 계수 등 복잡한 수학적 도구 필요

EGNN의 해결책:
- Higher-order representation **없이** E(n) equivariance 달성
- 상대 거리(squared distance)만으로 기하학적 정보 전달
- n차원으로 자연스럽게 확장 가능
- 단순한 MLP 기반 구현

---

## 2. Equivariance 정의 (Eq. 1)

함수 φ: X → Y가 군(group) G에 대해 **equivariant**하다는 것은:

```
φ(Tg(x)) = Sg(φ(x))    (Eq. 1)
```

- Tg: 입력 공간 X에서의 변환
- Sg: 출력 공간 Y에서의 대응되는 변환
- 입력을 변환한 후 함수를 적용한 결과 = 함수를 적용한 후 출력을 변환한 결과

**Invariance**는 equivariance의 특수한 경우로, Sg가 항등 변환인 경우:
```
φ(Tg(x)) = φ(x)
```

### EGNN이 보존하는 세 가지 equivariance

| # | 변환 | 수식 | 설명 |
|---|------|------|------|
| 1 | Translation | y + g = φ(x + g) | 입력을 g만큼 이동하면 출력도 g만큼 이동 |
| 2 | Rotation/Reflection | Qy = φ(Qx) | 직교 행렬 Q로 회전/반사하면 출력도 동일하게 변환 |
| 3 | Permutation | P(y) = φ(P(x)) | 노드 순서를 바꾸면 출력 순서도 동일하게 변경 |

**중요**: 좌표 x는 equivariant (변환에 따라 같이 변함), 노드 임베딩 h는 invariant (변환에 영향 없음)

---

## 3. 표준 GNN (Eq. 2) — 비교 기준

```
mij = φe(h_i^l, h_j^l, a_ij)        # edge operation
mi  = Σ_{j∈N(i)} mij                 # aggregation
h_i^{l+1} = φh(h_i^l, mi)           # node update
```

- h_i^l: 노드 i의 l번째 레이어 임베딩 (nf차원)
- a_ij: edge attribute
- φe, φh: MLP로 근사
- **한계**: 좌표 정보가 없고, 회전/이동에 대한 equivariance가 보장되지 않음

---

## 4. EGCL (Equivariant Graph Convolutional Layer) — 핵심 수식

### 입출력 정의
```
h^{l+1}, x^{l+1} = EGCL[h^l, x^l, E]
```
- 입력: 노드 임베딩 h^l, 좌표 x^l, 엣지 정보 E
- 출력: 업데이트된 h^{l+1}, x^{l+1}

### Eq. 3: Edge Operation (메시지 생성)
```
mij = φe(h_i^l, h_j^l, ||x_i^l - x_j^l||², a_ij)
```

**표준 GNN과의 차이**: 상대 **제곱 거리** `||x_i - x_j||²`를 입력에 추가
- 왜 제곱 거리인가? → 제곱 거리는 E(n) invariant:
  - 이동 불변: `||x_i + g - (x_j + g)||² = ||x_i - x_j||²`
  - 회전 불변: `||Qx_i - Qx_j||² = (x_i-x_j)^T Q^T Q (x_i-x_j) = ||x_i - x_j||²`
- φe는 MLP로 구현
- 출력 mij는 E(n) **invariant** (스칼라 값)

### Eq. 4: Coordinate Update (좌표 업데이트)
```
x_i^{l+1} = x_i^l + C · Σ_{j≠i} (x_i^l - x_j^l) · φx(mij)
```

**이것이 EGNN의 핵심 혁신**:
- `(x_i - x_j)`: 상대 방향 벡터 (radial direction)
- `φx(mij)`: mij를 입력받아 **스칼라** 가중치를 출력하는 MLP (R^nf → R^1)
- C = 1/(M-1): 합을 원소 수로 나누는 정규화 상수
- 각 입자의 위치를 **방사 방향 벡터장(radial vector field)**으로 업데이트

**왜 equivariant한가?** (Appendix A 증명 요약):
```
Translation: x_i+g + C·Σ((x_i+g)-(x_j+g))·φx(m) = x_i+g + C·Σ(x_i-x_j)·φx(m) = x_i^{l+1} + g ✓
Rotation:    Qx_i + C·Σ(Qx_i-Qx_j)·φx(m) = Qx_i + Q·C·Σ(x_i-x_j)·φx(m) = Q·x_i^{l+1} ✓
```
- φx(mij)가 스칼라이므로 Q를 밖으로 뺄 수 있다는 것이 핵심

### Eq. 5: Aggregation (메시지 집계)
```
mi = Σ_{j≠i} mij
```
- 표준 GNN과 동일
- 모든 이웃(또는 j≠i)으로부터의 메시지를 합산

### Eq. 6: Node Update (노드 임베딩 업데이트)
```
h_i^{l+1} = φh(h_i^l, mi)
```
- 표준 GNN과 동일
- h는 mij에만 의존하고, mij는 invariant → h^{l+1}도 invariant

---

## 5. Velocity 확장 (Eq. 7) — Section 3.2

입자의 운동량(momentum)을 명시적으로 추적하는 변형:

```
v_i^{l+1} = φv(h_i^l) · v_i^{init} + C · Σ_{j≠i} (x_i^l - x_j^l) · φx(mij)    (Eq. 7)
x_i^{l+1} = x_i^l + v_i^{l+1}
```

**Eq. 4와의 관계**:
- Eq. 4를 두 단계로 분해: 먼저 속도 계산 → 속도로 위치 업데이트
- `φv(h_i^l)`: 노드 임베딩을 스칼라로 매핑 (R^N → R^1)
- `v_i^{init}`: 초기 속도 (시스템에서 주어짐)
- v_init = 0이면 Eq. 4와 완전히 동일

**Equivariance 보존**:
- v는 좌표의 차이(방향 벡터)이므로 이동에 불변, 회전에 equivariant
- φv가 스칼라를 출력하므로 Q를 밖으로 뺄 수 있음

---

## 6. Edge Inference (Eq. 8) — Section 3.3

인접 행렬이 주어지지 않은 경우, 엣지를 추론하는 방법:

```
mi = Σ_{j≠i} eij · mij    (Eq. 8)
```

- `eij ≈ φ_inf(mij)`: attention과 유사한 soft edge estimation
- φ_inf: Linear + Sigmoid (R^nf → [0,1])
- mij가 이미 E(n) invariant이므로 eij도 invariant → 전체 equivariance 유지

---

## 7. Decoder for Graph Autoencoder (Eq. 9) — Section 5.2

그래프 오토인코더의 디코더:

```
Â_ij = 1 / (1 + exp(w · ||z_i - z_j||² + b))    (Eq. 9)
```

- z: 인코더의 출력 좌표 임베딩
- w, b: 학습 가능한 파라미터 (단 2개)
- 노드 간 상대 거리 기반 → E(n) invariant

---

## 8. 다른 방법론과의 비교 (Table 1)

| 방법 | Edge Operation | Equivariance | Spherical Harmonics |
|------|---------------|-------------|-------------------|
| GNN | φe(h_i, h_j, a_ij) | Permutation only | 불필요 |
| Radial Field | φ_rf(‖r_ij‖) · r_ij | E(n) | 불필요 |
| TFN | Σ_k W^{lk}(r_ji) h_i^k | SE(3) | **필요** |
| SchNet | φ_cf(‖r_ij‖) · φ_s(h_j) | E(n) invariant | 불필요 |
| **EGNN** | φe(h_i, h_j, ‖r_ij‖², a_ij) | **E(n) equivariant** | **불필요** |

**EGNN의 차별점**:
1. GNN의 유연성 (h를 전파) + Radial Field의 E(n) equivariance
2. 좌표 x와 임베딩 h가 edge operation에서 정보를 교환
3. Spherical harmonics 없이 equivariance 달성 → 계산 효율적, n차원 확장 가능

---

## 9. Equivariance 증명 요약 (Appendix A)

**증명 전략**: 귀납법으로 EGCL의 각 수식을 순서대로 검증

### Step 1: 전제
h^0이 E(n) invariant하다고 가정 (절대 위치/방향 정보를 인코딩하지 않음)

### Step 2: Eq. 3 (Edge) — invariant
```
mij = φe(h_i, h_j, ||Qx_i+g - (Qx_j+g)||², aij)
    = φe(h_i, h_j, ||Q(x_i-x_j)||², aij)
    = φe(h_i, h_j, (x_i-x_j)^T Q^T Q (x_i-x_j), aij)
    = φe(h_i, h_j, ||x_i-x_j||², aij)
```
→ mij는 E(n) **invariant**

### Step 3: Eq. 4 (Coord) — equivariant
```
Qx_i + g + C·Σ(Qx_i+g - Qx_j-g)·φx(mij)
= Qx_i + g + QC·Σ(x_i-x_j)·φx(mij)
= Q(x_i + C·Σ(x_i-x_j)·φx(mij)) + g
= Q·x_i^{l+1} + g
```
→ x^{l+1}은 E(n) **equivariant**

### Step 4: Eq. 5, 6 (Agg, Node) — invariant
- mi는 mij의 합 → invariant
- h^{l+1} = φh(h_i, mi) → invariant인 것들의 함수 → **invariant**

### 귀납 완성
EGCL 하나가 equivariance를 보존하므로, EGCL의 합성(여러 레이어)도 equivariant.

---

## 10. 실험 설계 개요

| 실험 | 태스크 | 출력 성질 | 사용 변형 |
|------|--------|----------|----------|
| N-body (5.1) | 입자 위치 예측 | Equivariant (좌표) | EGNN + velocity (Eq. 7) |
| Graph AE (5.2) | 인접 행렬 복원 | Invariant (엣지 유무) | EGNN + noise input |
| QM9 (5.3) | 분자 성질 예측 | Invariant (스칼라) | EGNN + edge inference (Eq. 8) |

---

## 11. 핵심 인사이트 요약

1. **"좌표 업데이트를 방사 방향으로만 수행"**이라는 단순한 아이디어가 E(n) equivariance를 보장
2. φx가 **스칼라**를 출력하기 때문에 회전 행렬 Q를 밖으로 뺄 수 있음 — 이것이 수학적 핵심
3. Higher-order representation이 불필요한 이유: 위치 정보만 있을 때 상대 거리가 기하학을 완전히 정의 (Appendix E)
4. h는 invariant, x는 equivariant로 역할이 명확히 분리됨
5. 구현 관점에서 "E_GCL = 기존 GCL + coord_model" 정도의 추가로 equivariance 획득
