# 08. Utility 함수 및 공유 패턴 분석

EGNN 코드베이스의 유틸리티 파일들과 데이터 준비 파이프라인, 그리고 코드 전반에서 반복적으로 나타나는 공유 패턴을 분석한다.

---

## 1. `utils.py` -- 최상위 유틸리티

**파일 경로:** `egnn/utils.py`

### 1.1 폴더 관리

```python
def create_folders(args):
    # args.outf / args.exp_name 하위에 images_recon, images_gen 디렉터리 생성
    os.makedirs(args.outf)                                    # top-level output
    os.makedirs(args.outf + '/' + args.exp_name)              # experiment dir
    os.makedirs(args.outf + '/' + args.exp_name + '/images_recon')
    os.makedirs(args.outf + '/' + args.exp_name + '/images_gen')

def makedir(path):
    os.makedirs(path)  # OSError silently ignored if exists
```

모든 `makedirs` 호출이 `try/except OSError: pass` 패턴을 사용한다. Python 3에서는 `os.makedirs(path, exist_ok=True)`가 더 안전하지만, 이 코드는 호환성 위주의 방어적 스타일이다.

`create_folders`는 autoencoder 실험(`main_ae.py`)에서 주로 사용되고, `makedir`은 `main_qm9.py`에서 사용된다:

```python
# main_qm9.py
utils.makedir(args.outf)
utils.makedir(args.outf + "/" + args.exp_name)
```

### 1.2 결과 정규화 (`normalize_res`)

```python
def normalize_res(res, keys=[]):
    for key in keys:
        if key != 'counter':
            res[key] = res[key] / res['counter']
    del res['counter']
    return res
```

학습/평가 루프에서 배치별로 누적한 결과를 epoch 단위 평균으로 변환하는 함수이다. 모든 training loop에서 `res` 딕셔너리에 `'counter'` 키로 누적 샘플 수를 기록하고, 이 함수로 나눈다. 실제 코드에서는 이 함수를 직접 호출하는 대신 `res['loss'] / res['counter']`로 인라인 계산하는 경우가 더 많다.

### 1.3 시각화 헬퍼 (`plot_coords`)

```python
def plot_coords(coords_mu, path, coords_logvar=None):
    # coords_logvar가 있으면 variance -> std 변환
    coords_std = torch.sqrt(torch.exp(coords_logvar))  # reparameterization: log(sigma^2) -> sigma
    # 2D scatter plot으로 좌표 시각화, path에 저장
```

VAE(Variational Autoencoder) 실험용 함수이다. `coords_logvar`가 주어지면 log-variance에서 standard deviation을 복원한다 (`sigma = sqrt(exp(log(sigma^2)))`). N-body나 QM9 실험에서는 사용되지 않는다.

### 1.4 Learning Rate 스케줄링 (`adjust_learning_rate`)

```python
def adjust_learning_rate(optimizer, epoch, lr_0, factor=0.5, epochs_decay=100):
    lr = lr_0 * (factor ** (epoch // epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

Step decay 방식의 LR 스케줄러이다. `epochs_decay` epoch마다 `factor`를 곱한다. 예를 들어 기본값에서 epoch 0-99는 `lr_0`, epoch 100-199는 `lr_0 * 0.5`, epoch 200-299는 `lr_0 * 0.25`가 된다.

실제 `main_qm9.py`에서는 이 함수 대신 PyTorch 내장 `CosineAnnealingLR`을 사용한다:

```python
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
```

`main_nbody.py`에서는 별도의 LR 스케줄러를 사용하지 않고 고정 learning rate(`5e-4`)를 사용한다.

### 1.5 노드 필터링 (`filter_nodes`)

```python
def filter_nodes(dataset, n_nodes):
    new_graphs = [g for g in dataset.graphs if len(g.nodes) == n_nodes]
    dataset.graphs = new_graphs
    dataset.n_nodes = n_nodes
    return dataset
```

그래프 오토인코더 실험에서 특정 노드 수를 가진 그래프만 필터링한다. `graph.py`의 `Graph` 객체를 사용하는 `ae_datasets` 모듈과 연관된 함수이다.

---

## 2. QM9 데이터 준비 파이프라인

QM9 데이터 로딩은 여러 파일에 걸쳐 계층적으로 이루어진다. 전체 흐름을 따라가 보자.

### 2.1 진입점: `qm9/dataset.py` -- `retrieve_dataloaders`

```python
def retrieve_dataloaders(batch_size, num_workers=1):
    args = init_argparse('qm9')         # (1) 기본 인자 설정
    args, datasets, num_species, charge_scale = initialize_datasets(
        args, args.datadir, 'qm9', ...)  # (2) 데이터 다운로드/로드
    # (3) 단위 변환: Hartree -> eV
    qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, ...}
    for dataset in datasets.values():
        dataset.convert_units(qm9_to_eV)
    # (4) DataLoader 구성 (collate_fn 사용)
    dataloaders = {split: DataLoader(dataset, collate_fn=collate_fn, ...) ...}
    return dataloaders, charge_scale
```

### 2.2 인자 설정: `qm9/args.py`

`args.py`는 원래 Cormorant (SE(3)-equivariant neural network) 프레임워크에서 가져온 파일이다. EGNN 프로젝트에서 실제로 사용되는 기본값:

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--datadir` | `'qm9/temp'` | 데이터 저장 경로 |
| `--num-train` | `-1` (전체) | 학습 샘플 수 |
| `--subtract-thermo` | `True` | 열화학 에너지 차감 |
| `--force-download` | `False` | 강제 재다운로드 |
| `--shuffle` | `True` | 데이터 셔플 |
| `--seed` | `1` | 랜덤 시드 |

`BoolArg` 클래스: argparse에서 `--flag True` / `--flag False` 형태의 문자열 boolean 인자를 처리하기 위한 커스텀 Action이다. `nargs='?'`로 설정하여 값 없이 `--flag`만 쓰면 기본값의 반전(True)이 된다.

`init_argparse('qm9')`: `parser.parse_args([])`로 빈 인자를 파싱하여 모든 기본값이 적용된 Namespace를 반환한다. 커맨드라인 인자가 아닌 프로그래매틱 호출용이다.

### 2.3 데이터 초기화: `qm9/data/utils.py` -- `initialize_datasets`

이 함수가 데이터 준비의 핵심 조율 역할을 한다:

```
1. prepare_dataset() 호출 -> 다운로드/처리된 .npz 파일 경로 반환
2. np.load()로 각 split(train/valid/test) 로드 -> torch.Tensor 변환
3. _get_species(): 전체 데이터셋의 고유 원자 종류(species) 추출
4. ProcessedDataset 생성: one-hot encoding, thermochemical energy 차감
5. num_species, max_charge 반환 -> 모델 초기화에 사용
```

`_get_species` 함수는 모든 split에 동일한 원자 종류가 포함되어 있는지 검증한다. zero-padding된 원자(charge=0)는 제거된다.

### 2.4 다운로드 및 전처리: `qm9/data/prepare/` 디렉터리

#### `download.py` -- `prepare_dataset` (오케스트레이터)

```
datadir/dataset/[subset/]{train,valid,test}.npz
```

형식의 파일이 존재하는지 확인하고, 없으면 `download_dataset_qm9` 또는 `download_dataset_md17`을 호출한다. 부분적으로만 처리된 경우(일부 split만 존재) `ValueError`를 발생시킨다.

#### `qm9.py` -- QM9 다운로드/처리

```
1. Figshare에서 dsgdb9nsd.xyz.tar.bz2 다운로드
2. gen_splits_gdb9(): 3,054개 불량 분자 제외, 나머지를 100k/N_valid/N_test로 분할
   - np.random.seed(0) 고정 -> 재현 가능한 분할
   - Train: 100,000 / Test: ~13,083 (10%) / Valid: 나머지
3. process_xyz_files()로 .xyz 파싱 -> {charges, positions, ...} 딕셔너리
4. get_thermo_dict(): atomref.txt에서 원자별 열화학 에너지 로드
5. add_thermo_targets(): 각 분자의 target에서 구성 원자의 열화학 에너지 합을 빼는 보정값 추가
```

열화학 에너지 보정 (`subtract_thermo`) 의미: QM9의 에너지 target은 절대값인데, 실제로 의미 있는 것은 분자 간 상대 에너지 차이이다. 각 원자 종류의 기본 에너지를 빼면 모델이 "분자 구조에 의한 에너지 차이"만 학습하게 된다.

#### `md17.py` -- MD17 다운로드/처리

```
1. quantum-machine.org에서 분자별 .npz 다운로드
2. 키 매핑: 'E' -> 'energies', 'R' -> 'positions', 'F' -> 'forces'
3. charges = np.tile(z, (num_mols, 1))  -- 모든 snapshot에 동일 원자 종류
4. gen_splits_md17(): 50k train / 10k valid / 10k test
   - 비연속적 인덱싱: [0:10k] train, [10k:20k] valid, [20k:30k] test, [30k:70k] train
```

#### `process.py` -- 파일 파싱

`process_xyz_files`: tarball이나 디렉터리에서 .xyz 파일들을 읽고, `process_file_fn` (예: `process_xyz_gdb9`)을 적용한다. 결과는 list-of-dicts에서 dict-of-lists로 변환되고, `pad_sequence`로 패딩하여 동일 크기 텐서로 스택된다.

`process_xyz_gdb9`: GDB9 xyz 형식 파싱. `'*^'` 표기를 `'e'` (지수)로 변환하는 처리가 포함된다. 17개 분자 property 중 16개를 파싱한다 (tag 제외).

`process_xyz_md17`: MD17 xyz 형식 파싱. 에너지와 force를 세미콜론으로 구분된 헤더에서 추출한다.

#### `utils.py` (prepare 하위) -- 공통 유틸리티

```python
def download_data(url, outfile='', binary=False):
    # urlopen으로 데이터 다운로드, 파일 저장
def is_int(str):
    # 문자열이 정수인지 체크 (excluded indices 파싱용)
def cleanup_file(file, cleanup=True):
    # 임시 파일 삭제
```

### 2.5 `ProcessedDataset` (`qm9/data/dataset.py`)

PyTorch `Dataset`을 확장한 클래스이다:

```python
class ProcessedDataset(Dataset):
    def __init__(self, data, included_species, num_pts, subtract_thermo, ...):
        # one-hot encoding 생성
        self.data['one_hot'] = (charges.unsqueeze(-1) == included_species.unsqueeze(0).unsqueeze(0))
        # thermochemical energy 차감
        for key in thermo_targets:
            data[key] -= data[key + '_thermo']
        # 셔플을 위한 permutation 인덱스
        self.perm = torch.randperm(len(data['charges']))[:num_pts]
```

주요 특징:
- Broadcasting을 이용한 one-hot 인코딩: `charges [N, max_atoms, 1] == species [1, 1, n_species]`
- 통계 자동 계산: 1D floating-point 텐서에 대해 mean/std 계산
- 단위 변환 메서드: `convert_units(dict)` -- Hartree를 eV로 변환하는 데 사용

### 2.6 Collation (`qm9/data/collate.py`)

```python
def collate_fn(batch):
    # 1. batch_stack: pad_sequence로 가변 크기 분자를 최대 크기로 패딩
    batch = {prop: batch_stack([mol[prop] for mol in batch]) ...}
    # 2. drop_zeros: 배치 내 모든 분자에서 빈(charge=0) 원자 열 제거
    to_keep = (batch['charges'].sum(0) > 0)
    batch = {key: drop_zeros(prop, to_keep) ...}
    # 3. atom_mask: 실제 원자 위치 마스크
    atom_mask = batch['charges'] > 0
    # 4. edge_mask: atom_mask의 외적 + 대각선 제거
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0)
    edge_mask *= diag_mask
    # -> shape: [batch_size * n_nodes * n_nodes, 1]
```

QM9에서는 분자마다 원자 수가 다르므로, 가장 큰 분자에 맞춰 zero-padding하고 mask로 유효 원자/에지를 표시한다. 이것은 N-body의 고정 크기 접근법과 대조적이다.

---

## 3. N-body 데이터 로딩 및 후처리

### 3.1 `n_body_system/dataloader.py` -- 커스텀 Dataloader

```python
class Dataloader():
    def expand_edges(self, edges, batch_size, n_nodes):
        # Single graph edges를 batch_size만큼 복제, 오프셋 추가
        for i in range(batch_size):
            rows.append(edges[0] + n_nodes * i)
            cols.append(edges[1] + n_nodes * i)
```

이 커스텀 dataloader는 현재 코드에서 사용되지 않는다 (deprecated). `main_nbody.py`에서는 PyTorch `DataLoader`와 `NBodyDataset.get_edges()`를 대신 사용한다.

### 3.2 `n_body_system/dataset_nbody.py` -- 데이터셋

Edge 구성 방식:

```python
def preprocess(self, loc, vel, edges, charges):
    rows, cols = [], []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:                    # <-- self-loop 제외
                edge_attr.append(edges[:, i, j])
                rows.append(i)
                cols.append(j)
```

`get_edges()` 메서드가 배치 단위 에지 확장을 담당한다 (아래 "공유 패턴" 참고).

### 3.3 `n_body_system/post_process.py` -- 결과 시각화

GNN, EGNN, baseline 세 모델의 학습 loss를 비교하는 단순한 plot 스크립트이다. 하드코딩된 loss 값 배열이 포함되어 있으며, `draw_result` 함수가 matplotlib으로 비교 그래프를 그린다. 논문의 Figure 또는 실험 결과 보고용으로 추정된다.

---

## 4. 코드베이스 공유 패턴

### 4.1 Edge Index 구성: Fully Connected (Self-loop 제외)

코드베이스 전체에서 에지를 구성하는 방법이 두 가지로 나뉜다:

**패턴 A: Self-loop 제외 (N-body, graph.py)**

```python
# n_body_system/dataset_nbody.py, graph.py의 _dense2attributes
for i in range(n_nodes):
    for j in range(n_nodes):
        if i != j:
            rows.append(i)
            cols.append(j)
# -> n_nodes * (n_nodes - 1) edges
```

N-body 시뮬레이션에서는 자기 자신과의 상호작용이 물리적으로 무의미하므로 self-loop를 제외한다.

**패턴 B: Self-loop 포함 (QM9)**

```python
# qm9/utils.py -- get_adj_matrix
for i in range(n_nodes):
    for j in range(n_nodes):
        rows.append(i + batch_idx * n_nodes)
        cols.append(j + batch_idx * n_nodes)
# -> n_nodes * n_nodes edges (self-loop 포함)
```

QM9에서는 self-loop를 포함하되, `edge_mask`로 유효하지 않은 에지를 0으로 마스킹한다. `collate.py`의 `collate_fn`에서 대각선을 마스킹하므로 실제 메시지 패싱에서는 self-loop가 차단된다:

```python
diag_mask = ~torch.eye(n_nodes, dtype=torch.bool).unsqueeze(0)
edge_mask *= diag_mask  # diagonal = 0 -> self-loop masked out
```

결론적으로, 두 방식 모두 self-loop 없는 fully connected 그래프를 의도한다. QM9에서는 가변 크기 분자를 다루기 위해 mask 기반 접근을 사용할 뿐이다.

### 4.2 배칭 (Batching): 노드 평탄화 + 에지 오프셋

EGNN 코드에서 그래프 배칭은 PyG(PyTorch Geometric) 스타일이 아니라 **수동 평탄화(flattening)**를 사용한다.

```
Graph 1: nodes [0,1,2], edges: 0->1, 1->0, 0->2, ...
Graph 2: nodes [0,1,2], edges: 0->1, 1->0, 0->2, ...

Batched (batch_size=2, n_nodes=3):
  nodes: [0,1,2,3,4,5]  (flattened)
  edges: [0->1, 1->0, ..., 3->4, 4->3, ...]  (offset by n_nodes*i)
```

이 패턴은 세 곳에서 동일하게 구현된다:

```python
# (1) NBodyDataset.get_edges()
for i in range(batch_size):
    rows.append(edges[0] + n_nodes * i)
    cols.append(edges[1] + n_nodes * i)

# (2) Dataloader.expand_edges() -- deprecated but same logic
for i in range(batch_size):
    rows.append(edges[0] + n_nodes * i)
    cols.append(edges[1] + n_nodes * i)

# (3) qm9/utils.py -- get_adj_matrix()
for batch_idx in range(batch_size):
    for i in range(n_nodes):
        for j in range(n_nodes):
            rows.append(i + batch_idx * n_nodes)
            cols.append(j + batch_idx * n_nodes)
```

핵심 아이디어: `n_nodes * batch_idx` 오프셋을 더하면, 여러 그래프가 하나의 큰 disconnected graph로 합쳐진다. 각 그래프의 에지가 자기 그래프의 노드만 참조하도록 보장된다.

데이터 텐서의 reshape도 일관된다:

```python
# main_nbody.py
data = [d.view(-1, d.size(2)) for d in data]
# [batch_size, n_nodes, features] -> [batch_size * n_nodes, features]

# main_qm9.py
atom_positions = data['positions'].view(batch_size * n_nodes, -1)
nodes = nodes.view(batch_size * n_nodes, -1)
```

**QM9의 추가 캐싱 최적화:**

```python
edges_dic = {}  # module-level cache
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        if batch_size in edges_dic[n_nodes]:
            return edges_dic[n_nodes][batch_size]  # cache hit
    # ... compute and store
```

`edges_dic`은 `{n_nodes: {batch_size: edges}}` 형태의 이중 딕셔너리로, 동일한 (n_nodes, batch_size) 조합에 대해 에지 인덱스를 재계산하지 않는다.

### 4.3 공통 Training Loop 패턴

두 main 스크립트(`main_nbody.py`, `main_qm9.py`)가 거의 동일한 학습 루프 구조를 공유한다:

```python
# Shared training loop skeleton
res = {'loss': 0, 'counter': 0, ...}

for epoch in range(args.epochs):
    # --- Train ---
    train(epoch, loader_train)

    # --- Validate/Test at interval ---
    if epoch % args.test_interval == 0:
        val_loss = train(epoch, loader_val, backprop=False)
        test_loss = train(epoch, loader_test, backprop=False)

        # --- Best model tracking ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_test_loss = test_loss
            best_epoch = epoch

    # --- Results logging ---
    json.dumps(results) -> outf/exp_name/losess.json
```

공통 요소:
1. **`res` 딕셔너리 패턴**: `{'loss': 0, 'counter': 0}`으로 초기화, 배치마다 `res['loss'] += loss.item() * batch_size`로 누적, `res['loss'] / res['counter']`로 평균
2. **`backprop` 플래그**: `train()` 함수 하나로 학습/평가 모두 처리. `backprop=True`이면 `model.train()` + `loss.backward()`, `False`이면 `model.eval()`
3. **Best model selection**: Validation loss 기준으로 best test loss를 기록 (early stopping 없이 best tracking만)
4. **JSON 로깅**: 매 epoch 결과를 `losess.json`에 저장 (파일명 오타 "losess"가 양쪽 모두에 존재)

차이점:

| | N-body (`main_nbody.py`) | QM9 (`main_qm9.py`) |
|---|---|---|
| Loss 함수 | `nn.MSELoss` | `nn.L1Loss` (MAE) |
| LR 스케줄링 | 없음 (고정 `5e-4`) | `CosineAnnealingLR` |
| 정규화 | 없음 | `(label - mean) / mad` |
| Edge 구성 | `dataset.get_edges()` | `qm9_utils.get_adj_matrix()` |
| Batch size | 100 | 96 |
| Hidden dim | 64 | 128 |
| Layers | 4 | 7 |

QM9에서의 label 정규화 패턴이 주목할 만하다:

```python
# Training: normalized target으로 loss 계산
loss = loss_l1(pred, (label - meann) / mad)

# Evaluation: prediction을 원래 스케일로 복원 후 loss 계산
loss = loss_l1(mad * pred + meann, label)
```

`mad` (Mean Absolute Deviation)로 정규화하면 서로 다른 스케일의 property를 학습할 때 loss가 비교 가능해진다.

---

## 5. Import 의존성 그래프

```
main_qm9.py
  |-- qm9/dataset.py (retrieve_dataloaders)
  |     |-- qm9/args.py (init_argparse, BoolArg)
  |     |-- qm9/data/utils.py (initialize_datasets)
  |     |     |-- qm9/data/dataset.py (ProcessedDataset)
  |     |     |-- qm9/data/prepare/__init__.py (prepare_dataset)
  |     |           |-- qm9/data/prepare/download.py (prepare_dataset)
  |     |           |     |-- qm9/data/prepare/qm9.py (download_dataset_qm9)
  |     |           |     |-- qm9/data/prepare/md17.py (download_dataset_md17)
  |     |           |-- qm9/data/prepare/process.py (process_xyz_files, process_xyz_gdb9)
  |     |           |-- qm9/data/prepare/utils.py (download_data, is_int, cleanup_file)
  |     |-- qm9/data/collate.py (collate_fn)
  |-- qm9/models.py (EGNN)
  |-- qm9/utils.py (compute_mean_mad, get_adj_matrix, preprocess_input)
  |-- utils.py (makedir)

main_nbody.py
  |-- n_body_system/dataset_nbody.py (NBodyDataset)
  |-- n_body_system/model.py (GNN, EGNN, EGNN_vel, Baseline, Linear, ...)
  |-- utils.py (create_folders) -- 간접 사용 (os.makedirs 인라인)

main_ae.py
  |-- ae_datasets/ (dataloader, d_creator, d_selector)
  |-- models/ (gcl, ae)
  |-- graph.py (Graph, networkx2graph, ...)
  |-- utils.py (create_folders, makedir, normalize_res, plot_coords, filter_nodes)
  |-- losess.py
  |-- eval.py

n_body_system/dataloader.py (deprecated -- not imported by main_nbody.py)
n_body_system/post_process.py (standalone visualization script)
```

### 의존성 관계 요약

- **`utils.py` (최상위)**: `main_qm9.py`와 `main_ae.py`에서 사용. `main_nbody.py`는 동일 기능을 인라인으로 구현
- **`qm9/` 패키지**: Cormorant 프레임워크에서 가져온 데이터 파이프라인. `args.py`의 대부분 옵션은 EGNN에서 사용하지 않음 (CG levels, spherical harmonics 등은 Cormorant 전용)
- **`n_body_system/`**: 자체 데이터셋/모델 구현. QM9 파이프라인과 독립적
- **`graph.py`**: autoencoder 실험 전용. N-body/QM9에서는 사용하지 않음

### 코드 중복 (주목할 점)

1. `qm9/dataset.py`의 `batch_stack`, `drop_zeros`가 `qm9/data/collate.py`에 복사되어 있음
2. 에지 확장 로직이 `NBodyDataset.get_edges()`와 `Dataloader.expand_edges()`에 중복
3. 디렉터리 생성이 `utils.create_folders`, `utils.makedir`, 인라인 `os.makedirs`로 세 가지 방식 공존
4. `charge_dict = {'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9}`가 `process.py`와 `qm9.py`에 중복 정의

---

## 6. 핵심 정리

| 구성 요소 | 역할 | 핵심 파일 |
|-----------|------|-----------|
| 폴더/로깅 관리 | 실험 디렉터리 생성, JSON 결과 저장 | `utils.py`, `main_*.py` |
| QM9 데이터 파이프라인 | 다운로드 -> xyz 파싱 -> split -> npz 저장 -> Dataset/DataLoader | `qm9/data/prepare/*.py`, `qm9/data/utils.py` |
| N-body 데이터 로딩 | numpy 로드 -> 텐서 변환 -> edge 구성 | `n_body_system/dataset_nbody.py` |
| Edge 구성 | Fully connected, self-loop 제외 (mask 또는 조건문) | `qm9/utils.py`, `dataset_nbody.py`, `collate.py` |
| 배칭 | 노드 평탄화 + edge offset | `get_edges()`, `get_adj_matrix()` |
| Training loop | res 딕셔너리 누적, backprop 플래그, val 기준 best tracking | `main_nbody.py`, `main_qm9.py` |

이 코드베이스는 연구 프로토타입의 특성을 잘 보여준다. 세 가지 실험(N-body, QM9, Autoencoder)이 공통 EGNN 모델을 공유하면서도, 데이터 로딩과 학습 루프는 각각 독립적으로 구현되어 있다. Cormorant에서 가져온 QM9 파이프라인의 많은 옵션이 EGNN에서는 사용되지 않지만 그대로 유지되어 있어, 학술 코드의 점진적 발전 과정을 보여준다.
