# jonal

画像分類データセットの読み込み、正規化、再現可能なラベルノイズ付与を行います。

```python
from dataset import load_dataset

train_dataset, evaluation_dataset = load_dataset(
    "cifar10",
    data_augmentation={
        "random_crop": {"padding": 4},
        "horizontal_flip": {"p": 0.5},
        "random_rotation": {"degrees": 15},
        "color_jitter": {"brightness": 0.2, "contrast": 0.2},
        "random_erasing": {"p": 0.25},
    },
    label_noise_rate=0.2,
    seed=42,
)
```

データ拡張の各辞書は、対応する`torchvision.transforms`へ渡す数値などの引数です。短縮形も利用できます。

```python
data_augmentation={
    "random_rotation": 15,  # degrees=15
    "horizontal_flip": 0.5,  # p=0.5
}
```

`data_augmentation=True`では従来どおり、CIFARに`random_crop + horizontal_flip`、EMNIST-digitsに`random_affine`を適用します。`False`または`None`では追加の拡張を行いません。

指定可能な種類: `random_crop`、`horizontal_flip`、`vertical_flip`、`random_rotation`、`random_affine`、`color_jitter`、`random_perspective`、`random_resized_crop`、`random_grayscale`、`gaussian_blur`、`random_erasing`。

- 対応データセット: `cifar10`、`cifar100`、`emnist-digits`
- データ拡張は訓練データだけに適用されます。
- 正規化は設定不要で、訓練・評価データの両方に必ず適用されます。
- EMNIST-digitsは訓練・評価ともに32×32へリサイズし、リサイズ後の訓練画像から
  算出した平均 `0.173469787916794`、標準偏差 `0.316553955125998` で
  正規化します。
- ノイズ率 `r` では `int(len(train_dataset) * r)` 件を選び、元とは異なるクラスへ等確率で変更します。
- ノイズ付与後の全ラベルは、例として `data/cifar10/cifar10_labels_noise-0p2_seed-42.npz` に保存され、次回は検証後に再利用されます。
- 評価データのラベルは変更されません。

## 実験の実行

`train.py` では、実験条件をコマンドラインから指定できます。引数名は
`--batch-size` と `--batch_size` のどちらの形式でも指定できます。

```bash
python3 train.py \
  --model resnet18 \
  --dataset cifar10 \
  --lr 0.1 \
  --batch-size 128 \
  --gpu 0 \
  --num-workers 4 \
  --augmentation '{"random_crop":{"padding":4},"horizontal_flip":{"p":0.5}}' \
  --label-noise 0.2 \
  --optimizer sgd \
  --loss crossentropy \
  --momentum 0.9 \
  --model-width 1.0 \
  --input-channels 3 \
  --seed 42 \
  --epochs 100 \
  --checkpoint-interval 1 \
  --wandb-mode offline
```

`--augmentation` には `none`、`default`、カンマ区切りの変換名、または
JSONの辞書・配列を指定できます。`--gpu auto` はCUDAが利用可能ならGPU 0、
それ以外はCPUを選択します。`--experiment-name` を省略すると、主要な条件から

`--loss` は `crossentropy`（既定値）、`mse`、`mae` から選択できます。
`mse` と `mae` はsoftmax確率とone-hotラベルのクラス方向平均を、サンプルごとの
損失として計算します。
実験名が自動生成されます。

結果は `results/<experiment_name>/` 以下へ保存されます。

- `config.json`: 実験条件
- `metrics.csv`: 各エポックの8評価項目
- `checkpoint/epoch_<epoch>.pt`: 指定エポックのモデル・optimizer状態
  （例: `checkpoint/epoch_10.pt`）

`--checkpoint-interval N` でNエポックごとに保存します。既定値は `1`（毎エポック）
で、間隔に一致しない場合でも最終エポックは保存します。`0` では保存しません。

`train.bash` からは環境変数で保存間隔とW&B設定を変更できます。

```bash
# 既定: 毎エポック保存、W&Bはローカルへのoffline記録
bash train.bash

# 5エポックごとに保存し、W&Bへオンライン同期
CHECKPOINT_INTERVAL=5 \
WANDB_MODE=online \
WANDB_PROJECT=jonal \
WANDB_ENTITY=my-team \
bash train.bash
```

W&Bには各エポックの8指標と実験設定を記録し、ソースコード保存は無効にしています。
既存結果の意図しない上書きを避けるため、同名の結果ディレクトリがある場合は
エラーになります。

訓練データの各要素は
`(image, original_label, training_label, is_noisy)` です。
`is_noisy` は、学習用ラベルが元ラベルから実際に変化した場合だけ `True` です。
全訓練指標と noisy/clean の部分集合指標は学習用ラベルに対して計算され、
該当サンプルが0件の部分集合は `NaN` として記録されます。

## 補間実験（空間・時間的不安定性）

`train.py` が保存したチェックポイントを使い、2サンプル `x0`, `x1` を結ぶ直線
`x(α) = (1 − α)x0 + αx1`（α = −0.5〜1.5、刻み0.01、201点）上で推論します。
チェックポイントは `--checkpoint-interval` で保存されたものをすべて使い、
`checkpoint/epoch_0.pt` があれば初期モデルも含めます。

```bash
# 3モードの推論と定量化・描画をまとめて実行
RUN_DIR=results/<experiment_name> NUM_PAIRS=1000 GPU=0 PHASE_EPOCHS='30,60' \
  bash interpolation.bash

# 個別に実行する場合
python3 interpolation.py --run-dir results/<experiment_name> --mode clean-noisy --num-pairs 1000
python3 interpolation_metrics.py --mode-dir interpolation/<experiment_name>/clean-noisy
```

ペアの作り方（距離は正規化後の入力空間でのL2、データ拡張なし）:

- `clean-clean`: cleanサンプルを `x0` とし、同じラベルのclean最近傍を `x1`
- `clean-noisy`: noisyサンプルを `x0`（クエリ）とし、その元ラベルと同じclean最近傍を `x1`
- `test-test`: テストサンプルを `x0` とし、同じラベルのテスト最近傍を `x1`

クエリは `--pair-seed` で再現可能に重複なく選びます。出力構成:

```
interpolation/<experiment_name>/<mode>/
├── config.json            # 実行条件・α・エポック一覧
├── pairs.json             # 全ペアのインデックス・ラベル・距離
├── pair_0000/log.npz      # 1ペアの全結果
├── ...
└── metrics/
    ├── spatial_instability.csv / spatial_instability_per_pair.npz   # INST_S
    ├── temporal_instability.csv / temporal_instability.npz          # INST_T [epoch, α]
    ├── occupancy.csv                                                # original/noisy/other占有率
    └── figures/{spatial_instability,temporal_instability_heatmap,occupancy}.pdf
```

`log.npz` には `probs`（[エポック, 201, クラス] のsoftmax確率、float32）、
`preds`（[エポック, 201] の予測ラベル）、`epochs`、`alphas`、`distance`、
`x0_*`/`x1_*`（`split`、`index`、`original_label`、`training_label`、`is_noisy`）を保存します。
完了済みの `log.npz` があるペアは再実行時にスキップされます。
メモリ使用量は `--pairs-per-chunk`（既定100）で調整できます。
INST_T の「t−1」は直前に評価したチェックポイントで、最初のエポックは `NaN` です。
