import os
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import math
from scipy.interpolate import interp1d
from scipy.signal import resample
import shutil
import numpy as np
from scipy.stats import mode
path="D:\\hand"

train_df=[]
label_df=[]
val_train_df=[]
val_label_df=[]
Motion_columns=["Timestamp",
               "Acceleration_X","Acceleration_Y","Acceleration_Z",
               "Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
               "Magnetometer_X","Magnetometer_Y","Magnetometer_Z" ]


def resample_Motion_data(motion_df, motion_freq=100):
    """
    모션 데이터 (가속도계, 자이로, 자기장 등)를 100Hz로 리샘플링하는 함수.
    """
    df=pd.DataFrame()
    df["Label"]=motion_df["Label"]
    df["Timestamp"]=motion_df["Timestamp"].astype(float)
    motion_df["Timestamp"] = pd.to_datetime(motion_df["Timestamp"], unit="ms").astype("int64") / 1e9
    min_time, max_time = motion_df["Timestamp"].min(), motion_df["Timestamp"].max()
    new_time_motion = np.linspace(min_time, max_time, int((max_time - min_time) * motion_freq))


    # 보간을 위한 딕셔너리 생성
    resampled_motion = {}

    for col in Motion_columns:
        if col in motion_df.columns:  # 해당 센서 데이터가 존재하는지 확인
            interp_func = interp1d(motion_df["Timestamp"], motion_df[col], kind='linear', fill_value="extrapolate")
            resampled_motion[col] = interp_func(new_time_motion)

    # 데이터프레임 변환
    motion_resampled_df = pd.DataFrame(resampled_motion)
    motion_resampled_df["Timestamp"] = new_time_motion  # 시간 정보 추가


    df["Timestamp"] = pd.to_numeric(df["Timestamp"])
    motion_resampled_df["Timestamp"] = pd.to_numeric(motion_resampled_df["Timestamp"])

    motion_resampled_df = pd.merge_asof(motion_resampled_df, df, on="Timestamp", direction="nearest")
    return motion_resampled_df



for f in os.listdir(path): 
    motion_df=[]
    if "train_data" in f:
        # motion_df=np.load(os.path.join(path,f),allow_pickle=True)
        train_df=np.load(os.path.join(path,f),allow_pickle=True)
    
    elif "train_label" in f:
        # motion_df=np.load(os.path.join(path,f),allow_pickle=True)
        label_df=np.load(os.path.join(path,f),allow_pickle=True)
   
    elif "val_Hand_data" in f:
        # motion_df=np.load(os.path.join(path,f),allow_pickle=True)
        val_train_df=np.load(os.path.join(path,f),allow_pickle=True)

    elif "val_Hand_label" in f:
        # motion_df=np.load(os.path.join(path,f),allow_pickle=True)
        val_label_df=np.load(os.path.join(path,f),allow_pickle=True)
#################################################################        
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import shutil
from geopy.distance import geodesic

import h5py

from torch.utils.data import Dataset




from torchvision.models import densenet
import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor

from torchvision.transforms._presets import ImageClassification
from torchvision.utils import _log_api_usage_once
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._meta import _IMAGENET_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface




__all__ = [
    "DenseNet"
]


class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)


# class _Transition_gps(nn.Sequential):
#     def __init__(self, num_input_features: int, num_output_features: int) -> None:
#         super().__init__()
#         self.norm = nn.BatchNorm1d(num_input_features)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv = nn.Conv1d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
#         self.pool = nn.AvgPool1d(kernel_size=2, stride=2)


class Multi_DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate: int = 12,
        block_config = (3,3,3,3),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 8,
        memory_efficient: bool = False,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        """ features 1 """
        # First convolution
        self.features_1 = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv1d(3, num_init_features, kernel_size=(7,), stride=(3,), padding=3, bias=False)),
                    ("norm0", nn.BatchNorm1d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features_1.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features_1.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features_1.add_module("norm5", nn.BatchNorm1d(num_features))

        """ features_2 """
        # First convolution
        self.features_2 = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv1d(3, num_init_features, kernel_size=(7,), stride=(3,), padding=3, bias=False)),
                    ("norm0", nn.BatchNorm1d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features_2.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features_2.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features_2.add_module("norm5", nn.BatchNorm1d(num_features))


        """ features_3 """
        # First convolution
        self.features_3 = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv1d(3, num_init_features, kernel_size=(7,), stride=(3,), padding=3, bias=False)),
                    ("norm0", nn.BatchNorm1d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features_3.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features_3.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features_3.add_module("norm5", nn.BatchNorm1d(num_features))


        # Avg_pooling -> Customized
        self.avg_pool1d = nn.AvgPool1d(kernel_size=4, stride=4)

        # Linear layer -> Customized
        # self.classifier = nn.Linear(num_features*3, num_classes)
        # self.classifier = nn.Sequential(
        #     # nn.Linear(576, 64),
        #     nn.Linear(num_features*3*3, 64),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5),
        #     nn.Linear(128, num_classes)
        # )
        self.classifier=None
#         self.classifier = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(64, 128),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(128, num_classes)
# )

        self.num_classes=num_classes
        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        out1 = self.features_1(x[:, 0:3, :])
        out2 = self.features_2(x[:, 3:6, :])
        out3 = self.features_3(x[:, 6:9, :])

        # features = self.features(x)

        out1 = F.relu(out1, inplace=True)
        out2 = F.relu(out2, inplace=True)
        out3 = F.relu(out3, inplace=True)

        out1 = self.avg_pool1d(out1)
        out2 = self.avg_pool1d(out2)
        out3 = self.avg_pool1d(out3)

        out = torch.cat([out1, out2, out3], dim=1)
        # print("🔍 Flatten input shape:", out.shape)
        out = torch.flatten(out, 1)
        # out = self.classifier(out)  # 그대로 사용

        if self.classifier is None:
            input_dim = out.shape[1]
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_classes)
            )
        self.classifier.to(out.device)  # GPU/CPU 위치 맞추기

        out = self.classifier(out)
        return out













class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        growth_rate: int = 12,
        block_config = (3,3,3,3),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        num_classes: int = 8,
        memory_efficient: bool = False,
    ) -> None:

        super().__init__()
        _log_api_usage_once(self)

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv1d(3, num_init_features, kernel_size=(7,), stride=(2,), padding=3, bias=False)),
                    ("norm0", nn.BatchNorm1d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool1d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm1d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool1d(out, 1)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model: nn.Module, weights: WeightsEnum, progress: bool) -> None:
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = weights.get_state_dict(progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    growth_rate: int,
    block_config: Tuple[int, int, int, int],
    num_init_features: int,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> DenseNet:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)

    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=progress)

    return model


_COMMON_META = {
    "min_size": (29, 29),
    "categories": _IMAGENET_CATEGORIES,
    "recipe": "https://github.com/pytorch/vision/pull/116",
    "_docs": """These weights are ported from LuaTorch.""",
}

from torch.utils.data import DataLoader
                
                # train_df=torch.tensor(train_df,dtype=torch.float)
                # val_train_df=torch.tensor(val_train_df,dtype=torch.float)
                # # train_df = torch.from_numpy(train_df).float()
                # # val_train_df = torch.from_numpy(val_train_df).float()
                # # label_df = torch.from_numpy(label_df).long()  # 라벨은 long 타입 (CrossEntropyLoss용)
                # # val_label_df = torch.from_numpy(val_label_df).long()
                
                
                # acc_train=train_df[:,:, 1:4]
                # gyro_train= train_df[:,:, 4:7] 
                # mag_train=train_df[:, :,7:10] 
                # # gps_train=gps[:, :,]
                # y_train=label_df-1
                
                # acc_train = acc_train.permute(0, 2, 1)  # → (batch, channels, time)
                # gyro_train = gyro_train.permute(0, 2, 1)  # → (batch, channels, time)
                # mag_train = mag_train.permute(0, 2, 1)  # → (batch, channels, time)
                
                
                
                # acc_valid=val_train_df[:,:, 1:4]
                # gyro_valid= val_train_df[:,:, 4:7] 
                # mag_valid=val_train_df[:,:, 7:10]
                # # gps_train=gps[:, :,]
                # y_valid=val_label_df-1

train_df=torch.tensor(train_df,dtype=torch.float)
X=train_df[:,:,1:]
val_train_df=torch.tensor(val_train_df,dtype=torch.float)


X = []  # list of shape (9, time_len)
y = []




import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class MotionDataset(Dataset):
    def __init__(self, X, y):
        if isinstance(X, torch.Tensor):
            self.X = X.float()
        else:
            self.X = torch.tensor(X, dtype=torch.float32)

        if isinstance(y, torch.Tensor):
            self.y = y.long()
        else:
            self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 🔹 데이터 준비
X_train=train_df[:,:,1:]

X_val=val_train_df[:,:,1:]


# ✅ 1. 라벨 정수화 (-1 → 0~7 범위)
y_train = (label_df - 1).flatten().astype(int)
y_val = (val_label_df - 1).flatten().astype(int)


# ✅ 2. LabelEncoder (선택: 레이블이 이미 정수면 생략 가능)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)

# ✅ 3. 클래스 수 확인 후 모델 생성
num_classes = len(np.unique(y_train))
# model= Multi_DenseNet(num_classes=num_classes).to(device)


 
# ✅ 4. CrossEntropyLoss를 쓰는 경우:
# 반드시 y는 torch.long이고 값이 0~num_classes-1 범위여야 함

X_train= X_train.permute(0, 2, 1)  # → (batch, channels, time)
X_val = X_val.permute(0, 2, 1)  # → (batch, channels, time)
X_train.shape
X_val.shape



train_dataset = MotionDataset(X_train, y_train)
val_dataset = MotionDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


# 🔹 모델 & 훈련 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




print("Input shape:", X_train.shape)
print("Unique labels:", np.unique(y_train))
print("Label dtype:", y_train.dtype)


# model = Multi_DenseNet(num_classes=8).to("cpu")
# sample = X_train[0].unsqueeze(0)  # shape: (1, 9, 6000)
# output = model(sample)






model = Multi_DenseNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

# # 🔹 학습 + 검증 루프
# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
    
#     # 🔍 검증
#     model.eval()
#     val_preds = []
#     val_trues = []

#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             preds = torch.argmax(outputs, dim=1)

#             val_preds.extend(preds.cpu().numpy())
#             val_trues.extend(labels.cpu().numpy())

#     val_acc = accuracy_score(val_trues, val_preds)
#     print(f"📘 Epoch {epoch+1} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f}")

# # 🔍 최종 검증 결과
# print("\n📊 Final Validation Report:")
# print(classification_report(val_trues, val_preds, digits=4))
# print("Confusion Matrix:")
# print(confusion_matrix(val_trues, val_preds))

# model.eval()
val_preds = []
val_trues = []
all_probs = []
def evaluate_loader(model, data_loader, device, num_classes, tag="검증"):
    model.eval()
    all_preds = []
    all_trues = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_trues.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu())

    all_probs = torch.cat(all_probs, dim=0)
    avg_probs = torch.mean(all_probs, dim=0)

    acc = accuracy_score(all_trues, all_preds)
    print(f"\n✅ {tag} 정확도: {acc:.4f}")
    print(f"\n📊 {tag} Softmax 평균 (클래스별):")
    for i in range(num_classes):
        print(f"  클래스 {i}: {avg_probs[i].item():.4f}")

        return acc, avg_probs, all_preds, all_trues
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
        
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
        
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
        
            print(f"\n📘 Epoch {epoch+1} | Loss: {running_loss:.4f}")
        
            # 🔍 에폭마다 train/val 성능 확인
            evaluate_loader(model, train_loader, device, num_classes, tag="Train")
            evaluate_loader(model, val_loader, device, num_classes, tag="Validation")
        # _________________________________________________________________________________________

# train_acc_list = []
# val_acc_list = []

# for epoch in range(num_epochs):
#     model.train()
#     running_loss = 0.0

#     for inputs, labels in train_loader:
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()

#     print(f"\n📘 Epoch {epoch+1} | Loss: {running_loss:.4f}")

#     # 🔍 에폭마다 정확도 평가 및 저장
#     train_acc, _, _, _ = evaluate_loader(model, train_loader, device, num_classes, tag="Train")
#     val_acc, _, _, _ = evaluate_loader(model, val_loader, device, num_classes, tag="Validation")

#     train_acc_list.append(train_acc)
#     val_acc_list.append(val_acc)

# # ✅ 50 에폭 평균 정확도 출력
# mean_train_acc = np.mean(train_acc_list)
# mean_val_acc = np.mean(val_acc_list)

# print("\n📊 🔥 최종 50 Epoch 평균 정확도 🔥")
# print(f"✅ Train 평균 정확도: {mean_train_acc:.4f}")
# print(f"✅ Validation 평균 정확도: {mean_val_acc:.4f}")


best_val_acc = 0.0
best_model_path = "best_model.pth"

train_acc_list = []
val_acc_list = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"\n📘 Epoch {epoch+1} | Loss: {running_loss:.4f}")

    # 에폭마다 정확도 평가
    train_acc, _, _, _ = evaluate_loader(model, train_loader, device, num_classes, tag="Train")
    val_acc, _, _, _ = evaluate_loader(model, val_loader, device, num_classes, tag="Validation")

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    # 🔒 성능 좋은 모델 저장
    if val_acc > best_val_acc:
        best_val_acc = val_acc

    # ⚠️ 먼저 dummy input으로 forward → classifier 내부 생성
        dummy_input = torch.randn(1, 9, 6000).to(device)
        _ = model(dummy_input)

    # 이제 classifier 포함해서 state_dict 저장됨
        torch.save(model.state_dict(), best_model_path)
        print(f"📥 Best model updated at epoch {epoch+1}, val_acc: {val_acc:.4f}")

   
# 🔚 학습 종료 후 평균 정확도 출력
print("\n📊 🔥 최종 50 Epoch 평균 정확도 🔥")
print(f"✅ Train 평균 정확도: {np.mean(train_acc_list):.4f}")
print(f"✅ Validation 평균 정확도: {np.mean(val_acc_list):.4f}")


################ ________________________________________________________________________

#############################################################
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
def predict_with_sliding_window_softmax(model, acc_all, gyro_all, mag_all, window_size, stride, device):
    # model.eval()
    softmax_list = []
    center_timestamps = []
    # acc_all = acc_all.reshape(-1, acc_all.shape[2]).T  # → (3, T)
    acc_all = acc_all.permute(0, 2, 1)  # (B, T, C)
    acc_all = acc_all.reshape(-1, acc_all.shape[2])  # (B*T, C)
    acc_all = acc_all.T  # (C, total_len)

    # acc_all = acc_all.permute(0, 2, 1).reshape(-1, acc_all.shape[1]).T  # (10, T)
    
    gyro_all = gyro_all.permute(0, 2, 1)  # (B, T, C)
    gyro_all = gyro_all.reshape(-1, gyro_all.shape[2])  # (B*T, C)
    gyro_all = gyro_all.T  # (C, total_len)

    mag_all = mag_all.permute(0, 2, 1)  # (B, T, C)
    mag_all = mag_all.reshape(-1, mag_all.shape[2])  # (B*T, C)
    mag_all = mag_all.T  # (C, total_len)


    total_len = acc_all.shape[1] 
    print(f"🔍 total_len: {total_len}, window_size: {window_size}, stride: {stride}")

    for start in range(0, total_len - window_size + 1, stride):
        # acc_slice = torch.tensor(acc_all[:, start:start + window_size])[None, :, :]
        # gyro_slice = torch.tensor(gyro_all[:, start:start + window_size])[None, :, :]
        # mag_slice = torch.tensor(mag_all[:, start:start + window_size])[None, :, :]
        acc_slice = acc_all[:, start:start + window_size][None, :, :]
        # print(acc_slice.shape)
        gyro_slice = gyro_all[:, start:start + window_size][None, :, :]
        mag_slice = mag_all[:, start:start + window_size][None, :, :]


        with torch.no_grad():
            merged_slice = torch.cat([acc_slice, gyro_slice, mag_slice], dim=1)  # (1, 9, window_size)
            # print(merged_slice.shape)
            output = model(merged_slice.to(device))
            softmax = torch.nn.functional.softmax(output, dim=1)  # (1, 8)
            # print("output:", output)
            # print("softmax:", softmax)

        softmax_list.append(softmax.cpu().numpy().flatten())  # (8,)
        center_timestamps.append(start + window_size // 2)

    return np.array(softmax_list), np.array(center_timestamps), total_len


def time_voting_softmax(softmax_outputs, center_timestamps, window_size, total_time_len):
    vote_sum = np.zeros((total_time_len, softmax_outputs.shape[1]))
    vote_count = np.zeros((total_time_len,))

    half_win = window_size // 2

    for i, center in enumerate(center_timestamps):
      
        start = max(0, center - half_win)
        end = min(total_time_len, center + half_win)

        vote_sum[start:end] += softmax_outputs[i]
        vote_count[start:end] += 1

    nonzero = vote_count > 0
    vote_sum[nonzero] /= vote_count[nonzero, None]
    
    final_labels = np.argmax(vote_sum, axis=1)
    return final_labels

from collections import Counter

def trip_voting(final_labels, trip_ids):

    """
    각 trip 내에서 majority voting을 적용해 라벨을 통일함
    """
    output_labels = np.zeros_like(final_labels)

    unique_trips = np.unique(trip_ids)
    for trip in unique_trips:
        indices = np.where(trip_ids == trip)[0]
        trip_labels = final_labels[indices]

        majority_label = mode(trip_labels, keepdims=False).mode  # 최빈값
        output_labels[indices] = majority_label

    return output_labels



# =======================================================
import numpy as np
from scipy.stats import mode

def downsample_predictions(pred_labels, target_len):
    total_len = len(pred_labels)
    step = 6000 # 몇 개씩 묶을지 결정

    downsampled = []
    for i in range(0, total_len, step):
        chunk = pred_labels[i:i+step]
        # print(chunk.shape)
        if len(chunk) == 0:
            continue
        majority = mode(chunk, keepdims=True).mode[0]
        downsampled.append(majority)
   
    return np.array(downsampled[:target_len])  # 길이 맞춰 잘라줌
# _______________________________________________________
def expand_labels_to_timestamps(val, total_len, window_size=6000, stride=100):
    """
    y_valid: 원래 validation 라벨 (예: (2283,))
    total_len: timestamp 전체 길이 (예: 13,698,000)
    window_size: segment window size (기본 60초 * 100Hz = 6000)
    stride: 슬라이딩 간격 (기본 1초 * 100Hz = 100)
    """

    # 몇 번 반복할지 계산 (각 라벨을 얼마나 확장할지)
    step = 6000
    # print(step)

    expanded = np.repeat(val, step)
    # 혹시 너무 길면 잘라줌
    return expanded
# ____________________________

window_size = 6000  # 60초
stride = 100        # 1초
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Multi_DenseNet().to(device)

# dummy forward → classifier 만들어짐
dummy_input = torch.randn(1, 9, 6000).to(device)
_ = model(dummy_input)

# 이제 안전하게 불러오기 가능
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()



softmax_outputs, center_timestamps, total_time_len = predict_with_sliding_window_softmax(
    model, X_val[:,:3,:],X_val[:,3:6,:],X_val[:,6:,:], window_size, stride, device)

time_voting_val = time_voting_softmax(
    softmax_outputs, center_timestamps, window_size, total_time_len
)
y_valid=(val_label_df - 1).flatten().astype(int)

# 정답도 동일하게 이어붙여서 
true_labels=y_valid
expand_true_label = expand_labels_to_timestamps(true_labels, total_len=len(time_voting_val))
# print(len(expand_true_label))  # ✅ 13698000으로 맞춰짐



# true_labels = np.squeeze(y_valid).reshape(-1)
# final_pred_labels_ds = downsample_predictions(time_voting_val, len(true_labels))
# print(len(final_pred_labels_ds))  # ✅ 2283으로 맞춰짐


# len(trip_voting_val)
# 길이 맞춰서 평가

accuracy = accuracy_score(expand_true_label, time_voting_val)
print(f"✅ Time Voting Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# accuracy=accuracy_score(expand_true_label,trip_voting_val)
# print(f"✅ Trip Voting Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


# ___________________________####################################################
# import torch
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# 🔹 데이터 준비
trip_val=seg_hand
trip_x_val=trip_val[58:-58,1:10,:]
trip_y_val=trip_val[58:-58,10,:].flatten()

# ✅ 1. 라벨 정수화 (-1 → 0~7 범위)
trip_y_val = (trip_y_val- 1)


# ✅ 2. LabelEncoder (선택: 레이블이 이미 정수면 생략 가능)
from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)
# y_val = le.transform(y_val)

# ✅ 3. 클래스 수 확인 후 모델 생성
num_classes = len(np.unique(trip_y_val))
# model= Multi_DenseNet(num_classes=num_classes).to(device)


 
# ✅ 4. CrossEntropyLoss를 쓰는 경우:
# 반드시 y는 torch.long이고 값이 0~num_classes-1 범위여야 함

# trip_x_val= trip_x_val.permute(0, 2, 1)  # → (batch, channels, time)

softmax_outputs, center_timestamps, total_time_len = predict_with_sliding_window_softmax(
    model, trip_x_val[:,:3,:],trip_x_val[:,3:6,:],trip_x_val[:,6:,:], window_size, stride, device)

time_voting_val = time_voting_softmax(
    softmax_outputs, center_timestamps, window_size, total_time_len)

trip_ids=trip_val[58:-58,-1,:].flatten()


accuracy = accuracy_score(expand_true_label, time_voting_val)


final_pred_labels_ds=trip_voting(time_voting_val  , trip_ids)
# 정답도 동일하게 이어붙여서 
true_labels=trip_y_val
# expand_true_label = expand_labels_to_timestamps(true_labels, total_len=len(time_voting_val))
# print(len(expand_true_label))  # ✅ 13698000으로 맞춰짐


# trip_voting_val=expand_labels_to_timestamps(final_pred_labels_ds, total_len=len(expand_true_label))
accuracy=accuracy_score(true_labels, time_voting_val)
accuracy=accuracy_score(final_pred_labels_ds,trip_y_val)
print(f"✅ Trip Voting Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


##########################################################33333
def segment_trips(data):
    """
    타임스탬프 간격이 10ms 이상 발생할 경우 새로운 Trip으로 분할하는 함수
    """
    # 타임스탬프 차이 계산 (ms 단위)
    data["Time_diff"] = data["Timestamp"].diff()

    # 10ms 이상 간격이 발생하면 새로운 Trip 시작
    data["Trip_id"] = ((data["Time_diff"] > 10) | (data["Label"] != data["Label"].shift())).cumsum()

  

    # 필요 없는 열 제거
    data = data.drop(columns=["Time_diff"])

    return data

X_val_trip=val_train_df[:,:,:]


# ✅ 1. 라벨 정수화 (-1 → 0~7 범위)
y_val_trip = (val_label_df - 1).flatten().astype(int)


# model= Multi_DenseNet(num_classes=num_classes).to(device)


 
# ✅ 4. CrossEntropyLoss를 쓰는 경우:
# 반드시 y는 torch.long이고 값이 0~num_classes-1 범위여야 함

X_val_trip = X_val_trip.permute(0,2,1).reshape(-1,10)# → (batch, channels, time)
X_val_trip.shape

Motion_columns=["Timestamp",
               "Acceleration_X","Acceleration_Y","Acceleration_Z",
               "Gyroscope_X","Gyroscope_Y","Gyroscope_Z",
               "Magnetometer_X","Magnetometer_Y","Magnetometer_Z" ]

trip_df=pd.DataFrame(X_val_trip,columns=Motion_columns)
trip_df["Label"]=expand_true_label
trip_df=segment_trips(trip_df)
trip_labels=trip_df["Trip_id"]
# dummy forward → classifier 만들어짐
dummy_input = torch.randn(1, 9, 6000).to(device)
_ = model(dummy_input)

# 이제 안전하게 불러오기 가능
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()


X=X_val_trip.reshape(-1,10)


softmax_outputs, center_timestamps, total_time_len = predict_with_sliding_window_softmax(
    model, X_val[:,:3,:],X_val[:,3:6,:],X_val[:,6:,:], window_size, stride, device)

time_voting_val = time_voting_softmax(
    softmax_outputs, center_timestamps, window_size, total_time_len
)
y_valid=(val_label_df - 1).flatten().astype(int)

# 정답도 동일하게 이어붙여서 
true_labels=y_valid
expand_true_label = expand_labels_to_timestamps(true_labels, total_len=len(time_voting_val))
# print(len(expand_true_label))  # ✅ 13698000으로 맞춰짐



final_pred_labels_ds=trip_voting(time_voting_val , trip_ids)
# true_labels = np.squeeze(y_valid).reshape(-1)
# final_pred_labels_ds = downsample_predictions(time_voting_val, len(true_labels))
# print(len(final_pred_labels_ds))  # ✅ 2283으로 맞춰짐


# len(trip_voting_val)
# 길이 맞춰서 평가

accuracy = accuracy_score(expand_true_label, time_voting_val)
print(f"✅ Time Voting Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# accuracy=accuracy_score(expand_true_label,trip_voting_val)
# print(f"✅ Trip Voting Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

##########################################################################################


from torchviz import make_dot
import torch

# 모델 인스턴스
model = Multi_DenseNet(num_classes=8)  # 또는 DenseNet()
model.eval()

# 더미 입력 (예: batch_size=1, 채널=9, 시퀀스 길이=300)
dummy_input = torch.randn(1, 9, 6000)

# 출력 계산
output = model(dummy_input)

# 시각화
make_dot(output, params=dict(model.named_parameters())).render("MultiDenseNet_graph", format="png")






from torch.utils.tensorboard import SummaryWriter

model = Multi_DenseNet(num_classes=8)
dummy_input = torch.randn(1, 9, 300)

writer = SummaryWriter(log_dir="./runs/multidensenet")
writer.add_graph(model, dummy_input)
writer.close()




import matplotlib.pyplot as plt
import numpy as np

# 가상의 예측 윈도우들 (8-class softmax 예측 예시)
num_windows = 60
num_classes = 8

# timestamp t를 포함하는 윈도우들에서 나온 softmax 출력
softmax_outputs = np.random.dirichlet(np.ones(num_classes), size=num_windows)

# 평균 예측 (voting 결과)
avg_probs = np.mean(softmax_outputs, axis=0)

# 시각화
plt.figure(figsize=(10, 5))
plt.bar(range(num_classes), avg_probs)
plt.xlabel("Transportation Mode (Label Index)")
plt.ylabel("Average Softmax Probability")
plt.title("Time Voting: Aggregated Predictions for Timestamp t")
plt.xticks(range(num_classes), ['Still', 'Walk', 'Run', 'Bike', 'Car', 'Bus', 'Train', 'Subway'], rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()





