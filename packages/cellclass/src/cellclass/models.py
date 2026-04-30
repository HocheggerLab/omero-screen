# ------------------------------------------------------------------------------
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.
# ------------------------------------------------------------------------------

"""Neural network model definitions for cell image classification."""

from __future__ import annotations

from collections.abc import Iterator
from enum import Enum
from typing import Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision import models


class Model(Enum):
    """Supported classification model architectures."""

    def __new__(cls, *args: object, **kwds: object) -> Model:
        """Create enum member, storing the optional size as a second argument."""
        obj = object.__new__(cls)
        obj._value_ = args[0]
        return obj

    def __init__(self, _: str, size: int = 224) -> None:
        """Store expected input image size alongside the enum value."""
        self._size_ = size

    densenet121 = "densenet121"
    densenet161 = "densenet161"
    densenet169 = "densenet169"
    densenet201 = "densenet201"
    squeezenet1_0 = "squeezenet1_0"
    shufflenet2x1_0 = "shufflenet2x1_0"
    shufflenet2x1_5 = "shufflenet2x1_5"
    shufflenet2x2_0 = "shufflenet2x2_0"
    efficientnetb3 = "efficientnetb3", 300
    efficientnetb4 = "efficientnetb4", 380
    efficientnetb3s = "efficientnetb3s", 300
    efficientnetb4s = "efficientnetb4s", 380
    resenext50_32x4d = "resenext50_32x4d"
    alexnet = "alexnet"
    alexnet_att = "alexnet_att"
    googlenet = "googlenet"
    googlenet_att = "googlenet_att"
    simplegooglenet = "simplegooglenet"
    simplegooglenet_att = "simplegooglenet_att"
    simplegooglenet_ch_att = "simplegooglenet_ch_att"
    simplegooglenet_ch_sp_att = "simplegooglenet_ch_sp_att"

    def __str__(self) -> str:
        """Return the string value of the enum member."""
        return str(self.value)

    @property
    def size(self) -> int:
        """Return the expected square input image size in pixels."""
        return int(self._size_)  # type: ignore[attr-defined]


def create_model(
    model: Union[Model, str],
    num_classes: int,
    num_channels: int = 1,
    weights: str | None = None,
    freeze_weights: bool = False,
    dropout: float = 0.5,
) -> nn.Module | None:
    """Create a classification model for cell image data.

    Args:
        model: Model architecture enum value or string name.
        num_classes: Number of output classes.
        num_channels: Number of input image channels.
        weights: Pretrained weights identifier (e.g. ``'DEFAULT'``).
        freeze_weights: Freeze pretrained weights; only new layers are trained.
        dropout: Dropout probability for the final classification layer.

    Returns:
        Instantiated ``nn.Module``, or ``None`` if the model is not recognised.

    """
    if isinstance(model, str):
        model = Model(model)

    mapping: dict[Model, nn.Module] = {
        Model.densenet121: ROIBasedDenseNetModel121(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.densenet161: ROIBasedDenseNetModel161(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.densenet169: ROIBasedDenseNetModel169(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.densenet201: ROIBasedDenseNetModel201(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.squeezenet1_0: ROIBasedSqueezeNetModel1_0(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.shufflenet2x1_0: ROIBasedShuffleNetModel2x1_0(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.shufflenet2x1_5: ROIBasedShuffleNetModel2x1_5(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.shufflenet2x2_0: ROIBasedShuffleNetModel2x2_0(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.efficientnetb3: ROIBasedEfficientNetB3(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.efficientnetb4: ROIBasedEfficientNetB4(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.efficientnetb3s: ROIBasedEfficientNetB3s(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            freeze_weights=freeze_weights,
            dropout=dropout,
        ),
        Model.efficientnetb4s: ROIBasedEfficientNetB4s(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            freeze_weights=freeze_weights,
            dropout=dropout,
        ),
        Model.resenext50_32x4d: ROIBasedResNeXt50_32x4d(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.alexnet: CustomAlexNet(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.alexnet_att: AlexNetAttention(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.googlenet: CustomGoogLeNet(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.googlenet_att: GoogLeNetAttention(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.simplegooglenet: SimpleGoogLeNet(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.simplegooglenet_att: SimpleGoogLeNetAttention(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.simplegooglenet_ch_att: SimpleGoogLeNetChannelAttention(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
        Model.simplegooglenet_ch_sp_att: SimpleGoogLeNetAttentionBoth(
            num_classes,
            num_channels=num_channels,
            weights=weights,
            dropout=dropout,
        ),
    }
    return mapping.get(model)


class ROIBasedDenseNetModel121(nn.Module):  # type: ignore[misc]
    """ROIBased Dense Net Model121 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.densenet121(weights=weights)
        self.roi_model.features.conv0 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.roi_model.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_features_roi = self.roi_model.classifier.in_features
        self.roi_model.classifier = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model.features(roi)
        roi_features = torch.flatten(roi_features, 1)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedDenseNetModel161(nn.Module):  # type: ignore[misc]
    """ROIBased Dense Net Model161 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.densenet161(weights=weights)
        self.roi_model.features.conv0 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=96,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.roi_model.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_features_roi = self.roi_model.classifier.in_features
        self.roi_model.classifier = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model.features(roi)
        roi_features = torch.flatten(roi_features, 1)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedDenseNetModel169(nn.Module):  # type: ignore[misc]
    """ROIBased Dense Net Model169 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.densenet169(weights=weights)
        self.roi_model.features.conv0 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.roi_model.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_features_roi = self.roi_model.classifier.in_features
        self.roi_model.classifier = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model.features(roi)
        roi_features = torch.flatten(roi_features, 1)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedDenseNetModel201(nn.Module):  # type: ignore[misc]
    """ROIBased Dense Net Model201 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.densenet201(weights=weights)
        self.roi_model.features.conv0 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.roi_model.features.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        num_features_roi = self.roi_model.classifier.in_features
        self.roi_model.classifier = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model.features(roi)
        roi_features = torch.flatten(roi_features, 1)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedSqueezeNetModel1_0(nn.Module):  # type: ignore[misc]
    """ROIBased Squeeze Net Model1_0 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.squeezenet1_0(weights=weights)
        self.roi_model.features[0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=96,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.roi_model.classifier = nn.Identity()
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model.features(roi)
        roi_features = self.global_avg_pool(roi_features)
        roi_features = torch.flatten(roi_features, 1)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedShuffleNetModel2x1_0(nn.Module):  # type: ignore[misc]
    """ROIBased Shuffle Net Model2x1_0 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.shufflenet_v2_x1_0(weights=weights)
        self.roi_model.conv1[0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        num_features_roi = self.roi_model.fc.in_features
        self.roi_model.fc = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model(roi)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedShuffleNetModel2x1_5(nn.Module):  # type: ignore[misc]
    """ROIBased Shuffle Net Model2x1_5 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.shufflenet_v2_x1_5(weights=weights)
        self.roi_model.conv1[0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        num_features_roi = self.roi_model.fc.in_features
        self.roi_model.fc = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model(roi)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedShuffleNetModel2x2_0(nn.Module):  # type: ignore[misc]
    """ROIBased Shuffle Net Model2x2_0 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.shufflenet_v2_x2_0(weights=weights)
        self.roi_model.conv1[0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=24,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        num_features_roi = self.roi_model.fc.in_features
        self.roi_model.fc = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model(roi)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedEfficientNetB3(nn.Module):  # type: ignore[misc]
    """ROIBased Efficient Net B3 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.3,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.efficientnet_b3(weights=weights)
        self.roi_model.features[0][0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=40,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        num_features_roi = self.roi_model.classifier[1].in_features
        self.roi_model.classifier = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model(roi)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedEfficientNetB4(nn.Module):  # type: ignore[misc]
    """ROIBased Efficient Net B4 classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.4,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.efficientnet_b4(weights=weights)
        self.roi_model.features[0][0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=48,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        num_features_roi = self.roi_model.classifier[1].in_features
        self.roi_model.classifier = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model(roi)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class ROIBasedEfficientNetB3s(nn.Module):  # type: ignore[misc]
    """ROIBased Efficient Net B3s classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        freeze_weights: bool = False,
        dropout: float = 0.3,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self._freeze_weights = freeze_weights
        self.roi_model = m = models.efficientnet_b3(weights=weights)
        if freeze_weights:
            for param in m.parameters():
                param.requires_grad = False
        m.features[0][0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=40,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        m.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        num_features_roi = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(num_features_roi, num_classes)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        return self.roi_model(roi)  # type: ignore[no-any-return]

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield trainable parameters, respecting freeze_weights setting."""
        if self._freeze_weights:
            m = self.roi_model
            yield from m.features[0][0].parameters(recurse=recurse)
            yield from m.classifier[1].parameters(recurse=recurse)
            return
        yield from super().parameters(recurse=recurse)


class ROIBasedEfficientNetB4s(nn.Module):  # type: ignore[misc]
    """ROIBased Efficient Net B4s classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        freeze_weights: bool = False,
        dropout: float = 0.4,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self._freeze_weights = freeze_weights
        self.roi_model = m = models.efficientnet_b4(weights=weights)
        if freeze_weights:
            for param in m.parameters():
                param.requires_grad = False
        m.features[0][0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=48,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        m.classifier[0] = nn.Dropout(p=dropout, inplace=True)
        num_features_roi = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(num_features_roi, num_classes)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        return self.roi_model(roi)  # type: ignore[no-any-return]

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Yield trainable parameters, respecting freeze_weights setting."""
        if self._freeze_weights:
            m = self.roi_model
            yield from m.features[0][0].parameters(recurse=recurse)
            yield from m.classifier[1].parameters(recurse=recurse)
            return
        yield from super().parameters(recurse=recurse)


class ROIBasedResNeXt50_32x4d(nn.Module):  # type: ignore[misc]
    """ROIBased Res Ne Xt50_32x4d classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = models.resnext50_32x4d(weights=weights)
        self.roi_model.conv1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        num_features_roi = self.roi_model.fc.in_features
        self.roi_model.fc = nn.Identity()
        self.fc1 = nn.Linear(num_features_roi, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, roi: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        roi_features = self.roi_model(roi)
        roi_features = torch.flatten(roi_features, 1)
        x = torch.relu(self.fc1(roi_features))
        return self.fc2(self.dropout(x))  # type: ignore[no-any-return]


class AttentionModule(nn.Module):  # type: ignore[misc]
    """Spatial attention module combining two feature maps."""

    def __init__(
        self,
        in_channels_l: int,
        in_channels_m: int,
        out_channels: int = 256,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.convL = nn.Conv2d(in_channels_l, out_channels, kernel_size=1)
        self.convM = nn.Conv2d(in_channels_m, out_channels, kernel_size=1)
        self.attention_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(
        self, feature_l: Tensor, feature_m: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute attended feature fusion from two feature maps."""
        F_l = F.interpolate(
            self.convL(feature_l),
            size=feature_m.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        F_m = self.convM(feature_m)
        F_combined = self.attention_conv(F.relu(F_l + F_m))
        attention_map = torch.sigmoid(F_combined)
        weighted_features = attention_map * feature_m
        return weighted_features, attention_map


class ChannelAttentionModule(nn.Module):  # type: ignore[misc]
    """Channel attention module."""

    def __init__(self, num_channels: int = 256) -> None:
        """Initialise the model layers."""
        super().__init__()
        reduction_ratio = 16
        self.mlp = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // reduction_ratio, num_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        x_avg = torch.mean(x, dim=(2, 3), keepdim=True)
        x_max = torch.amax(x, dim=(2, 3), keepdim=True)
        x_avg = x_avg.view(x.size(0), -1)
        x_max = x_max.view(x.size(0), -1)
        attention = torch.sigmoid(self.mlp(x_avg) + self.mlp(x_max))
        attention = attention.view(attention.size(0), -1, 1, 1)
        return attention * x  # type: ignore[no-any-return]


class CustomAlexNet(nn.Module):  # type: ignore[misc]
    """Custom Alex Net classifier."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.roi_model = m = models.alexnet(weights=weights)
        m.features[0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=11,
            stride=4,
            padding=2,
        )
        m.classifier[0] = nn.Dropout(p=dropout)
        m.classifier[3] = nn.Dropout(p=dropout)
        num_features_roi = m.classifier[6].in_features
        m.classifier[6] = nn.Linear(num_features_roi, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        return self.roi_model(x)  # type: ignore[no-any-return]


class AlexNetAttention(nn.Module):  # type: ignore[misc]
    """AlexNet with spatial attention."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.features = models.alexnet(weights=weights).features
        self.features[0] = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=11,
            stride=4,
            padding=2,
        )
        self.attention_1 = AttentionModule(
            in_channels_l=256, in_channels_m=384, out_channels=256
        )
        self.attention_2 = AttentionModule(
            in_channels_l=256, in_channels_m=256, out_channels=256
        )
        # Note: input (H,W) must be divisible by (6,6) on MPS — CPU/CUDA only.
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear((384 + 256 + 256) * 6 * 6, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        layer_5 = self.features[:8](x)
        layer_6 = self.features[8:10](layer_5)
        layer_l = self.features[10:](layer_6)
        att_1, _ = self.attention_1(layer_l, layer_5)
        att_2, _ = self.attention_2(layer_l, layer_6)
        x = torch.cat(
            (self.avgpool(att_1), self.avgpool(att_2), self.avgpool(layer_l)),
            dim=1,
        )
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)  # type: ignore[no-any-return]


class CustomGoogLeNet(nn.Module):  # type: ignore[misc]
    """GoogLeNet without auxiliary classifiers."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.2,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.googlenet = m = models.googlenet(weights=weights)
        m.conv1.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        m.dropout = nn.Dropout(p=dropout)
        m.fc = nn.Linear(m.fc.in_features, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        g = self.googlenet
        x = g.conv1(x)
        x = g.maxpool1(x)
        x = g.conv2(x)
        x = g.conv3(x)
        x = g.maxpool2(x)
        x = g.inception3a(x)
        x = g.inception3b(x)
        x = g.maxpool3(x)
        x = g.inception4a(x)
        x = g.inception4b(x)
        x = g.inception4c(x)
        x = g.inception4d(x)
        x = g.inception4e(x)
        x = g.maxpool4(x)
        x = g.inception5a(x)
        x = g.inception5b(x)
        x = g.avgpool(x)
        x = torch.flatten(x, 1)
        x = g.dropout(x)
        return g.fc(x)  # type: ignore[no-any-return]


class GoogLeNetAttention(nn.Module):  # type: ignore[misc]
    """GoogLeNet with spatial attention at inception4a and inception4e."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        self.googlenet = m = models.googlenet(weights=weights)
        m.conv1.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.attention_1 = AttentionModule(
            in_channels_l=1024, in_channels_m=512, out_channels=256
        )
        self.attention_2 = AttentionModule(
            in_channels_l=1024, in_channels_m=832, out_channels=256
        )
        m.dropout = nn.Dropout(p=dropout)
        m.fc = nn.Linear(512 + 832 + 1024, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        g = self.googlenet
        x = g.conv1(x)
        x = g.maxpool1(x)
        x = g.conv2(x)
        x = g.conv3(x)
        x = g.maxpool2(x)
        x = g.inception3a(x)
        x = g.inception3b(x)
        x = g.maxpool3(x)
        layer_4a = g.inception4a(x)
        x = g.inception4b(layer_4a)
        x = g.inception4c(x)
        x = g.inception4d(x)
        layer_4e = g.inception4e(x)
        x = g.maxpool4(layer_4e)
        x = g.inception5a(x)
        layer_l = g.inception5b(x)
        att_1, _ = self.attention_1(layer_l, layer_4a)
        att_2, _ = self.attention_2(layer_l, layer_4e)
        x = torch.cat(
            (g.avgpool(att_1), g.avgpool(att_2), g.avgpool(layer_l)), dim=1
        )
        x = torch.flatten(x, 1)
        x = g.dropout(x)
        return g.fc(x)  # type: ignore[no-any-return]


class SimpleGoogLeNet(nn.Module):  # type: ignore[misc]
    """Lightweight GoogLeNet using a subset of inception blocks."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.2,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        m = models.googlenet(weights=weights)
        self.conv1 = m.conv1
        self.maxpool1 = m.maxpool1
        self.conv2 = m.conv2
        self.conv3 = m.conv3
        self.maxpool2 = m.maxpool2
        self.inception3a = m.inception3a
        self.inception3b = m.inception3b
        self.maxpool3 = m.maxpool3
        self.inception4a = m.inception4a
        self.inception4d = m.inception4d
        self.inception4e = m.inception4e
        self.maxpool4 = m.maxpool4
        self.avgpool = m.avgpool
        self.conv1.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(832, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)  # type: ignore[no-any-return]


class SimpleGoogLeNetAttention(nn.Module):  # type: ignore[misc]
    """Lightweight GoogLeNet with spatial attention."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        m = models.googlenet(weights=weights)
        self.conv1 = m.conv1
        self.maxpool1 = m.maxpool1
        self.conv2 = m.conv2
        self.conv3 = m.conv3
        self.maxpool2 = m.maxpool2
        self.inception3a = m.inception3a
        self.inception3b = m.inception3b
        self.maxpool3 = m.maxpool3
        self.inception4a = m.inception4a
        self.inception4d = m.inception4d
        self.inception4e = m.inception4e
        self.maxpool4 = m.maxpool4
        self.avgpool = m.avgpool
        self.conv1.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(480 + 512 + 832, num_classes)
        self.attention_1 = AttentionModule(
            in_channels_l=832, in_channels_m=480, out_channels=256
        )
        self.attention_2 = AttentionModule(
            in_channels_l=832, in_channels_m=512, out_channels=256
        )

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Union[Tensor, tuple[Tensor, Tensor, Tensor]]:
        """Run the forward pass, optionally returning attention maps."""
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        layer_3b = self.inception3b(x)
        x = self.maxpool3(layer_3b)
        layer_4a = self.inception4a(x)
        x = self.inception4d(layer_4a)
        layer_l = self.inception4e(x)
        x = self.maxpool4(layer_l)
        att_1, map_1 = self.attention_1(layer_l, layer_3b)
        att_2, map_2 = self.attention_2(layer_l, layer_4a)
        x = torch.cat(
            (self.avgpool(att_1), self.avgpool(att_2), self.avgpool(layer_l)),
            dim=1,
        )
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if return_attention:
            return x, map_1, map_2
        return x


class SimpleGoogLeNetChannelAttention(nn.Module):  # type: ignore[misc]
    """Lightweight GoogLeNet with channel attention."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        m = models.googlenet(weights=weights)
        self.conv1 = m.conv1
        self.maxpool1 = m.maxpool1
        self.conv2 = m.conv2
        self.conv3 = m.conv3
        self.maxpool2 = m.maxpool2
        self.inception3a = m.inception3a
        self.inception3b = m.inception3b
        self.maxpool3 = m.maxpool3
        self.inception4a = m.inception4a
        self.inception4d = m.inception4d
        self.inception4e = m.inception4e
        self.maxpool4 = m.maxpool4
        self.avgpool = m.avgpool
        self.conv1.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(480 + 512 + 832, num_classes)
        self.ch_attention_1 = ChannelAttentionModule(num_channels=480)
        self.ch_attention_2 = ChannelAttentionModule(num_channels=512)

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        layer_3b = self.inception3b(x)
        x = self.maxpool3(layer_3b)
        layer_4a = self.inception4a(x)
        x = self.inception4d(layer_4a)
        layer_l = self.inception4e(x)
        x = self.maxpool4(layer_l)
        ch_att_1 = self.ch_attention_1(layer_3b)
        ch_att_2 = self.ch_attention_2(layer_4a)
        x = torch.cat(
            (
                self.avgpool(ch_att_1),
                self.avgpool(ch_att_2),
                self.avgpool(layer_l),
            ),
            dim=1,
        )
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)  # type: ignore[no-any-return]


class SimpleGoogLeNetAttentionBoth(nn.Module):  # type: ignore[misc]
    """Lightweight GoogLeNet with both channel and spatial attention."""

    def __init__(
        self,
        num_classes: int,
        num_channels: int = 1,
        weights: str | None = None,
        dropout: float = 0.5,
    ) -> None:
        """Initialise the model layers."""
        super().__init__()
        m = models.googlenet(weights=weights)
        self.conv1 = m.conv1
        self.maxpool1 = m.maxpool1
        self.conv2 = m.conv2
        self.conv3 = m.conv3
        self.maxpool2 = m.maxpool2
        self.inception3a = m.inception3a
        self.inception3b = m.inception3b
        self.maxpool3 = m.maxpool3
        self.inception4a = m.inception4a
        self.inception4d = m.inception4d
        self.inception4e = m.inception4e
        self.maxpool4 = m.maxpool4
        self.avgpool = m.avgpool
        self.conv1.conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(480 + 512 + 832, num_classes)
        self.ch_attention_1 = ChannelAttentionModule(num_channels=480)
        self.ch_attention_2 = ChannelAttentionModule(num_channels=512)
        self.attention_1 = AttentionModule(
            in_channels_l=832, in_channels_m=480, out_channels=256
        )
        self.attention_2 = AttentionModule(
            in_channels_l=832, in_channels_m=512, out_channels=256
        )

    def forward(self, x: Tensor) -> Tensor:
        """Run the forward pass and return logits."""
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        layer_3b = self.inception3b(x)
        x = self.maxpool3(layer_3b)
        layer_4a = self.inception4a(x)
        x = self.inception4d(layer_4a)
        layer_l = self.inception4e(x)
        x = self.maxpool4(layer_l)
        ch_att_1 = self.ch_attention_1(layer_3b)
        ch_att_2 = self.ch_attention_2(layer_4a)
        att_1, _ = self.attention_1(layer_l, ch_att_1)
        att_2, _ = self.attention_2(layer_l, ch_att_2)
        x = torch.cat(
            (self.avgpool(att_1), self.avgpool(att_2), self.avgpool(layer_l)),
            dim=1,
        )
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)  # type: ignore[no-any-return]
