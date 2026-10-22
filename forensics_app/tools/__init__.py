"""Register course functionality here so it appears in the sidebar."""

from .channel_split import ChannelSplitTool
from .canny import CannyTool, EdgeVisualizationTool
from .channel_swap import ChannelSwapTool
from .contrast_stretch import ContrastStretchTool
from .convolution import ConvolutionTool
from .corners import CornerDetectionTool
from .grayscale import GrayscaleTool
from .filters import GaussianTool, MedianTool, SobelTool, SharpenTool
from .histogram import HistogramTool
from .hog import HogTool
from .image_info import ImageInfoTool
from .keypoints import KeypointDescriptorTool
from .masking import MaskingTool
from .person_detection import PersonDetectionTool
from .registry import ToolRegistry
from .template_matching import CrossCorrelationTool, TemplateMatchingTool


def build_tool_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            ImageInfoTool(),
            GrayscaleTool(),
            ChannelSplitTool(),
            ChannelSwapTool(),
            MaskingTool(),
            HistogramTool(),
            ContrastStretchTool(),
            ConvolutionTool(),
            GaussianTool(),
            MedianTool(),
            SobelTool(),
            CannyTool(),
            EdgeVisualizationTool(),
            SharpenTool(),
            CrossCorrelationTool(),
            TemplateMatchingTool(),
            HogTool(),
            PersonDetectionTool(),
            CornerDetectionTool(),
            KeypointDescriptorTool(),
        ]
    )


__all__ = ["ToolRegistry", "build_tool_registry"]
