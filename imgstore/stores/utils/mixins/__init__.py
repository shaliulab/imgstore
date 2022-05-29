from imgstore.stores.utils.mixins.cv2 import CV2Mixin
from .cm import ContextManagerMixin
from .cv2 import CV2Mixin
from .extra_data import ExtraDataMixin
from .extract import ExtractMixin
from .get import GetMixin
from .index import IndexMixin


MIXINS = [
    ContextManagerMixin,
    CV2Mixin,
    ExtraDataMixin,
    ExtractMixin,
    GetMixin,
    IndexMixin,
]