import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset


logger = logging.getLogger("dinov2")
VECTOR_LENGTH = 500

# _Target = int

# class _Split(Enum):
#     TRAIN = "train"
#     VAL = "val"
#     TEST = "test"  # NOTE: torchvision does not support the test split

#     @property
#     def length(self) -> int:
#         split_lengths = {
#             _Split.TRAIN: 34_745,
#             _Split.VAL: 3_923,
#             _Split.TEST: 10,
#         }
#         return split_lengths[self]

#     def get_dirname(self, class_id: Optional[str] = None) -> str:
#         return self.value if class_id is None else os.path.join(self.value, class_id)

#     def get_image_relpath(self, actual_index: int, class_id: Optional[str] = None) -> str:
#         dirname = self.get_dirname(class_id)
#         if self == _Split.TRAIN:
#             basename = f"{class_id}_{actual_index}"
#         else:  # self in (_Split.VAL, _Split.TEST):
#             basename = f"ILSVRC2012_{self.value}_{actual_index:08d}"
#         return os.path.join(dirname, basename + ".JPEG")

#     def parse_image_relpath(self, image_relpath: str) -> Tuple[str, int]:
#         assert self != _Split.TEST
#         dirname, filename = os.path.split(image_relpath)
#         class_id = os.path.split(dirname)[-1]
#         basename, _ = os.path.splitext(filename)
#         actual_index = int(basename.split("_")[-1])
#         return class_id, actual_index


class OCTA(ExtendedVisionDataset):
    # Target = Union[_Target]
    # Split = Union[_Split]
    Target = Union[np.ndarray, None]

    def __init__(
        self,
        *,
        # split: "ImageNet.Split",
        # split: "",
        root: str,
        extra: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self._extra_root = extra
        # self._split = split

        # self._entries = Optional[np.ndarray] = None
        self._entries = None
        # self._class_ids = None
        # self._class_names = None

        if not os.path.exists(self._get_extra_full_path(self._entries_path)):
            logger.info("Metadata cache not found – generating ...")
            self._dump_entries()

    # @property
    # def split(self) -> "ImageNet.Split":
    #     return self._split

    def _get_extra_full_path(self, extra_path: str) -> str:
        return os.path.join(self._extra_root, extra_path)

    def _load_extra(self, extra_path: str) -> np.ndarray:
        extra_full_path = self._get_extra_full_path(extra_path)
        return np.load(extra_full_path, mmap_mode="r")

    def _save_extra(self, extra_array: np.ndarray, extra_path: str) -> None:
        extra_full_path = self._get_extra_full_path(extra_path)
        os.makedirs(self._extra_root, exist_ok=True)
        np.save(extra_full_path, extra_array)

    @property
    def _entries_path(self) -> str:
        return f"entries.npy"

    # @property
    # def _entries_path(self) -> str:
    #     return f"entries-{self._split.value.upper()}.npy"

    # @property
    # def _class_ids_path(self) -> str:
    #     return f"class-ids-{self._split.value.upper()}.npy"

    # @property
    # def _class_names_path(self) -> str:
    #     return f"class-names-{self._split.value.upper()}.npy"

    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    # def _get_class_ids(self) -> np.ndarray:
    #     if self._split == _Split.TEST:
    #         assert False, "Class IDs are not available in TEST split"
    #     if self._class_ids is None:
    #         self._class_ids = self._load_extra(self._class_ids_path)
    #     assert self._class_ids is not None
    #     return self._class_ids

    # def _get_class_names(self) -> np.ndarray:
    #     if self._split == _Split.TEST:
    #         assert False, "Class names are not available in TEST split"
    #     if self._class_names is None:
    #         self._class_names = self._load_extra(self._class_names_path)
    #     assert self._class_names is not None
    #     return self._class_names

    # def find_class_id(self, class_index: int) -> str:
    #     class_ids = self._get_class_ids()
    #     return str(class_ids[class_index])

    # def find_class_name(self, class_index: int) -> str:
    #     class_names = self._get_class_names()
    #     return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        # entries = self._get_entries()
        # actual_index = entries[index]["actual_index"]

        # class_id = self.get_class_id(index)

        # image_relpath = self.split.get_image_relpath(actual_index, class_id)
        # image_full_path = os.path.join(self.root, image_relpath)
        # with open(image_full_path, mode="rb") as f:
        #     image_data = f.read()
        # return image_data

        img_relpath = self._get_entries()[index]["filename"]
        with open(os.path.join(self.root, img_relpath), mode="rb") as f:
            return f.read()

    def get_target(self, index: int) -> Optional[Target]:
        # entries = self._get_entries()
        # class_index = entries[index]["class_index"]
        # return None if self.split == _Split.TEST else int(class_index)
        entry = self._get_entries()[index]
        code = entry["code"]

        if code == 2:
            return np.zeros(VECTOR_LENGTH, dtype=np.uint32)

        if code == 1:
            base_name, _ = os.path.splitext(os.path.basename(entry["filename"]))
            txt_path = os.path.join(self.root, "labeled", base_name + ".txt")
            return self._load_label_vector(txt_path)

        return None

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return None if self.split == _Split.TEST else entries["class_index"]

    # def get_class_id(self, index: int) -> Optional[str]:
    #     entries = self._get_entries()
    #     class_id = entries[index]["class_id"]
    #     return None if self.split == _Split.TEST else str(class_id)

    # def get_class_name(self, index: int) -> Optional[str]:
    #     entries = self._get_entries()
    #     class_name = entries[index]["class_name"]
    #     return None if self.split == _Split.TEST else str(class_name)

    def __len__(self) -> int:
        entries = self._get_entries()
        # assert len(entries) == self.split.length
        return len(entries)

    # def _load_labels(self, labels_path: str) -> List[Tuple[str, str]]:
    #     labels_full_path = os.path.join(self.root, labels_path)
    #     labels = []

    #     try:
    #         with open(labels_full_path, "r") as f:
    #             reader = csv.reader(f)
    #             for row in reader:
    #                 class_id, class_name = row
    #                 labels.append((class_id, class_name))
    #     except OSError as e:
    #         raise RuntimeError(f'can not read labels file "{labels_full_path}"') from e

    #     return labels

    def _dump_entries(self) -> None:
        raw_dir = os.path.join(self.root, "raw")
        labeled_dir = os.path.join(self.root, "labeled")
        background_dir = os.path.join(self.root, "background")

        def collect_imgs(root):
            return [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(".jpg")]

        raw_imgs = collect_imgs(raw_dir)
        background_imgs = collect_imgs(background_dir)

        imgs = raw_imgs + background_imgs

        dtype = np.dtype(
            [
                ("filename", "U256"),
                ("code", "<u1"),
            ]
        )

        entries_array = np.empty(len(imgs), dtype=dtype)

        for idx, img_path in enumerate(imgs):
            rel_path = os.path.relpath(img_path, self.root)
            base_name, _ = os.path.splitext(os.path.basename(img_path))

            if img_path.startswith(background_dir):
                code = 2
            else:
                txt_path = os.path.join(labeled_dir, base_name + ".txt")
                code = 1 if os.path.exists(txt_path) else 0

            entries_array[idx] = (rel_path, code)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _load_label_vector(self, txt_path: str) -> np.ndarray:
        vec = np.loadtxt(txt_path, dtype=np.uint32)
        if vec.shape[0] != VECTOR_LENGTH:
            raise ValueError(f"{txt_path} must contain {VECTOR_LENGTH} got shape {vec.shape}")
        return vec

    # def _dump_class_ids_and_names(self) -> None:
    #     split = self.split
    #     if split == ImageNet.Split.TEST:
    #         return

    #     entries_array = self._load_extra(self._entries_path)

    #     max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
    #     for entry in entries_array:
    #         class_index, class_id, class_name = (
    #             entry["class_index"],
    #             entry["class_id"],
    #             entry["class_name"],
    #         )
    #         max_class_index = max(int(class_index), max_class_index)
    #         max_class_id_length = max(len(str(class_id)), max_class_id_length)
    #         max_class_name_length = max(len(str(class_name)), max_class_name_length)

    #     class_count = max_class_index + 1
    #     class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
    #     class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
    #     for entry in entries_array:
    #         class_index, class_id, class_name = (
    #             entry["class_index"],
    #             entry["class_id"],
    #             entry["class_name"],
    #         )
    #         class_ids_array[class_index] = class_id
    #         class_names_array[class_index] = class_name

    #     logger.info(f'saving class IDs to "{self._class_ids_path}"')
    #     self._save_extra(class_ids_array, self._class_ids_path)

    #     logger.info(f'saving class names to "{self._class_names_path}"')
    #     self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        # self._dump_class_ids_and_names()
