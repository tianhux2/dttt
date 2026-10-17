# coding=utf-8
import os
import glob
from typing import List

class ShardedFileWriter:
    """
    File-like write stream that splits binary data into multiple shard files 
    (base_path.part000, base_path.part001, ...) when max_bytes limit is reached.
    Default max_bytes is 4 GB (4 * 1024 * 1024 * 1024 bytes).
    """
    def __init__(self, base_path: str, max_bytes: int = 4 * 1024 * 1024 * 1024):
        self.base_path = base_path
        self.max_bytes = max_bytes
        self.part_index = 0
        self.current_bytes = 0
        self.total_written = 0
        self.current_file = None
        self._open_next_part()

    def _open_next_part(self) -> None:
        if self.current_file:
            self.current_file.close()
        part_path = f"{self.base_path}.part{self.part_index:03d}"
        parent_dir = os.path.dirname(part_path)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            
        self.current_file = open(part_path, "wb")
        self.part_index += 1
        self.current_bytes = 0

    def write(self, data: bytes) -> int:
        data_len = len(data)
        offset = 0
        while offset < data_len:
            remaining_in_part = self.max_bytes - self.current_bytes
            if remaining_in_part <= 0:
                self._open_next_part()
                remaining_in_part = self.max_bytes
            chunk_size = min(data_len - offset, remaining_in_part)
            self.current_file.write(data[offset : offset + chunk_size])
            self.current_bytes += chunk_size
            self.total_written += chunk_size
            offset += chunk_size
        return data_len

    def tell(self) -> int:
        return self.total_written

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 1 and offset == 0: # SEEK_CUR 0
            return self.total_written
        if whence == 2 and offset == 0: # SEEK_END 0
            return self.total_written
        # If seeking within current file part
        if whence == 0:
            if self.current_file:
                # Fallback seek on active part
                part_start_offset = self.total_written - self.current_bytes
                if offset >= part_start_offset:
                    rel_pos = offset - part_start_offset
                    self.current_file.seek(rel_pos)
                    self.current_bytes = rel_pos
                    self.total_written = part_start_offset + rel_pos
                    return self.total_written
        return self.total_written

    def flush(self) -> None:
        if self.current_file:
            self.current_file.flush()

    def close(self) -> None:
        if self.current_file:
            self.current_file.close()
            self.current_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class ShardedFileReader:
    """
    File-like read stream that seamlessly reads across shard files 
    (base_path.part000, base_path.part001, ...).
    Supports tell() and seek() required by numpy/equinox deserialization.
    Falls back to reading base_path directly if it exists as a single file 
    for 100% backward compatibility with un-sharded checkpoints.
    """
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.part_files: List[str] = []
        self.part_sizes: List[int] = []
        self.part_offsets: List[int] = []
        self.total_length = 0
        self.position = 0
        
        self.current_part_index = -1
        self.current_file = None
        
        part0 = f"{base_path}.part000"
        if os.path.exists(part0):
            self.is_sharded = True
            self.part_files = sorted(glob.glob(f"{base_path}.part*"))
            
            # Strict sequence contiguity validation (000, 001, 002 ...)
            for idx, fpath in enumerate(self.part_files):
                suffix = fpath.rsplit(".part", 1)[-1]
                if not suffix.isdigit() or int(suffix) != idx:
                    raise ValueError(
                        f"❌ Checkpoint part sequence corrupted or illegally renamed! "
                        f"Expected file 'part{idx:03d}', but found '{os.path.basename(fpath)}'."
                    )
            
            self.part_sizes = [os.path.getsize(f) for f in self.part_files]
            
            cumulative = 0
            for sz in self.part_sizes:
                self.part_offsets.append(cumulative)
                cumulative += sz
            self.total_length = cumulative
            self._seek_to_position(0)
        elif os.path.exists(base_path):
            self.is_sharded = False
            self.current_file = open(base_path, "rb")
            self.total_length = os.path.getsize(base_path)
        else:
            raise FileNotFoundError(f"❌ Checkpoint file not found at '{base_path}' or '{part0}'")

    def _seek_to_position(self, pos: int) -> None:
        if not self.is_sharded:
            if self.current_file:
                self.current_file.seek(pos)
                self.position = self.current_file.tell()
            return

        pos = max(0, min(pos, self.total_length))
        self.position = pos

        if pos == self.total_length and self.total_length > 0:
            # Positioned at EOF
            part_idx = len(self.part_files) - 1
            rel_offset = self.part_sizes[part_idx]
        else:
            part_idx = 0
            for i, offset in enumerate(self.part_offsets):
                if i + 1 < len(self.part_offsets):
                    if self.part_offsets[i] <= pos < self.part_offsets[i + 1]:
                        part_idx = i
                        break
                else:
                    part_idx = i
            rel_offset = pos - self.part_offsets[part_idx]

        if self.current_part_index != part_idx or self.current_file is None:
            if self.current_file:
                self.current_file.close()
            self.current_file = open(self.part_files[part_idx], "rb")
            self.current_part_index = part_idx

        self.current_file.seek(rel_offset)

    def tell(self) -> int:
        if not self.is_sharded:
            return self.current_file.tell()
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0: # SEEK_SET
            target = offset
        elif whence == 1: # SEEK_CUR
            target = self.position + offset
        elif whence == 2: # SEEK_END
            target = self.total_length + offset
        else:
            raise ValueError(f"Invalid whence value: {whence}")

        self._seek_to_position(target)
        return self.tell()

    def read(self, size: int = -1) -> bytes:
        if not self.is_sharded:
            chunk = self.current_file.read(size)
            self.position = self.current_file.tell()
            return chunk

        if size is None or size < 0:
            size = self.total_length - self.position

        if size <= 0 or self.position >= self.total_length:
            return b""

        buffer = bytearray()
        remaining = size

        while remaining > 0 and self.position < self.total_length:
            curr_part_start = self.part_offsets[self.current_part_index]
            curr_part_size = self.part_sizes[self.current_part_index]
            offset_in_part = self.position - curr_part_start
            bytes_left_in_part = curr_part_size - offset_in_part

            to_read = min(remaining, bytes_left_in_part)
            chunk = self.current_file.read(to_read)
            if not chunk:
                break
            buffer.extend(chunk)
            self.position += len(chunk)
            remaining -= len(chunk)

            if self.position - curr_part_start >= curr_part_size:
                if self.current_part_index + 1 < len(self.part_files):
                    self._seek_to_position(self.position)
                else:
                    break

        return bytes(buffer)

    def close(self) -> None:
        if self.current_file:
            self.current_file.close()
            self.current_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def get_checkpoint_files(base_path: str) -> List[str]:
    """
    Returns all file paths associated with a checkpoint base_path.
    Includes base_path.part* if sharded, or [base_path] if single un-sharded file.
    """
    parts = sorted(glob.glob(f"{base_path}.part*"))
    if parts:
        return parts
    elif os.path.exists(base_path):
        return [base_path]
    return []

def checkpoint_exists(base_path: str) -> bool:
    """
    Checks if a checkpoint exists, either as sharded part000 or single file.
    """
    return os.path.exists(f"{base_path}.part000") or os.path.exists(base_path)
