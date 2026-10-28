import io, os, zipfile

class MultiFileReader(io.RawIOBase):
    """Present several ordered files as one seekable file."""
    def __init__(self, paths):
        self.paths = paths
        self.files = [open(p, 'rb', buffering=1024*1024) for p in paths]
        self.sizes = [os.path.getsize(p) for p in paths]
        self.offsets = []
        total = 0
        for s in self.sizes:
            self.offsets.append(total)
            total += s
        self.total = total
        self.pos = 0

    def seek(self, offset, whence=0):
        if whence == 0: self.pos = offset
        elif whence == 1: self.pos += offset
        else: self.pos = self.total + offset
        return self.pos

    def tell(self):
        return self.pos

    def readable(self): return True
    def seekable(self): return True

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total - self.pos
        if self.pos >= self.total:
            return b''
        n = min(n, self.total - self.pos)
        chunks = []
        remaining = n
        while remaining > 0:
            # find file index
            i = self._file_index(self.pos)
            f = self.files[i]
            local = self.pos - self.offsets[i]
            f.seek(local)
            want = min(remaining, self.sizes[i] - local)
            data = f.read(want)
            if not data:
                break
            chunks.append(data)
            self.pos += len(data)
            remaining -= len(data)
        return b''.join(chunks)

    def _file_index(self, pos):
        # binary search over offsets
        lo, hi = 0, len(self.offsets) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.offsets[mid] <= pos: lo = mid
            else: hi = mid - 1
        return lo

    def close(self):
        for f in self.files:
            f.close()

def open_split_zip(directory):
    parts = sorted(p for p in os.listdir(directory) if p.startswith('images.z'))
    parts = [os.path.join(directory, p) for p in parts] + [os.path.join(directory, 'images.zip')]
    return zipfile.ZipFile(io.BufferedReader(MultiFileReader(parts), buffer_size=4*1024*1024))


# --- allow multi-disk (split) archives: full replacement of _EndRecData64 ---
import struct
def _EndRecData64_multidisk(fpin, offset, endrec):
    offset -= zipfile.sizeEndCentDir64Locator
    if offset < 0:
        return endrec
    fpin.seek(offset)
    data = fpin.read(zipfile.sizeEndCentDir64Locator)
    if len(data) != zipfile.sizeEndCentDir64Locator:
        raise OSError("Unknown I/O error")
    sig, diskno, reloff, disks = struct.unpack(zipfile.structEndArchive64Locator, data)
    if sig != zipfile.stringEndArchive64Locator:
        return endrec
    # NOTE: disk_no/disks check removed - we present the parts as one file.

    offset -= zipfile.sizeEndCentDir64
    fpin.seek(reloff)
    extrasz = offset - reloff
    data = fpin.read(zipfile.sizeEndCentDir64)
    if len(data) != zipfile.sizeEndCentDir64:
        raise OSError("Unknown I/O error")
    if not data.startswith(zipfile.stringEndArchive64) and reloff != offset:
        fpin.seek(offset)
        extrasz = 0
        data = fpin.read(zipfile.sizeEndCentDir64)
        if len(data) != zipfile.sizeEndCentDir64:
            raise OSError("Unknown I/O error")
    if not data.startswith(zipfile.stringEndArchive64):
        raise zipfile.BadZipFile("Zip64 end of central directory record not found")

    sig, sz, create_version, read_version, disk_num, disk_dir, \
        dircount, dircount2, dirsize, diroffset = \
        struct.unpack(zipfile.structEndArchive64, data)
    if (diroffset + dirsize != reloff or
        sz + 12 != zipfile.sizeEndCentDir64 + extrasz):
        raise zipfile.BadZipFile("Corrupt zip64 end of central directory record")

    endrec[zipfile._ECD_SIGNATURE] = sig
    endrec[zipfile._ECD_DISK_NUMBER] = 0
    endrec[zipfile._ECD_DISK_START] = 0
    endrec[zipfile._ECD_ENTRIES_THIS_DISK] = dircount
    endrec[zipfile._ECD_ENTRIES_TOTAL] = dircount2
    endrec[zipfile._ECD_SIZE] = dirsize
    endrec[zipfile._ECD_OFFSET] = diroffset
    endrec[zipfile._ECD_LOCATION] = offset - extrasz
    return endrec

zipfile._EndRecData64 = _EndRecData64_multidisk
