"""Verified downloads and safe archive extraction (no pickle metadata is loaded)."""
import hashlib
import re
import json
import time
from concurrent.futures import ThreadPoolExecutor
import shutil
import tarfile
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

from common.network import enable_system_trust


def safe_path(root, name):
    root = Path(root).resolve()
    name = name.replace("\\", "/")
    if ":" in name:
        raise ValueError("Archive drive/stream path rejected")
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Unsafe archive path: {name}")
    return target


def extract(archive, root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = safe_path(root, entry.filename)
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                        raise ValueError("Archive symlink rejected")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(entry) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
    else:
        with tarfile.open(archive) as tar:
            for entry in tar:
                target = safe_path(root, entry.name)
                if entry.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif entry.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(entry) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    raise ValueError("Archive links/devices rejected")


def download_ranges(url,path,workers):
    """Resume an immutable HTTP resource in bounded ranges, retaining the existing prefix."""
    path=Path(path);partial=path.with_name(path.name+".part")
    plan_file=path.with_name(path.name+".ranges.json")
    headers={"User-Agent":"MMIP-week5/1.0","Range":"bytes=0-0"}
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=120) as response:
        span=re.fullmatch(r"bytes 0-0/(\d+)",response.headers.get("Content-Range",""))
        if response.status!=206 or not span:return False
        total=int(span[1]);response.read(1)
    offset=partial.stat().st_size if partial.exists() else 0
    if offset==total:
        partial.replace(path);return True
    if offset>total:raise IOError("Partial download exceeds server length")
    if plan_file.exists():
        plan=json.loads(plan_file.read_text())
        if plan["url"]!=url or plan["total"]!=total or plan["offset"]!=offset:
            raise IOError("Range plan disagrees with existing prefix; preserve and inspect partial download")
    else:
        step=(total-offset+workers-1)//workers
        plan={"url":url,"total":total,"offset":offset,"ranges":[[start,min(start+step,total)-1]
              for start in range(offset,total,step)]}
        plan_file.write_text(json.dumps(plan))
    def fetch(item):
        index,(start,end)=item
        chunk=path.with_name(path.name+f".range{index}")
        for attempt in range(3):
            existing=chunk.stat().st_size if chunk.exists() else 0
            length=end-start+1
            if existing==length:return chunk
            if existing>length:raise IOError("Range chunk exceeds planned length")
            request=urllib.request.Request(url,headers={"User-Agent":"MMIP-week5/1.0","Range":f"bytes={start+existing}-{end}"})
            try:
                with urllib.request.urlopen(request,timeout=120) as response:
                    expected=f"bytes {start+existing}-{end}/{total}"
                    if response.status!=206 or response.headers.get("Content-Range")!=expected:
                        raise IOError("Server did not honor requested byte range")
                    with chunk.open("ab") as dst:
                        while block:=response.read(1024*1024):dst.write(block)
                if chunk.stat().st_size!=length:raise IOError("Truncated range")
                print(f"  Range {index+1}/{len(plan['ranges'])} complete ({length/1024**2:.0f} MiB)",flush=True)
                return chunk
            except (OSError,urllib.error.URLError):
                if attempt==2:raise
                time.sleep(2**attempt)
    print(f"  {len(plan['ranges'])} ranged streams; retain {offset/1024**2:.0f} MiB existing prefix",flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:chunks=list(pool.map(fetch,enumerate(plan["ranges"])))
    assembled=path.with_name(path.name+".assembled")
    with assembled.open("wb") as dst:
        if partial.exists():
            with partial.open("rb") as src:shutil.copyfileobj(src,dst)
        for chunk in chunks:
            with chunk.open("rb") as src:shutil.copyfileobj(src,dst)
    if assembled.stat().st_size!=total:raise IOError("Assembled archive length mismatch")
    assembled.replace(path)
    # These are exactly the temporary files created above; no recursive deletion.
    for temp in [partial,plan_file,*chunks]:
        if temp.exists():temp.unlink()
    return True


def download(url, path, md5=None,workers=1):
    enable_system_trust()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() and workers>1:
        download_ranges(url,path,workers)
    if not path.exists():
        partial = path.with_name(path.name + ".part")
        print(f"Downloading {url}", flush=True)
        offset=partial.stat().st_size if partial.exists() else 0
        headers={"User-Agent":"MMIP-week5/1.0"}
        if offset:headers["Range"]=f"bytes={offset}-"
        req = urllib.request.Request(url,headers=headers)
        try:
            response=urllib.request.urlopen(req,timeout=120)
        except urllib.error.HTTPError as error:
            if offset and error.code in (403,416):
                error.close()
                print("  Server rejected range resume; restarting the archive download",flush=True)
                response=urllib.request.urlopen(urllib.request.Request(url,headers={"User-Agent":"MMIP-week5/1.0"}),timeout=120)
            else:raise
        with response as src:
            if src.status==206:
                span=re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)",src.headers.get("Content-Range",""))
                if not span or int(span[1])!=offset:raise IOError("Invalid resume range; partial file preserved")
                total=int(span[3]);mode="ab"
            else:
                offset=0;total=int(src.headers.get("Content-Length",0));mode="wb"
            written, logged = offset, offset
            if offset:print(f"  Resume at {offset/1024**2:.0f} MiB",flush=True)
            with partial.open(mode) as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
                    written += len(block)
                    if written - logged >= 64 * 1024**2:
                        print(f"  {written/1024**2:.0f}/{total/1024**2:.0f} MiB", flush=True)
                        logged = written
        if total and total != written:
            raise IOError("Incomplete download; retry")
        partial.replace(path)
    if md5:
        checksum = hashlib.md5()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024*1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != md5:
            raise ValueError(f"Corrupt archive: {path}; preserve or remove it before retrying")
    return path
