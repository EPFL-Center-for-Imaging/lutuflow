import re
from typing import List


class TagsProcessor:
    @classmethod
    def get_specimen_tags(cls, tags: List[str]) -> List[str]:
        r = re.compile(r"(C|Animal|ANIMAL)-?\d+")
        return list(sorted(filter(r.match, tags)))

    @classmethod
    def get_image_tags(cls, img_tags: List[str]) -> List[str]:
        r = re.compile(r"(image)(s?)", re.IGNORECASE)
        return list(filter(r.match, img_tags))

    @classmethod
    def get_raw_pred_tags(cls, img_tags: List[str]) -> List[str]:
        r = re.compile(r".*pred.*", re.IGNORECASE)
        return list(filter(r.match, img_tags))

    @classmethod
    def get_scan_time_tags(cls, img_tags: List[str]) -> List[str]:
        """Finds a time stamp (e.g. 'T2' or 'scan1') among image tags based on a regular expression."""
        r = re.compile(r"(Tm?|SCAN|scan)-?\d+")
        return list(sorted(filter(r.match, img_tags)))
    
    @classmethod
    def get_scan_time_idx(cls, time_tag: str) -> float:
        t = re.findall(r"m?\d+", time_tag)[0]
        t = -1.0 if t == "m1" else float(t)
        return t
        
