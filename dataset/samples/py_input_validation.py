def parse_page(params):
    page = int(params.get("page", "1"))
    size = int(params.get("size", "20"))
    offset = (page - 1) * size
    return offset, size


def parse_ids(raw):
    return [int(x) for x in raw.split(",")]
