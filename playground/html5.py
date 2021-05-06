from pathlib import Path


def html5(contents):
    return open(Path(__file__).parent / "5.html").read().replace("<!-- contents -->", contents)
