"""Read plots from SVG files exactly, from their vector drawing."""

import base64
import binascii
import gzip
import html
import math
import os
import re
import unicodedata
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Iterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, unquote_to_bytes, urlsplit
from xml.parsers import expat

import cv2
import numpy as np
import numpy.typing as npt

from .axes import _PLAIN, _SUPERSCRIPT, _normal, _Text
from .calibration import Point
from .digitize import Colour, Image
from .pdf import (
    _SUBSCRIPTS,
    _SUPERSCRIPTS,
    PREVIEW_SCALE,
    Box,
    Drawing,
    _Char,
    _holds,
    _joined,
    _mapped,
    _overlap,
    _Path,
    _Picture,
    _runs,
    _written,
)

type _Outline = tuple[
    list[npt.NDArray[np.float64]], list[bool], npt.NDArray[np.float64]
]
"""The subpaths of a shape, whether each is closed, and its straight pieces."""

_MAX_SIZE = 64 * 2**20  # an SVG file may hold this many bytes, decompressed
_MAX_DEPTH = 256  # elements may be nested this deep, through <use> elements too
_MAX_DRAWN = 10**6  # an SVG file may draw this many elements, through <use> elements
_COMMENT = "#comment"  # the tag of an XML comment, as read
_SVG = "http://www.w3.org/2000/svg"  # the namespaces of SVG's elements and attributes
_XLINK = "http://www.w3.org/1999/xlink"
_POINTS = {  # points in each unit of length, a CSS pixel when none is given
    "": 0.75,
    "px": 0.75,
    "pt": 1,
    "pc": 12,
    "in": 72,
    "cm": 72 / 2.54,
    "mm": 72 / 25.4,
    "q": 72 / 101.6,
}
_PIXELS = {  # CSS pixels, the user units of text and lines, in each unit of length
    unit: points / _POINTS["px"] for unit, points in _POINTS.items()
}
_PAGE = (300.0, 150.0)  # the size in pixels of a drawing that gives none, as in CSS
_FONT_SIZE = 16.0  # the font size in pixels of text that gives none, as in CSS
_EX = 0.5  # an ex is this many font sizes
_SUPER = 0.4  # baseline-shift: super raises text this many of its line's font sizes
_SUB = -0.2  # and sub this many, so lowers it
_INK = (0.0, 0.73)  # characters reach from and to this many font sizes up
_DESCENDING = (-0.21, 0.73)  # characters with descenders, in _DESCENDERS
_BRACKETS = (-0.13, 0.76)  # brackets, in _BRACKETED
_DESCENDERS = frozenset("gjpqy,;_")
_BRACKETED = frozenset("()[]{}|")
_NARROW = frozenset(" .,:;'!|()[]-fijlrtI")  # characters 0.3 font sizes wide
_WIDE = frozenset("mwMW%@")  # characters 0.9 font sizes wide; capitals 0.7, others 0.6
_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\u2060\ufeff")  # such as after superscripts
_ALIGN = {"Min": 0.0, "Mid": 0.5, "Max": 1.0}  # preserveAspectRatio's alignments
_EVERYWHERE: Box = (-math.inf, -math.inf, math.inf, math.inf)  # hides nothing
_NOWHERE: Box = (math.inf, math.inf, -math.inf, -math.inf)  # hides everything
_SHIFT = 4  # OpenCV draws to a sixteenth of a pixel, 2 to the -4
_DRAWN = frozenset(  # elements that draw, or that hold elements that do
    {"g", "a", "switch", "svg", "use", "text", "image"}
    | {"path", "rect", "circle", "ellipse", "line", "polyline", "polygon"}
)
_INHERITED = frozenset(  # properties that elements take from those that hold them
    {"fill", "fill-opacity", "stroke", "stroke-opacity", "stroke-width"}
    | {"stroke-dasharray", "color", "visibility", "font-size", "text-anchor"}
)
_ENDS = {"M": 0, "L": 0, "T": 0, "C": 4, "S": 2, "Q": 2, "A": 5}  # end x's index
_ARGUMENTS = {  # how many values each command of a path takes
    "M": 2,
    "L": 2,
    "H": 1,
    "V": 1,
    "C": 6,
    "S": 4,
    "Q": 4,
    "T": 2,
    "A": 7,
}
_COMMAND = re.compile(r"[\s,]*([MmZzLlHhVvCcSsQqTtAa])")  # a path's command
_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")  # 1, -.5 or 2e3
_ARGUMENT = re.compile(rf"[\s,]*({_NUMBER.pattern})")  # a number after a command
_FLAG = re.compile(r"[\s,]*([01])")  # an arc's flags may be written without spaces
_LENGTH = re.compile(rf"({_NUMBER.pattern})(px|pt|pc|in|cm|mm|q|em|ex|%)?")  # 3pt
_TRANSFORM = re.compile(  # one of a transform's transformations, such as scale(2)
    r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)"
)
_URL = re.compile(r"url\(\s*['\"]?#([^'\")]*)['\"]?\s*\)\s*(.*)")  # url(#id) red
_HEX = re.compile(r"#(?:([0-9a-f]{6}|[0-9a-f]{8})|([0-9a-f]{3,4}))")  # #1f77b4
_RGB = re.compile(r"rgba?\(([^)]*)\)")  # rgb(31, 119, 180)
_DOLLAR = re.compile(r"(?<!\\)\$")  # a dollar sign that starts or ends mathtext
_TEX = re.compile(r"\\[A-Za-z]+|\\.|.", re.DOTALL)  # a command or a character
_GREEK = re.findall(  # the names of Greek letters, as Unicode spells them
    r"\w+",
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lamda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega",
)
_SYMBOLS = (  # what mathtext's commands write, other than their own names
    {name: unicodedata.lookup(f"GREEK SMALL LETTER {name}") for name in _GREEK}
    | {
        name.capitalize(): unicodedata.lookup(f"GREEK CAPITAL LETTER {name}")
        for name in _GREEK
    }
    | {"lambda": "\u03bb", "Lambda": "\u039b", "varepsilon": "\u03b5"}
    | {"vartheta": "\u03d1", "varphi": "\u03c6", "varrho": "\u03f1", "ell": "\u2113"}
    | {"times": "\u00d7", "cdot": "\u00b7", "pm": "\u00b1", "mp": "\u2213"}
    | {"circ": "\u2218", "degree": "\u00b0", "infty": "\u221e", "AA": "\u00c5"}
    | {"partial": "\u2202", "nabla": "\u2207", "hbar": "\u0127", "prime": "\u2032"}
    | {"approx": "\u2248", "sim": "\u223c", "propto": "\u221d", "neq": "\u2260"}
    | {"leq": "\u2264", "geq": "\u2265", "le": "\u2264", "ge": "\u2265"}
    | {"langle": "\u27e8", "rangle": "\u27e9", "rightarrow": "\u2192", "to": "\u2192"}
    | {"minus": "\u2212", "div": "\u00f7", "ldots": "\u2026", "cdots": "\u22ef"}
    | dict.fromkeys([",", ":", ";", " ", "quad", "qquad"], " ")
    | dict.fromkeys(["!", "left", "right", "displaystyle"], "")
)
_UNARY = frozenset(  # mathtext commands that write their argument as it is
    re.findall(
        r"\w+",
        "mathdefault mathrm mathit mathbf mathsf mathtt mathcal mathbb mathfrak "
        "mathregular mathnormal text textrm textit textbf operatorname boldsymbol "
        "hat bar vec dot ddot tilde acute grave breve check overline underline "
        "widehat widetilde",
    )
)
_NAMED: dict[str, Colour] = {  # CSS's named colours
    name: (int(code[:2], 16), int(code[2:4], 16), int(code[4:], 16))
    for name, code in re.findall(
        r"(\w+):(\w+)",
        """
aliceblue:f0f8ff antiquewhite:faebd7 aqua:00ffff aquamarine:7fffd4 azure:f0ffff
beige:f5f5dc bisque:ffe4c4 black:000000 blanchedalmond:ffebcd blue:0000ff
blueviolet:8a2be2 brown:a52a2a burlywood:deb887 cadetblue:5f9ea0 chartreuse:7fff00
chocolate:d2691e coral:ff7f50 cornflowerblue:6495ed cornsilk:fff8dc crimson:dc143c
cyan:00ffff darkblue:00008b darkcyan:008b8b darkgoldenrod:b8860b darkgray:a9a9a9
darkgreen:006400 darkgrey:a9a9a9 darkkhaki:bdb76b darkmagenta:8b008b
darkolivegreen:556b2f darkorange:ff8c00 darkorchid:9932cc darkred:8b0000
darksalmon:e9967a darkseagreen:8fbc8f darkslateblue:483d8b darkslategray:2f4f4f
darkslategrey:2f4f4f darkturquoise:00ced1 darkviolet:9400d3 deeppink:ff1493
deepskyblue:00bfff dimgray:696969 dimgrey:696969 dodgerblue:1e90ff firebrick:b22222
floralwhite:fffaf0 forestgreen:228b22 fuchsia:ff00ff gainsboro:dcdcdc
ghostwhite:f8f8ff gold:ffd700 goldenrod:daa520 gray:808080 green:008000
greenyellow:adff2f grey:808080 honeydew:f0fff0 hotpink:ff69b4 indianred:cd5c5c
indigo:4b0082 ivory:fffff0 khaki:f0e68c lavender:e6e6fa lavenderblush:fff0f5
lawngreen:7cfc00 lemonchiffon:fffacd lightblue:add8e6 lightcoral:f08080
lightcyan:e0ffff lightgoldenrodyellow:fafad2 lightgray:d3d3d3 lightgreen:90ee90
lightgrey:d3d3d3 lightpink:ffb6c1 lightsalmon:ffa07a lightseagreen:20b2aa
lightskyblue:87cefa lightslategray:778899 lightslategrey:778899
lightsteelblue:b0c4de lightyellow:ffffe0 lime:00ff00 limegreen:32cd32 linen:faf0e6
magenta:ff00ff maroon:800000 mediumaquamarine:66cdaa mediumblue:0000cd
mediumorchid:ba55d3 mediumpurple:9370db mediumseagreen:3cb371
mediumslateblue:7b68ee mediumspringgreen:00fa9a mediumturquoise:48d1cc
mediumvioletred:c71585 midnightblue:191970 mintcream:f5fffa mistyrose:ffe4e1
moccasin:ffe4b5 navajowhite:ffdead navy:000080 oldlace:fdf5e6 olive:808000
olivedrab:6b8e23 orange:ffa500 orangered:ff4500 orchid:da70d6 palegoldenrod:eee8aa
palegreen:98fb98 paleturquoise:afeeee palevioletred:db7093 papayawhip:ffefd5
peachpuff:ffdab9 peru:cd853f pink:ffc0cb plum:dda0dd powderblue:b0e0e6
purple:800080 rebeccapurple:663399 red:ff0000 rosybrown:bc8f8f royalblue:4169e1
saddlebrown:8b4513 salmon:fa8072 sandybrown:f4a460 seagreen:2e8b57 seashell:fff5ee
sienna:a0522d silver:c0c0c0 skyblue:87ceeb slateblue:6a5acd slategray:708090
slategrey:708090 snow:fffafa springgreen:00ff7f steelblue:4682b4 tan:d2b48c
teal:008080 thistle:d8bfd8 tomato:ff6347 turquoise:40e0d0 violet:ee82ee
wheat:f5deb3 white:ffffff whitesmoke:f5f5f5 yellow:ffff00 yellowgreen:9acd32
""",
    )
}


@dataclass(frozen=True, slots=True)
class _State:
    """What an element of an SVG file takes from the elements that hold it.

    ``matrix`` maps from its user space to points from the bottom-left corner
    of the drawing, and ``clip`` is the box it is shown in. ``styles`` holds
    the properties it inherits, as written, ``opacity`` that of the groups
    that hold it, ``size`` its font size in user units, and ``viewport`` the
    width and height of the viewport it lies in, in user units. ``anchor``
    is where the origin of the ``<use>`` element that draws it lies, or None
    if none does, ``depth`` how deeply it lies, through ``<use>`` elements,
    and ``holders`` the ids of the elements that hold it, through them too.
    """

    matrix: npt.NDArray[np.float64]
    clip: Box
    styles: dict[str, str]
    opacity: float
    size: float
    viewport: tuple[float, float]
    anchor: Point | None
    depth: int
    holders: frozenset[str]


@dataclass(frozen=True, slots=True)
class _Span:
    """A ``<text>`` element, or a ``<tspan>`` in one, as its characters are placed.

    ``state`` is what the element takes and declares, ``places`` its lists
    of x, y, dx and dy, for its characters in turn, ``rise`` how far its
    baseline lies above that of the text, and ``holder`` the index of the
    span that holds it, or None for the text, all in user units.
    """

    state: _State
    places: dict[str, list[float]]
    rise: float
    holder: int | None


type _Letter = tuple[str, _Span, float, float, float]
"""A character of a text, its span, its x and y, and its width, in user units."""


def load_svg(path: str | os.PathLike[str], box: Box | None = None) -> Drawing:
    """Read what an SVG file draws: its text, paths and images.

    The drawing is read as a page of a PDF is, so that ``read_pdf_axes``,
    ``find_pdf_colours``, ``trace_pdf_curves`` and ``find_pdf_markers`` work
    on it, with positions in points, 72 to the inch, from its bottom-left
    corner. A user unit is a CSS pixel, 0.75 points, unless the root's
    ``viewBox`` maps the width and height the file gives to it otherwise, as
    matplotlib's, whose user units are points.

    Paths are read through their transforms, with the points they pass
    through, leaving out the control points of curved pieces, and arcs give
    their points at each quarter turn of their ellipses. Each ``<use>`` of a
    shape is drawn as a form is in a PDF, with its origin at its x and y, as
    matplotlib places markers. Paths are shown in the boxes around their
    clip paths, and colours are taken from presentation attributes and
    ``style`` attributes, as elements inherit them, including currentColor,
    while a gradient gives its first colour. Text is read from ``<text>``
    and ``<tspan>`` elements, placing each character as the PDF reader does,
    though as the file does not give its width, each is taken to be about
    0.6 of its font size wide; superscripts are raised by ``baseline-shift``
    or ``dy``. matplotlib's text drawn as glyph outlines, in groups whose
    ids start ``text_``, is read from the comment before each line of it,
    with the box of its outlines, and so is its mathtext written as text, a
    character at a time. Images, as data URIs or files beside the SVG file,
    are read on white, with any transparency.

    Files compressed with gzip, as ``.svgz`` files are, are read too. Files
    larger than ``_MAX_SIZE`` bytes, decompressed, and those that declare XML
    entities, which can expand to fill the memory, are refused.

    Args:
        path: The SVG file.
        box: The left, bottom, right and top edges of the part of the drawing
            to read, in points from its bottom-left corner, such as one plot
            in it. Text whose middle lies outside it is left out, and so are
            paths and images wholly outside it; the rest is hidden beyond it,
            as though clipped. Defaults to the whole drawing.

    Returns:
        What the file draws, within the box, as on page 1.

    Raises:
        ValueError: If the file cannot be read as SVG, is too large, declares
            entities, draws too much, or the box lies off the drawing.
    """
    name = os.fspath(path)
    root = _parsed(_read(name), name)
    if root.tag != "svg":
        msg = f"{name!r} is not an SVG file: its root is <{root.tag}>, not <svg>"
        raise ValueError(msg)
    width, height = _size(root)
    bounds = (0.0, 0.0, width, height)
    area = bounds if box is None else _overlap(bounds, box)
    if area is None:
        msg = f"the box {_written(box)} lies off the drawing, {_written(bounds)}"
        raise ValueError(msg)
    view = _view_box(root)
    pixel = _POINTS["px"]
    page = np.array([[1, 0, 0], [0, -1, height], [0, 0, 1]])
    fit = _fit(view, width, height, root.get("preserveAspectRatio", ""))
    matrix = page @ (fit if view else np.diag([pixel, pixel, 1]))
    viewport = (view[2], view[3]) if view else (width / pixel, height / pixel)
    reader = _Reader(root, name)
    reader.walk(
        root,
        _State(matrix, area, {}, 1, _FONT_SIZE, viewport, None, 0, frozenset()),
    )
    texts = [_joined(run) for run in _runs(reader.chars)] + reader.texts
    return Drawing(
        name,
        1,
        area,
        tuple(text for text in texts if text.text and _holds(area, (text.x, text.y))),
        tuple(reader.paths),
        tuple(reader.pictures),
    )


def render_svg(drawing: Drawing, scale: float = PREVIEW_SCALE) -> Image:
    """Draw what was read of an SVG file as an image, as for a preview.

    The paths, images and lines of text that were read are drawn with
    OpenCV, each within its clip: paths, filled and stroked, then images,
    darkening what lies under them, so that axes drawn before an image of
    curves, on white, still show, then text, in black, fitted to its box if
    it runs across, up or down. Slanted text is drawn as its box.

    Args:
        drawing: What the file draws, as ``load_svg`` returns it.
        scale: Pixels per point.

    Returns:
        A ``(height, width, 3)`` ``uint8`` array of red, green and blue
        levels, whose bottom-left corner is that of the drawing's box.
    """
    left, bottom, right, top = drawing.box
    width = max(1, round((right - left) * scale))
    height = max(1, round((top - bottom) * scale))
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    def edges(box: Box) -> npt.NDArray[np.int_]:
        """Return the first and last rows and columns, past the end, of a box."""
        ends = np.array([top - box[3], top - box[1], box[0] - left, box[2] - left])
        return np.round(ends * scale).astype(int)

    def inside(*boxes: Box) -> tuple[slice, slice]:
        """Return the rows and columns of the canvas where boxes overlap."""
        found = np.array([edges(box) for box in boxes])
        first, start = np.maximum(found[:, [0, 2]].max(axis=0), 0)
        last, stop = np.minimum(found[:, [1, 3]].min(axis=0), (height, width))
        return slice(first, last), slice(start, stop)

    for path in drawing.paths:
        rows, columns = inside(path.clip)
        view = canvas[rows, columns]
        corner = np.array([left + columns.start / scale, top - rows.start / scale])
        polygons = [
            np.round((points - corner) * (scale, -scale) * 2**_SHIFT).astype(np.int32)
            for points in path.subpaths
        ]
        if path.fill is not None:
            cv2.fillPoly(view, polygons, path.fill, cv2.LINE_AA, _SHIFT)
        if path.stroke is not None:
            thickness = max(1, round(path.width * scale))
            for polygon, closed in zip(polygons, path.closed, strict=True):
                cv2.polylines(
                    view, [polygon], closed, path.stroke, thickness, cv2.LINE_AA, _SHIFT
                )
    for picture in drawing.pictures:
        first, last, start, stop = edges(picture.box)
        full = cv2.resize(
            picture.image,
            (max(1, stop - start), max(1, last - first)),
            interpolation=cv2.INTER_AREA,
        )
        rows, columns = inside(picture.box, picture.clip)
        shown = full[
            rows.start - first : rows.stop - first,
            columns.start - start : columns.stop - start,
        ]
        np.minimum(canvas[rows, columns], shown, out=canvas[rows, columns])
    for text in drawing.texts:
        rows, columns = inside((text.left, text.bottom, text.right, text.top))
        _write(canvas[rows, columns], text)
    return canvas


class _Reader:
    """Read what the elements of an SVG file draw, from its root down."""

    def __init__(self, root: ET.Element, name: str) -> None:
        """Start reading the SVG file of the given name and root."""
        self.name = name
        self.folder = Path(name).parent
        self.ids: dict[str, ET.Element] = {}
        for element in root.iter():
            if identity := element.get("id"):
                self.ids.setdefault(identity, element)
        self.drawn = 0
        self.chars: list[_Char] = []
        self.texts: list[_Text] = []
        self.paths: list[_Path] = []
        self.pictures: list[_Picture] = []

    def walk(self, element: ET.Element, state: _State, *, used: bool = False) -> None:
        """Read what an element draws, with what it takes from those that hold it.

        A ``<symbol>`` is drawn only as the target of a ``<use>``.
        """
        tag = element.tag
        if not (tag in _DRAWN or (used and tag == "symbol")):
            return
        entered = self._entered(element, state)
        if entered is None:
            return
        if tag == "use":
            self._use(element, entered)
        elif tag == "text":
            self._text(element, entered)
        elif tag == "image":
            self._image(element, entered)
        elif tag in {"g", "a", "switch", "svg", "symbol"}:
            if tag == "svg" and state.depth:
                entered = self._viewport(element, entered)
            self._children(element, entered)
        else:
            self._shape(element, entered)

    def _entered(self, element: ET.Element, state: _State) -> _State | None:
        """Return what an element draws with, or None if it draws nothing.

        It draws nothing if its display is none, or its clip path hides it.

        Raises:
            ValueError: If the element lies more than ``_MAX_DEPTH`` deep, or
                more than ``_MAX_DRAWN`` elements have been drawn.
        """
        declared = _declared(element)
        if declared.get("display") == "none":
            return None
        self._count(state.depth)
        matrix = state.matrix @ _transform(element.get("transform", ""))
        clip = _overlap(state.clip, self._clip(declared.get("clip-path", ""), matrix))
        if clip is None:
            return None
        identity = element.get("id")
        inherited = {
            key: declared[key]
            for key in _INHERITED & declared.keys()
            if declared[key] != "inherit"
        }
        size = _length(declared.get("font-size"), state.size, state.size)
        return replace(
            state,
            matrix=matrix,
            clip=clip,
            styles=state.styles | inherited,
            opacity=state.opacity * _opacity(declared.get("opacity")),
            size=state.size if size is None else size,
            depth=state.depth + 1,
            holders=state.holders | {identity} if identity else state.holders,
        )

    def _count(self, depth: int) -> None:
        """Count an element drawn at a depth, checking that the file draws no more.

        Raises:
            ValueError: If the element lies more than ``_MAX_DEPTH`` deep, or
                more than ``_MAX_DRAWN`` elements have been drawn.
        """
        self.drawn += 1
        if depth > _MAX_DEPTH:
            msg = f"SVG {self.name!r} nests elements more than {_MAX_DEPTH} deep"
            raise ValueError(msg)
        if self.drawn > _MAX_DRAWN:
            msg = f"SVG {self.name!r} draws more than {_MAX_DRAWN} elements"
            raise ValueError(msg)

    def _viewport(self, element: ET.Element, state: _State) -> _State:
        """Return what an ``<svg>`` inside another draws with, in its viewport.

        The viewport's x, y, width and height are in the user units of the
        element that holds it, and percentages are of that element's
        viewport. Its ``viewBox`` is fitted to it, and its content is clipped
        to it.
        """
        width, height = state.viewport
        x = _length(element.get("x"), state.size, width) or 0
        y = _length(element.get("y"), state.size, height) or 0
        wide = _length(element.get("width"), state.size, width) or width
        high = _length(element.get("height"), state.size, height) or height
        view = _view_box(element)
        corners = np.array([(x, y), (x + wide, y + high)], dtype=np.float64)
        clip = _overlap(state.clip, _bounds(_mapped_all(state.matrix, corners)))
        preserve = element.get("preserveAspectRatio", "")
        return replace(
            state,
            matrix=state.matrix @ _translation(x, y) @ _fit(view, wide, high, preserve),
            clip=_NOWHERE if clip is None else clip,
            viewport=(view[2], view[3]) if view else (wide, high),
        )

    def _children(self, element: ET.Element, state: _State) -> None:
        """Read what the children of an element draw.

        In a group whose id starts ``text_``, the element after a comment
        is a line of text that matplotlib writes, as ``_commented`` reads it.
        """
        glyphs = element.get("id", "").startswith("text_")
        previous = None
        for child in element:
            if glyphs and previous is not None and previous.tag == _COMMENT:
                self._commented(previous.text or "", child, state)
            else:
                self.walk(child, state)
            previous = child

    def _use(self, element: ET.Element, state: _State) -> None:
        """Read what a ``<use>`` draws: its target, with its origin at its x and y.

        A target that is not in the file, or that holds the ``<use>``, draws
        nothing, and nor do targets whose ids start ``glyph``, as those of
        cairo's text drawn as glyph outlines do, as they would be taken for
        markers.
        """
        reference = _reference(element.get("href", ""))
        target = self.ids.get(reference)
        if (
            target is None
            or reference in state.holders
            or reference.startswith("glyph")
        ):
            return
        width, height = state.viewport
        x = _length(element.get("x"), state.size, width) or 0
        y = _length(element.get("y"), state.size, height) or 0
        matrix = state.matrix @ _translation(x, y)
        moved = replace(state, matrix=matrix, anchor=_mapped(matrix, (0, 0)))
        self.walk(target, moved, used=True)

    def _shape(self, element: ET.Element, state: _State) -> None:
        """Read the path that a shape, such as a ``<path>`` or ``<circle>``, draws.

        A path that is neither stroked nor filled, is hidden, or lies wholly
        outside its clip is left out. A ``<line>`` is never filled.
        """
        outline = _outline(element, state.size, state.viewport)
        if outline is None or not _visible(state):
            return
        subpaths, closed, segments = outline
        stroke = self._paint(state, "stroke", "none")
        fill = None if element.tag == "line" else self._paint(state, "fill", "black")
        points = [_mapped_all(state.matrix, subpath) for subpath in subpaths]
        box = _bounds(np.vstack(points))
        if (stroke is None and fill is None) or _overlap(state.clip, box) is None:
            return
        scale = _scale(state.matrix)
        width = _length(state.styles.get("stroke-width"), state.size, state.size)
        ends = _mapped_all(state.matrix, segments.reshape(-1, 2)).reshape(-1, 4)
        self.paths.append(
            _Path(
                tuple(points),
                tuple(closed),
                ends,
                stroke,
                fill,
                state.clip,
                state.anchor,
                width=(1 if width is None else width) * scale,
                dashes=tuple(
                    length * scale
                    for length in _dashes(state.styles.get("stroke-dasharray"))
                ),
            )
        )

    def _paint(self, state: _State, kind: str, default: str) -> Colour | None:
        """Return the colour that a shape is stroked or filled in, or None if not.

        ``kind`` is stroke or fill, and ``default`` the paint of an element
        that gives none. A shape is not painted where its paint is none, or
        its opacity, with that of the groups that hold it, is 0. A gradient
        paints in its first colour, unless a colour is given after it, and
        anything else a URL names, such as a pattern, in black.
        """
        paint = state.styles.get(kind, default).strip()
        if match := _URL.fullmatch(paint):
            target = self.ids.get(match[1])
            stop = None if target is None else next(target.iter("stop"), None)
            first = "black" if stop is None else _declared(stop).get("stop-color")
            paint = match[2] or first or "black"
        if paint.lower() == "currentcolor":
            paint = state.styles.get("color", "black")
        opacity = state.opacity * _opacity(state.styles.get(f"{kind}-opacity"))
        return _colour(paint) if opacity > 0 else None

    def _clip(self, reference: str, matrix: npt.NDArray[np.float64]) -> Box:
        """Return the box around the clip path that a clip-path property names.

        The clip path lies in the user space of the element it clips, which
        ``matrix`` maps to the page, and its box is that around its shapes.
        A clip-path that names no clip path clips nothing, and a clip path
        without shapes clips everything away.
        """
        match = _URL.fullmatch(reference.strip())
        target = self.ids.get(match[1]) if match else None
        if target is None or target.tag != "clipPath":
            return _EVERYWHERE
        points = self._ink(target, matrix, 0)
        return _bounds(np.vstack(points)) if points else _NOWHERE

    def _ink(
        self, element: ET.Element, matrix: npt.NDArray[np.float64], depth: int
    ) -> list[npt.NDArray[np.float64]]:
        """Return the points, on the page, of the shapes an element draws.

        ``matrix`` maps from the user space of the element that holds it to
        the page. Shapes in ``<defs>`` are left out, but those that a
        ``<use>`` draws are not.
        """
        self._count(depth)
        if element.tag == "defs":
            return []
        matrix = matrix @ _transform(element.get("transform", ""))
        if element.tag == "use":
            target = self.ids.get(_reference(element.get("href", "")))
            x, y = (_length(element.get(name), _FONT_SIZE) or 0 for name in "xy")
            return (
                []
                if target is None
                else self._ink(target, matrix @ _translation(x, y), depth + 1)
            )
        outline = _outline(element, _FONT_SIZE, (0, 0))
        if outline is not None:
            return [_mapped_all(matrix, points) for points in outline[0]]
        return [
            points
            for child in element
            for points in self._ink(child, matrix, depth + 1)
        ]

    def _commented(self, comment: str, element: ET.Element, state: _State) -> None:
        """Read a line of text that matplotlib writes after a comment holding it.

        matplotlib draws text as glyph outlines, by default, and writes
        mathtext as text a character at a time, grouped by font rather than
        in the order they are read, each after a comment that holds it. The
        line's text is the comment, with its mathtext written as plain text,
        and its box that around its glyphs, or its characters, as
        ``_text`` places them.
        """
        if next(element.iter("text"), None) is None:
            points = self._ink(element, state.matrix, state.depth)
            matrix = state.matrix @ _transform(element.get("transform", ""))
            boxes = [_bounds(np.vstack(points))] if points else []
            slant = _angle(matrix)
        else:
            first = len(self.chars)
            self.walk(element, state)
            chars, self.chars[first:] = self.chars[first:], []
            boxes = [char.box for char in chars]
            slant = chars[0].angle if chars else 0
        if boxes:
            self.texts.append(
                _Text(
                    _normal(_plain(html.unescape(comment.strip()))),
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                    slant=slant,
                )
            )

    def _text(self, element: ET.Element, state: _State) -> None:
        """Read the characters of a ``<text>`` element and the ``<tspan>``s in it.

        White space is collapsed, as SVG lays text out, and characters are
        placed as ``_placed`` places them. A run of characters placed after
        one another is moved back by half its width, or its width, if its
        text-anchor is middle or end, and stretched to the text's textLength
        if it gives one and is one run. A space at the end of a run is left
        out, as the next character is placed on its own, and so are hidden
        characters.
        """
        spans = [_Span(state, _places(element, state), 0, None)]
        runs = _placed(spans, _collapsed(self._letters(element, spans, 0)))
        length = _length(element.get("textLength"), state.size, state.viewport[0])
        for run in runs:
            start = run[0][2]
            width = sum(advance for *_, advance in run)
            stretch = length / width if length and width and len(runs) == 1 else 1
            anchor = run[0][1].state.styles.get("text-anchor", "start").strip()
            shift = width * stretch * {"middle": 0.5, "end": 1.0}.get(anchor, 0)
            for number, (char, span, x, y, advance) in enumerate(run):
                last = char == " " and number == len(run) - 1
                if not (char in _ZERO_WIDTH or last) and _visible(span.state):
                    moved = start + (x - start) * stretch - shift
                    self.chars.append(_letter(char, span, moved, y, advance * stretch))

    def _letters(
        self, element: ET.Element, spans: list[_Span], index: int
    ) -> list[tuple[str, int]]:
        """Return the characters of a span of a text, each with its span's index.

        ``spans`` holds the spans read so far, the span at ``index`` being that
        of ``element``, and the spans in it are added. A ``<tspan>``,
        ``<textPath>`` or ``<a>`` in it is a span of its own, raised by its
        baseline-shift, and the characters after it are those of the span
        that holds it.
        """
        span = spans[index]
        letters = [(char, index) for char in element.text or ""]
        for child in element:
            if child.tag in {"tspan", "textPath", "a"}:
                entered = self._entered(child, span.state)
                if entered is not None:
                    raised = _shift(
                        _declared(child).get("baseline-shift"), span.state.size
                    )
                    places = _places(child, entered)
                    spans.append(_Span(entered, places, span.rise + raised, index))
                    letters += self._letters(child, spans, len(spans) - 1)
            letters += [(char, index) for char in child.tail or ""]
        return letters

    def _image(self, element: ET.Element, state: _State) -> None:
        """Read an image that an ``<image>`` draws, on white, as it is shown.

        The image is a data URI or a file, named relative to the SVG file,
        fitted to the element's x, y, width and height, which default to its
        size in pixels, as its preserveAspectRatio places it, and clipped to
        them. It is turned to lie upright on the page, as matplotlib draws
        images upside down and turns them over.
        """
        pixels = _decoded(element.get("href", ""), self.folder)
        if pixels is None or not _visible(state):
            return
        rows, columns = pixels.shape[:2]
        width, height = state.viewport
        x = _length(element.get("x"), state.size, width) or 0
        y = _length(element.get("y"), state.size, height) or 0
        wide = _length(element.get("width"), state.size, width) or columns
        high = _length(element.get("height"), state.size, height) or rows
        preserve = element.get("preserveAspectRatio", "")
        fit = _fit((0, 0, columns, rows), wide, high, preserve)
        matrix = state.matrix @ _translation(x, y) @ fit
        corners = np.array([(0, 0), (columns, rows)], dtype=np.float64)
        box = _bounds(_mapped_all(matrix, corners))
        frame = np.array([(x, y), (x + wide, y + high)], dtype=np.float64)
        clip = _overlap(state.clip, _bounds(_mapped_all(state.matrix, frame)))
        if clip is None or _overlap(clip, box) is None:
            return
        along, down = matrix[:2, 0], matrix[:2, 1]  # across columns, and down rows
        if abs(along[0]) < abs(along[1]):
            pixels = pixels.swapaxes(0, 1)
            along, down = down, along
        if along[0] < 0:
            pixels = pixels[:, ::-1]
        if down[1] > 0:
            pixels = pixels[::-1]
        self.pictures.append(_Picture(_on_white(pixels, state.opacity), box, clip))


def _read(name: str) -> bytes:
    """Return the bytes of an SVG file, decompressed if compressed with gzip.

    Raises:
        ValueError: If the file cannot be read, or holds more than
            ``_MAX_SIZE`` bytes, decompressed.
    """
    try:
        with Path(name).open("rb") as file:
            data = file.read(_MAX_SIZE + 1)
            if data.startswith(b"\x1f\x8b"):
                file.seek(0)
                data = gzip.GzipFile(fileobj=file).read(_MAX_SIZE + 1)
    except (OSError, EOFError, zlib.error) as error:
        msg = f"cannot read SVG {name!r}: {getattr(error, 'strerror', None) or error}"
        raise ValueError(msg) from None
    if len(data) > _MAX_SIZE:
        msg = f"SVG {name!r} is larger than {_MAX_SIZE // 2**20} MiB"
        raise ValueError(msg)
    return data


def _parsed(data: bytes, name: str) -> ET.Element:
    """Return the root of an SVG file's XML, as ``_name`` names its elements.

    Comments in the root are elements too, tagged ``_COMMENT``, holding
    their text.

    Raises:
        ValueError: If the XML is not well formed, or declares entities.
    """
    builder = ET.TreeBuilder()
    parser = expat.ParserCreate(namespace_separator=" ")
    parser.buffer_text = True
    depth = 0  # how many elements are open

    def start(tag: str, attributes: dict[str, str]) -> None:
        nonlocal depth
        depth += 1
        builder.start(
            _name(tag), {_name(key): value for key, value in attributes.items()}
        )

    def end(tag: str) -> None:
        nonlocal depth
        depth -= 1
        builder.end(_name(tag))

    def comment(text: str) -> None:
        if depth:
            builder.start(_COMMENT, {})
            builder.data(text)
            builder.end(_COMMENT)

    def refuse(*_: object) -> None:
        msg = f"SVG {name!r} declares XML entities, which figrecover does not read"
        raise ValueError(msg)

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = builder.data
    parser.CommentHandler = comment
    parser.EntityDeclHandler = refuse
    try:
        parser.Parse(data, True)
    except expat.ExpatError as error:
        msg = f"cannot read SVG {name!r}: {error}"
        raise ValueError(msg) from None
    return builder.close()


def _name(name: str) -> str:
    """Return an element's or attribute's name without SVG's or XLink's namespace.

    Names in other namespaces keep theirs, as ``{namespace}name``.
    """
    namespace, _, local = name.rpartition(" ")
    return local if namespace in {"", _SVG, _XLINK} else f"{{{namespace}}}{local}"


def _size(root: ET.Element) -> tuple[float, float]:
    """Return the width and height of an SVG file's drawing, in points.

    They are those its root gives, in units of length or pixels, or else
    those of its viewBox, in pixels, or 300 by 150 pixels, as in CSS. A
    width or height in percent is not given, and one missing beside the
    other follows from the viewBox's shape.
    """
    width, height = _absolute(root.get("width")), _absolute(root.get("height"))
    view = _view_box(root)
    if view is not None:
        width = width or (height / view[3] if height else _POINTS["px"]) * view[2]
        height = height or width * view[3] / view[2]
    return width or _PAGE[0] * _POINTS["px"], height or _PAGE[1] * _POINTS["px"]


def _absolute(text: str | None) -> float | None:
    """Return a length in points, or None if it is not one in a unit of length.

    A number without a unit is in pixels.
    """
    match = _LENGTH.fullmatch((text or "").strip().lower())
    unit = (match[2] or "") if match else "%"
    return float(match[1]) * _POINTS[unit] if match and unit in _POINTS else None


def _view_box(element: ET.Element) -> tuple[float, float, float, float] | None:
    """Return an element's viewBox: its x, y, width and height, or None if none."""
    match _numbers(element.get("viewBox", "")):
        case [x, y, width, height] if width > 0 and height > 0:
            return x, y, width, height
    return None


def _fit(
    view: tuple[float, float, float, float] | None,
    width: float,
    height: float,
    preserve: str,
) -> npt.NDArray[np.float64]:
    """Return the matrix that fits a viewBox to a viewport of a width and height.

    It is fitted as preserveAspectRatio, such as ``xMidYMid meet``, the
    default, says: scaled alike across and up to fit within the viewport,
    or to cover it with slice, and aligned within it, or stretched to fill
    it with none. Without a viewBox, user units are left as they are.
    """
    if view is None:
        return np.eye(3)
    x, y, wide, high = view
    words = preserve.split() or ["xMidYMid"]
    x_scale, y_scale = width / wide, height / high
    if words[0] != "none":
        x_scale = y_scale = (max if "slice" in words else min)(x_scale, y_scale)
    x_shift = (width - wide * x_scale) * _ALIGN.get(words[0][1:4], 0.5)
    y_shift = (height - high * y_scale) * _ALIGN.get(words[0][5:8], 0.5)
    return np.array(
        [
            [x_scale, 0, x_shift - x * x_scale],
            [0, y_scale, y_shift - y * y_scale],
            [0, 0, 1],
        ]
    )


def _transform(text: str) -> npt.NDArray[np.float64]:
    """Return the matrix of a transform attribute, such as ``translate(1 2)``."""
    matrix = np.eye(3)
    for kind, values in _TRANSFORM.findall(text):
        matrix = matrix @ _transformation(kind, _numbers(values))
    return matrix


def _transformation(kind: str, values: list[float]) -> npt.NDArray[np.float64]:
    """Return the matrix of one transformation, such as rotate, with its values.

    One with the wrong number of values is left out, as the identity.
    """
    a, b, c, d, e, f = 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    match kind, values:
        case "matrix", [a, b, c, d, e, f]:
            pass
        case "translate", [e]:
            pass
        case "translate", [e, f]:
            pass
        case "scale", [a]:
            d = a
        case "scale", [a, d]:
            pass
        case "rotate", [angle, *centre] if len(centre) in {0, 2}:
            turn = math.radians(angle)
            a = d = round(math.cos(turn), 15)
            b = round(math.sin(turn), 15)
            c = -b
            x, y = centre or (0.0, 0.0)
            e, f = x - a * x - c * y, y - b * x - d * y
        case "skewX", [angle]:
            c = math.tan(math.radians(angle))
        case "skewY", [angle]:
            b = math.tan(math.radians(angle))
        case _:
            pass
    return np.array([[a, c, e], [b, d, f], [0, 0, 1]])


def _translation(x: float, y: float) -> npt.NDArray[np.float64]:
    """Return the matrix that moves points by x and y."""
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float64)


def _declared(element: ET.Element) -> dict[str, str]:
    """Return an element's attributes, with the properties its style declares.

    Declarations in the style attribute override presentation attributes.
    """
    declared = dict(element.attrib)
    for declaration in element.get("style", "").split(";"):
        key, _, value = declaration.partition(":")
        if value.strip():
            declared[key.strip().lower()] = value.replace("!important", "").strip()
    return declared


def _visible(state: _State) -> bool:
    """Return whether an element is shown, by its visibility."""
    return state.styles.get("visibility", "").strip() not in {"hidden", "collapse"}


def _numbers(text: str) -> list[float]:
    """Return the numbers in a list of them, such as a viewBox or a transform's."""
    return [float(number) for number in _NUMBER.findall(text)]


def _length(
    text: str | None, size: float, reference: float | None = None
) -> float | None:
    """Return a length, such as 12, 3pt or 1.5em, in user units, or None if none.

    Ems and exes are of the font size ``size``, and percentages of
    ``reference``; a percentage without a reference is no length.
    """
    match = _LENGTH.fullmatch((text or "").strip().lower())
    if match is None:
        return None
    value, unit = float(match[1]), match[2] or ""
    if unit == "%":
        return None if reference is None else value / 100 * reference
    if unit in {"em", "ex"}:
        return value * size * (1 if unit == "em" else _EX)
    return value * _PIXELS[unit]


def _lengths(text: str | None, size: float) -> list[float]:
    """Return a list of lengths, such as an x or dasharray, or none if one is wrong."""
    lengths = []
    for part in (text or "").replace(",", " ").split():
        length = _length(part, size)
        if length is None:
            return []
        lengths.append(length)
    return lengths


def _dashes(text: str | None) -> list[float]:
    """Return the lengths of a dash pattern's dashes and gaps, or none if solid.

    A pattern of an odd number of lengths is repeated, as SVG draws it, and
    one with a negative length, or none but zeros, is solid.
    """
    lengths = _lengths(text, _FONT_SIZE)
    if not lengths or min(lengths) < 0 or not sum(lengths):
        return []
    return lengths * (1 + len(lengths) % 2)


def _opacity(text: str | None) -> float:
    """Return an opacity, from 0 to 1, written as a number or percentage, or 1."""
    value = _length(text, 1, 1)
    return 1.0 if value is None else min(max(value, 0.0), 1.0)


def _colour(text: str) -> Colour | None:
    """Return a colour, such as #1f77b4, rgb(31, 119, 180) or steelblue.

    Colours that are wholly transparent, and anything else, such as none,
    give None.
    """
    text = text.strip().lower()
    if match := _HEX.fullmatch(text):
        digits = match[1] or "".join(2 * digit for digit in match[2])
        red, green, blue, *alpha = bytes.fromhex(digits)
        return None if alpha == [0] else (red, green, blue)
    if match := _RGB.fullmatch(text):
        parts = match[1].replace(",", " ").replace("/", " ").split()
        values = [_length(part, 1, 255) for part in parts]
        if len(values) not in {3, 4} or None in values:
            return None
        red, green, blue = (round(min(max(value or 0, 0), 255)) for value in values[:3])
        return None if values[3:] == [0] else (red, green, blue)
    return _NAMED.get(text)


def _reference(text: str) -> str:
    """Return the id that a reference such as ``#m1`` names, or none if not local."""
    return text.strip()[1:] if text.strip().startswith("#") else ""


def _outline(
    element: ET.Element, size: float, viewport: tuple[float, float]
) -> _Outline | None:
    """Return the outline of a shape in its own user space, or None if not a shape.

    The subpaths, closed flags and straight pieces are those of ``_Path``. A
    circle or ellipse passes through its leftmost, rightmost, highest and
    lowest points, as do its curves, and a rectangle's corners are square.
    Lengths in percent are of the viewport's width, height or diagonal. A
    rectangle, circle or ellipse without size, or a polygon without points,
    is no shape.
    """
    width, height = viewport
    diagonal = math.hypot(width, height) / math.sqrt(2)

    def length(name: str, reference: float) -> float:
        return _length(element.get(name), size, reference) or 0

    x, y = length("x", width), length("y", height)
    wide, high = length("width", width), length("height", height)
    centre = (length("cx", width), length("cy", height))
    across = length("rx", width) or length("r", diagonal)
    up = length("ry", height) or length("r", diagonal)
    values = _numbers(element.get("points", ""))
    points = list(zip(values[::2], values[1::2], strict=False))
    match element.tag:
        case "path":
            return _path(element.get("d", ""))
        case "rect" if wide > 0 and high > 0:
            corners = [(x, y), (x + wide, y), (x + wide, y + high), (x, y + high)]
            return _polygon(corners, closed=True)
        case "circle" | "ellipse" if across > 0 and up > 0:
            ends = [
                (centre[0] + across, centre[1]),
                (centre[0], centre[1] + up),
                (centre[0] - across, centre[1]),
                (centre[0], centre[1] - up),
            ]
            return [np.array(ends)], [True], np.empty((0, 4))
        case "line":
            ends = [(length("x1", width), length("y1", height))]
            return _polygon([*ends, (length("x2", width), length("y2", height))])
        case "polyline" | "polygon" if points:
            return _polygon(points, closed=element.tag == "polygon")
        case _:
            return None


def _polygon(points: list[Point], *, closed: bool = False) -> _Outline:
    """Return the outline of a polygon, or a line through points if not closed."""
    corners = np.array(points, dtype=np.float64)
    ends = np.vstack([corners, corners[:1]]) if closed else corners
    return [corners], [closed], np.hstack([ends[:-1], ends[1:]])


def _pieces(data: str) -> Iterator[tuple[str, list[float]]]:
    """Yield the commands of a path's data, each with its values.

    A command's values can be repeated without the command, and a move's as
    lines. The data is read up to the first error, as SVG draws it.
    """
    position, command = 0, ""
    while True:
        match = _COMMAND.match(data, position)
        if match:
            command, position = match[1], match.end()
            if command in "Zz":
                yield "Z", []
                continue
        elif command in {"", "Z", "z"}:
            return
        values = []
        for index in range(_ARGUMENTS[command.upper()]):
            pattern = _FLAG if command in "Aa" and index in {3, 4} else _ARGUMENT
            found = pattern.match(data, position)
            if found is None:
                return
            values.append(float(found[1]))
            position = found.end()
        yield command, values
        command = {"M": "L", "m": "l"}.get(command, command)


def _path(data: str) -> _Outline | None:
    """Return the outline of a path's data, or None if it draws nothing.

    Each subpath passes through the ends of its pieces, without the control
    points of curves, and through the points of arcs that ``_arc`` gives.
    Data that does not start with a move draws nothing.
    """
    subpaths: list[list[Point]] = []
    closed: list[bool] = []
    segments: list[tuple[float, float, float, float]] = []
    current = start = (0.0, 0.0)
    for command, values in _pieces(data):
        kind = command.upper()
        if not subpaths and kind != "M":
            break
        end = start if kind == "Z" else _end(command, values, current)
        if kind == "M":
            start = end
            subpaths.append([end])
            closed.append(False)
        elif kind == "Z":
            closed[-1] = True
            segments.append((*current, *start))
        else:
            if closed[-1]:  # drawing on after a close starts where it closed
                subpaths.append([start])
                closed.append(False)
            if kind == "A" and values[0] and values[1]:
                subpaths[-1] += _arc(current, values[:5], end)
            else:
                if kind in "LHVA":
                    segments.append((*current, *end))
                subpaths[-1].append(end)
        current = end
    if not subpaths:
        return None
    return (
        [np.array(subpath, dtype=np.float64) for subpath in subpaths],
        closed,
        np.array(segments, dtype=np.float64).reshape(-1, 4),
    )


def _end(command: str, values: list[float], current: Point) -> Point:
    """Return where a piece of a path ends, from the current point.

    The values of a command written in lower case are relative to the
    current point.
    """
    x, y = current if command.islower() else (0.0, 0.0)
    kind = command.upper()
    if kind == "H":
        return x + values[0], current[1]
    if kind == "V":
        return current[0], y + values[0]
    return x + values[_ENDS[kind]], y + values[_ENDS[kind] + 1]


def _arc(start: Point, values: list[float], end: Point) -> list[Point]:
    """Return the points that an arc from a start passes through, to its end.

    ``values`` are its radii, the rotation of its ellipse in degrees, and its
    large-arc and sweep flags. It passes through the points at each quarter
    turn of its ellipse, as the curves that draw it end there, which for an
    unrotated ellipse are its leftmost, rightmost, highest and lowest. Radii
    too small to reach the end are enlarged, as SVG does.
    """
    if start == end:
        return []
    across, up, rotation, large, sweep = values
    cos, sin = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
    half_x, half_y = (start[0] - end[0]) / 2, (start[1] - end[1]) / 2
    x, y = cos * half_x + sin * half_y, cos * half_y - sin * half_x
    across, up = abs(across), abs(up)
    excess = max(1, math.sqrt((x / across) ** 2 + (y / up) ** 2))
    across, up = across * excess, up * excess
    squares = (across * y) ** 2 + (up * x) ** 2
    factor = math.sqrt(max(0, (across * up) ** 2 - squares) / squares)
    factor = -factor if large == sweep else factor
    centre_x, centre_y = factor * across * y / up, -factor * up * x / across
    middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
    centre = (
        cos * centre_x - sin * centre_y + middle[0],
        sin * centre_x + cos * centre_y + middle[1],
    )
    first = math.atan2((y - centre_y) / up, (x - centre_x) / across)
    last = math.atan2((-y - centre_y) / up, (-x - centre_x) / across)
    turn = (last - first) % math.tau
    turn = turn if sweep else turn - math.tau
    quarter = math.pi / 2
    if turn > 0:
        quarters = range(
            math.floor(first / quarter) + 1, math.ceil((first + turn) / quarter)
        )
    else:
        quarters = range(
            math.ceil(first / quarter) - 1, math.floor((first + turn) / quarter), -1
        )
    points = [
        (
            centre[0] + across * math.cos(angle) * cos - up * math.sin(angle) * sin,
            centre[1] + across * math.cos(angle) * sin + up * math.sin(angle) * cos,
        )
        for angle in (number * quarter for number in quarters)
    ]
    return [*points, end]


def _mapped_all(
    matrix: npt.NDArray[np.float64], points: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Return points, as rows of x and y, mapped by a matrix."""
    return np.asarray(points @ matrix[:2, :2].T + matrix[:2, 2], dtype=np.float64)


def _bounds(points: npt.NDArray[np.float64]) -> Box:
    """Return the box around points, as rows of x and y."""
    left, bottom = (float(value) for value in points.min(axis=0))
    right, top = (float(value) for value in points.max(axis=0))
    return left, bottom, right, top


def _scale(matrix: npt.NDArray[np.float64]) -> float:
    """Return how much a matrix enlarges lengths, on average across and up."""
    return math.sqrt(abs(float(np.linalg.det(matrix[:2, :2]))))


def _angle(matrix: npt.NDArray[np.float64]) -> float:
    """Return the angle in degrees, anticlockwise from across, of a matrix's x-axis."""
    return round(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])), 9)


def _collapsed(letters: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Return a text's characters with its white space collapsed, as SVG lays it out.

    Line breaks are left out, tabs become spaces, spaces at either end are
    left out, and spaces in a row become one.
    """
    kept: list[tuple[str, int]] = []
    for char, index in letters:
        space = " " if char == "\t" else char
        if space in "\r\n" or (space == " " and (not kept or kept[-1][0] == " ")):
            continue
        kept.append((space, index))
    return kept[:-1] if kept and kept[-1][0] == " " else kept


def _place(
    spans: list[_Span], starts: dict[int, int], index: int, number: int, name: str
) -> float | None:
    """Return the x, y, dx or dy of a character of a text, or None if none.

    It is that of the innermost span holding the character whose list of
    them reaches it, counting from the first character the span holds.
    """
    span: int | None = index
    while span is not None:
        values = spans[span].places[name]
        if number - starts[span] < len(values):
            return values[number - starts[span]]
        span = spans[span].holder
    return None


def _places(element: ET.Element, state: _State) -> dict[str, list[float]]:
    """Return the x, y, dx and dy lists of a ``<text>`` or ``<tspan>``."""
    return {
        name: _lengths(element.get(name), state.size) for name in ("x", "y", "dx", "dy")
    }


def _placed(spans: list[_Span], letters: list[tuple[str, int]]) -> list[list[_Letter]]:
    """Return the characters of a text placed, in runs placed after one another.

    Each character lies where the x and y of the innermost span holding it
    that has one for it places it, or else after the one before, by its
    estimated width, moved by such a span's dx and dy. A character placed
    by an x or y starts a run.
    """
    starts: dict[int, int] = {}  # the index of each span's first character
    for number, (_, index) in enumerate(letters):
        holder: int | None = index
        while holder is not None and holder not in starts:
            starts[holder] = number
            holder = spans[holder].holder
    runs: list[list[_Letter]] = []
    x = y = 0.0
    for number, (char, index) in enumerate(letters):
        places = {
            name: _place(spans, starts, index, number, name)
            for name in ("x", "y", "dx", "dy")
        }
        if not runs or places["x"] is not None or places["y"] is not None:
            runs.append([])
        x = (x if places["x"] is None else places["x"]) + (places["dx"] or 0)
        y = (y if places["y"] is None else places["y"]) + (places["dy"] or 0)
        advance = 0 if char in _ZERO_WIDTH else _advance(char) * spans[index].state.size
        runs[-1].append((char, spans[index], x, y, advance))
        x += advance
    return runs


def _shift(text: str | None, size: float) -> float:
    """Return how far a baseline-shift raises text, in user units.

    Super and sub raise and lower it by ``_SUPER`` and ``_SUB`` of the font
    size of the line it is on, ``size``, and lengths and percentages of it
    by as much.
    """
    text = (text or "").strip()
    if text in {"super", "sub"}:
        return (_SUPER if text == "super" else _SUB) * size
    return _length(text, size, size) or 0


def _advance(char: str) -> float:
    """Return the estimated width of a character, in font sizes."""
    if char in _NARROW:
        return 0.3
    if char in _WIDE:
        return 0.9
    return 0.7 if char.isupper() else 0.6


def _letter(char: str, span: _Span, x: float, y: float, advance: float) -> _Char:
    """Return a character of a text, as ``_Char`` describes it, placed at x and y.

    Its box runs over its advance, and from its baseline, raised by its
    span's rise, as far up as ``_INK`` gives, or ``_DESCENDING`` or
    ``_BRACKETS`` for characters that descend below their line.
    """
    matrix, size = span.state.matrix, span.state.size
    base = y - span.rise
    if char in _BRACKETED:
        ink = _BRACKETS
    elif char in _DESCENDERS:
        ink = _DESCENDING
    else:
        ink = _INK
    low, high = (base - edge * size for edge in ink)
    corners = np.array([(x, low), (x + advance, low), (x, high), (x + advance, high)])
    along = float(np.hypot(*matrix[:2, 0]))
    return _Char(
        char,
        _bounds(_mapped_all(matrix, corners)),
        _mapped(matrix, (x, base)),
        size * _scale(matrix),
        _angle(matrix),
        advance * along,
    )


def _plain(text: str) -> str:
    """Return text written with matplotlib's mathtext, between dollar signs, as plain.

    Text with an odd number of dollar signs, or none, is plain already.
    """
    pieces = _DOLLAR.split(text)
    if len(pieces) % 2 == 0 or len(pieces) == 1:
        return text
    plain = (
        _tex(_TEX.findall(piece)[::-1]) if number % 2 else piece
        for number, piece in enumerate(pieces)
    )
    return "".join(plain).replace("\\$", "$")


def _tex(tokens: list[str]) -> str:
    """Return the plain text of mathtext, up to the brace that closes its group.

    ``tokens`` are the commands and characters that remain, the next last,
    and those read are taken off. Superscripts and subscripts are written
    as Unicode has them, where it does, and minus signs as such.
    """
    parts: list[str] = []
    while tokens and (part := tokens.pop()) != "}":
        if part in {"^", "_"}:
            script = _atom(tokens)
            if part == "_":
                parts.append(script.translate(_SUBSCRIPTS))
            else:
                parts.append(
                    "\u00b0" if script == "\u2218" else script.translate(_SUPERSCRIPTS)
                )
        else:
            tokens.append(part)
            parts.append(_atom(tokens))
    return "".join(parts)


def _atom(tokens: list[str]) -> str:
    """Return the plain text of the next group, command or character of mathtext."""
    part = tokens.pop() if tokens else ""
    if part == "{":
        return _tex(tokens)
    if part.startswith("\\"):
        name = part[1:]
        if name in _UNARY:
            return _atom(tokens)
        if name == "frac":
            return f"{_atom(tokens)}/{_atom(tokens)}"
        if name == "sqrt":
            return "\u221a" + _atom(tokens)
        return _SYMBOLS.get(name, name)
    return "\u2212" if part == "-" else part.strip()


def _decoded(href: str, folder: Path) -> npt.NDArray[Any] | None:
    """Return the pixels of an image given as a data URI or a file, or None.

    A file is named relative to ``folder``, and a URL of another kind, such
    as a web address, gives None, as does anything that is not an image.
    """
    if href.startswith("data:"):
        header, _, payload = href.partition(",")
        try:
            data = (
                base64.b64decode(payload)
                if header.endswith(";base64")
                else unquote_to_bytes(payload)
            )
        except binascii.Error:
            return None
    elif href and not urlsplit(href).scheme:
        try:
            data = (folder / unquote(href)).read_bytes()
        except OSError:
            return None
    else:
        return None
    if not data:
        return None
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return None if pixels is None else np.asarray(pixels)


def _on_white(pixels: npt.NDArray[Any], opacity: float) -> Image:
    """Return pixels as OpenCV decodes them, in red, green and blue, on white.

    Their transparency, if any, is taken with an opacity.
    """
    match pixels.shape:
        case (_, _, 4):  # blue, green, red and alpha
            shown = pixels
        case (_, _, 3):  # blue, green and red
            shown = cv2.cvtColor(pixels, cv2.COLOR_BGR2BGRA)
        case _:  # grey
            shown = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGRA)
    values = shown.astype(np.float64) / np.iinfo(pixels.dtype.str).max
    alpha = values[..., 3:] * opacity
    on_white = values[..., 2::-1] * alpha + (1 - alpha)
    return np.round(on_white * 255).astype(np.uint8)


def _write(view: Image, text: _Text) -> None:
    """Write a line of text in its box, a view of a canvas, as ``render_svg`` does.

    Text running up or down is written across and turned, and slanted text
    is drawn as the outline of its box. Minus signs are written as dashes,
    superscripts after a caret, and other characters than ASCII ones as
    question marks. Text in a box less than a pixel high or wide is left out.
    """
    rows, columns = view.shape[:2]
    upright = abs(abs(text.slant) - 90) < 1
    if not (rows and columns):
        return
    if not (upright or abs(text.slant) < 1):
        cv2.rectangle(view, (0, 0), (columns - 1, rows - 1), (128, 128, 128), 1)
        return
    plain = _SUPERSCRIPT.sub(r"^\1", text.text).translate(_PLAIN)
    plain = unicodedata.normalize("NFKD", plain)
    plain = plain.encode("ascii", "replace").decode("ascii")
    across, tall = (rows, columns) if upright else (columns, rows)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (width, height), base = cv2.getTextSize(plain, font, 1, 1)
    scale = min(across / width, tall / (height + base))
    patch: Image = np.full((tall, across, 3), 255, dtype=np.uint8)
    corner = (0, tall - round(base * scale))
    cv2.putText(patch, plain, corner, font, scale, (0, 0, 0), 1, cv2.LINE_AA)
    if upright:
        turn = (
            cv2.ROTATE_90_COUNTERCLOCKWISE
            if text.slant > 0
            else cv2.ROTATE_90_CLOCKWISE
        )
        patch = np.asarray(cv2.rotate(patch, turn), dtype=np.uint8)
    np.minimum(view, patch, out=view)
