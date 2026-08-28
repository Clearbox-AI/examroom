from benchsmith.authoring.openstax import (
    flatten_math,
    parse_quantity,
    sigfig_tolerance,
    split_parts,
    strip_html,
)
from benchsmith.scoring import first_number

MATHML = (
    "<math display=\"inline\"><semantics><mrow><mn>3</mn><mtext>.</mtext>"
    "<mrow><mtext>0 </mtext><mo stretchy=\"false\">×</mo>"
    "<msup><mtext>10</mtext><mrow><mn>4</mn></mrow></msup></mrow>"
    "<mtext> m/s</mtext></mrow>"
    "<annotation-xml encoding=\"MathML-Content\"><mn>999</mn></annotation-xml>"
    "</semantics></math>"
)


def test_flatten_math_drops_annotation_and_renders_sup():
    assert flatten_math(MATHML) == "3.0 ×10^4 m/s"


def test_strip_html_inlines_math():
    txt = strip_html(f"<p>speed is {MATHML} here</p>")
    assert "10^4 m/s" in txt and "999" not in txt


def test_split_parts():
    assert split_parts("(a) 7 m (b) 8 m") == {"a": "7 m", "b": "8 m"}
    assert split_parts("just 7 m") == {"": "just 7 m"}
    # a stray out-of-order "(a)" reference inside part text is not a new part
    assert list(split_parts("(a) x (b) same as (a) plus 1")) == ["a", "b"]


def test_parse_quantity():
    q = parse_quantity("3. 0 × 10^4 m/s")
    assert q.value == 3.0e4 and q.unit == "m/s" and q.sigfigs == 2
    q = parse_quantity("384,000 km")
    assert q.value == 384000 and q.sigfigs == 3
    q = parse_quantity("v_0 = 4.95 m/s")
    assert q.value == 4.95 and q.unit == "m/s"
    q = parse_quantity("2 × 10⁷ years")
    assert q.value == 2e7 and q.unit == "years"
    q = parse_quantity("-2.50 m/s^2")
    assert q.value == -2.50 and q.sigfigs == 3
    q = parse_quantity("34.3 km/h, 25º S of E.")
    assert q.value == 34.3 and q.unit == "km/h" and q.rest
    assert parse_quantity("It does not make sense.") is None
    assert parse_quantity("v>v_max") is None


def test_sigfig_tolerance():
    assert sigfig_tolerance(parse_quantity("40.0 km/h")) == 0.05
    assert sigfig_tolerance(parse_quantity("384,000 km")) == 500.0
    assert sigfig_tolerance(parse_quantity("0 m/s")) == 0.0


def test_first_number_scientific_forms():
    assert first_number("3.0 × 10^4 m/s") == 3.0e4
    assert first_number("3.0 x 10^4") == 3.0e4
    assert first_number("about 2 × 10⁷ years") == 2e7
    assert first_number("10^7 years") == 1e7
    assert first_number("1.5e-3 m") == 1.5e-3
    assert first_number("−2.50 m/s^2") == -2.50
    assert first_number("27 cm/s") == 27
    assert first_number("first 3 then 10") == 3          # no false ×10 join
    assert first_number("13 x 10-inch boards") == 13     # exponent needs ^
    assert first_number("no numbers") is None
