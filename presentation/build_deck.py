"""Builds the 5-slide design review deck: python presentation/build_deck.py

This is a design-stage deck: it presents the problem, data, architecture and evaluation plan,
not model results (those belong in the final demo).
"""
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "presentation" / "FairPrice_Design_Review.pptx"
MAP_FIGURE = ROOT / "reports" / "figures" / "05_geographic_value_map.png"

# Palette: deep slate (dominant), Californian terracotta (accent), teal (support)
NAVY = RGBColor(0x1B, 0x2F, 0x3A)
NAVY_LIGHT = RGBColor(0x27, 0x42, 0x50)
TERRA = RGBColor(0xC8, 0x55, 0x3D)
TEAL = RGBColor(0x2A, 0x7F, 0x7A)
INK = RGBColor(0x1E, 0x25, 0x2B)
MUTED = RGBColor(0x5F, 0x6B, 0x73)
CARD = RGBColor(0xF1, 0xF4, 0xF5)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ICE = RGBColor(0xC9, 0xD6, 0xDD)
HEAD, BODY = "Cambria", "Calibri"

TEAM_NAME = "Linear Regression"
TEAM = [
    ("Hardik", "20234067", "B.Tech (ECE)"),
    ("Kartikay Singh", "20238013", "B.Tech (ECM)"),
    ("Rahul Kumar Srivastava", "20235137", "B.Tech (EE)"),
    ("Chandan Kumar Gupta", "20238005", "B.Tech (ECM)"),
]


# ---------------------------------------------------------------- helpers
def text(slide, x, y, w, h, content, size=15, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, spacing_after=0, name=None):
    """content: a string, or a list of paragraphs; a paragraph is a string or a list of (text, overrides)."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        box.name = name
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = anchor
    paragraphs = content if isinstance(content, list) else [content]
    for i, para in enumerate(paragraphs):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.alignment = align
        p.space_after = Pt(spacing_after)
        runs = para if isinstance(para, list) else [(para, {})]
        for run_text, opts in runs:
            run = p.add_run()
            run.text = run_text
            f = run.font
            f.name = opts.get("font", font)
            f.size = Pt(opts.get("size", size))
            f.bold = opts.get("bold", bold)
            f.italic = opts.get("italic", italic)
            f.color.rgb = opts.get("color", color)
    return box


def card(slide, x, y, w, h, fill=CARD, radius=0.06, name=None):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = radius
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    if name:
        shape.name = name
    return shape


def badge(slide, x, y, label, fill=TERRA, size=0.42, font_size=14):
    """Numbered circle: the deck's recurring motif."""
    circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(size), Inches(size))
    circle.fill.solid()
    circle.fill.fore_color.rgb = fill
    circle.line.fill.background()
    circle.shadow.inherit = False
    frame = circle.text_frame
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = frame.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = label
    run.font.size, run.font.bold, run.font.name = Pt(font_size), True, BODY
    run.font.color.rgb = WHITE
    return circle


def arrow(slide, x, y, w=0.32, h=0.26, color=MUTED):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def background(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def title(slide, heading, kicker):
    text(slide, 0.6, 0.45, 12, 0.3, kicker.upper(), size=11, color=TERRA, bold=True, name="Kicker")
    text(slide, 0.6, 0.75, 12, 0.7, heading, size=32, color=NAVY, bold=True, font=HEAD, name="Title")


def footer(slide, number):
    text(slide, 0.6, 7.0, 8, 0.25, f"FairPrice  |  Team {TEAM_NAME}  |  Design review", size=10, color=MUTED)
    text(slide, 11.73, 7.0, 1.0, 0.25, str(number), size=10, color=MUTED, align=PP_ALIGN.RIGHT)


# ---------------------------------------------------------------- slides
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, NAVY)
    text(s, 0.7, 1.0, 6.8, 0.3, "DATA SCIENCE HACKATHON  |  DESIGN REVIEW", size=12, color=TERRA, bold=True)
    text(s, 0.7, 1.55, 6.8, 1.1, "FairPrice", size=60, color=WHITE, bold=True, font=HEAD)
    text(s, 0.7, 2.75, 6.4, 1.3, "Predicting California house values, explaining what drives them, "
         "and telling buyers whether an asking price is fair.", size=20, color=ICE)
    text(s, 0.7, 4.55, 6.4, 0.9,
         "Regression  |  Conventional ML + Neural Network  |  Streamlit app",
         size=14, color=ICE)

    card(s, 7.9, 1.0, 4.8, 5.5, fill=NAVY_LIGHT, radius=0.05, name="Team card")
    text(s, 8.3, 1.35, 4.0, 0.3, "TEAM", size=11, color=TERRA, bold=True)
    text(s, 8.3, 1.65, 4.0, 0.5, TEAM_NAME, size=24, color=WHITE, bold=True, font=HEAD)
    for i, (name, sid, branch) in enumerate(TEAM):
        y = 2.5 + i * 0.95
        badge(s, 8.3, y + 0.04, str(i + 1), fill=TERRA, size=0.42)
        text(s, 8.95, y, 3.6, 0.35, name, size=16, color=WHITE, bold=True)
        text(s, 8.95, y + 0.36, 3.6, 0.3, f"ID {sid}  |  {branch}", size=13, color=ICE)
    s.notes_slide.notes_text_frame.text = (
        "Introduce the team. One line on the idea: we estimate what homes in a California "
        "neighbourhood are worth, explain the estimate, and check asking prices against it.")


def slide_problem(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, WHITE)
    title(s, "Is this house priced fairly?", "Problem statement and business use case")

    text(s, 0.6, 1.75, 5.6, 1.6, [
        "Most buyers and sellers rely on gut feeling or an agent's opinion to judge a price. "
        "There is rarely a quick, transparent way to check it against data.",
        [("Our goal: ", {"bold": True, "color": NAVY}),
         ("build a model that predicts a neighbourhood's median house value from census data, "
          "and is honest about how sure it is.", {})],
    ], size=15, color=INK, spacing_after=10)

    steps = [("Predict", "Estimate the median home value from income, age, size, crowding and location."),
             ("Explain", "Show which factors pushed the estimate up or down (SHAP)."),
             ("Judge", "Compare an asking price with the estimate: good deal, fair or overpriced.")]
    for i, (head, body) in enumerate(steps):
        y = 3.75 + i * 0.95
        badge(s, 0.6, y, str(i + 1), fill=NAVY)
        text(s, 1.2, y - 0.02, 5.0, 0.3, head, size=16, color=NAVY, bold=True)
        text(s, 1.2, y + 0.3, 5.0, 0.55, body, size=14, color=MUTED)

    text(s, 6.9, 1.75, 5.8, 0.3, "WHO IT HELPS", size=11, color=TERRA, bold=True)
    uses = [("Home buyers", "Avoid overpaying with a data-backed second opinion before making an offer."),
            ("Sellers and agents", "Set a realistic listing price and back it up with clear reasons."),
            ("Banks and lenders", "Sanity-check property valuations before approving a home loan."),
            ("Property platforms", "Automatically flag listings that look far above the market.")]
    for i, (head, body) in enumerate(uses):
        col, row = i % 2, i // 2
        x, y = 6.9 + col * 2.95, 2.15 + row * 2.35
        card(s, x, y, 2.8, 2.15, name=f"Use case {i + 1}")
        badge(s, x + 0.25, y + 0.25, str(i + 1), fill=TERRA, size=0.4, font_size=13)
        text(s, x + 0.25, y + 0.8, 2.35, 0.35, head, size=16, color=NAVY, bold=True)
        text(s, x + 0.25, y + 1.17, 2.35, 0.9, body, size=13, color=INK)
    footer(s, 2)
    s.notes_slide.notes_text_frame.text = (
        "Problem: pricing is opaque. We predict, explain and judge. The business value is a fast, "
        "transparent second opinion for buyers, sellers, lenders and listing platforms.")


def slide_dataset(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, WHITE)
    title(s, "California Housing, 1990 census", "Dataset")

    stats = [("20,640", "neighbourhoods (census block groups)"),
             ("8", "input features + 1 target"),
             ("992", "rows with the $500k price cap, removed")]
    for i, (value, label) in enumerate(stats):
        x = 0.6 + i * 2.15
        text(s, x, 1.75, 2.0, 0.75, value, size=36, color=TERRA if i == 2 else NAVY, bold=True, font=HEAD)
        text(s, x, 2.5, 1.95, 0.6, label, size=12, color=MUTED)

    card(s, 0.6, 3.3, 6.25, 1.5, name="Features card")
    text(s, 0.85, 3.45, 5.8, 0.3, "Features", size=14, color=NAVY, bold=True)
    text(s, 0.85, 3.8, 5.8, 0.95,
         "Median income, house age, average rooms, average bedrooms, population, average occupancy, "
         "latitude, longitude.  Target: median house value (in $100,000s).", size=13, color=INK)

    text(s, 0.6, 5.0, 6.2, 0.3, "WHAT THE DATA TOLD US", size=11, color=TERRA, bold=True)
    findings = [
        "Prices are capped at $500k, so those rows have no true value and are dropped.",
        "Extreme outliers (up to 1,243 occupants per home) are clipped, not deleted.",
        "Rooms and bedrooms are 0.85 correlated, so we use a bedroom ratio instead.",
    ]
    for i, line in enumerate(findings):
        y = 5.35 + i * 0.48
        badge(s, 0.6, y, str(i + 1), fill=NAVY, size=0.32, font_size=11)
        text(s, 1.05, y + 0.02, 5.8, 0.45, line, size=13, color=INK)

    if MAP_FIGURE.exists():
        pic = s.shapes.add_picture(str(MAP_FIGURE), Inches(7.25), Inches(1.65), height=Inches(4.95))
        pic.name = "Value map"
        text(s, 7.25, 6.65, 5.5, 0.3, "Coastal neighbourhoods near LA and the Bay Area are the most expensive.",
             size=11, color=MUTED, italic=True)
    footer(s, 3)
    s.notes_slide.notes_text_frame.text = (
        "Each row is a neighbourhood, not a house, and prices are in 1990 dollars. Three findings drove "
        "our cleaning: the price cap, extreme outliers, and redundant room features.")


def slide_architecture(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, WHITE)
    title(s, "How we are solving it", "Architecture and approach")

    stages = [("Ingest", "Load data,\ncheck schema"),
              ("Clean", "Range rules,\ndrop capped rows"),
              ("Explore", "Distributions,\ncorrelations, map"),
              ("Engineer", "Ratios, distance\nto nearest city"),
              ("Train", "5 models, tuning,\n5-fold CV"),
              ("Serve", "Best model + SHAP\nin Streamlit")]
    w, gap = 1.68, 0.38
    for i, (head, body) in enumerate(stages):
        x = 0.6 + i * (w + gap)
        card(s, x, 1.75, w, 1.45, fill=NAVY if i in (4, 5) else CARD, name=f"Stage {head}")
        dark = i in (4, 5)
        text(s, x + 0.18, 1.9, w - 0.3, 0.35, head, size=16, bold=True, color=WHITE if dark else NAVY)
        text(s, x + 0.18, 2.3, w - 0.3, 0.8, body, size=12, color=ICE if dark else MUTED)
        if i < len(stages) - 1:
            arrow(s, x + w + 0.06, 2.35, w=0.26)

    text(s, 0.6, 3.55, 6.0, 0.3, "TWO MODEL FAMILIES, ONE PIPELINE", size=11, color=TERRA, bold=True)
    card(s, 0.6, 3.9, 6.0, 1.25, name="Conventional ML")
    text(s, 0.85, 4.02, 5.5, 0.3, "Conventional ML", size=15, color=NAVY, bold=True)
    text(s, 0.85, 4.38, 5.5, 0.7, "Linear Regression (baseline), Ridge, Random Forest, XGBoost. "
         "Tuned with randomized search.", size=13, color=INK)
    card(s, 0.6, 5.3, 6.0, 1.35, name="Neural network")
    text(s, 0.85, 5.42, 5.5, 0.3, "Artificial Neural Network (Keras)", size=15, color=NAVY, bold=True)
    text(s, 0.85, 5.78, 5.5, 0.8, "Dense layers with ReLU/ELU, He/Glorot init, dropout and L2. "
         "Adam or SGD, early stopping. Same features, same CV.", size=13, color=INK)

    text(s, 7.1, 3.55, 5.6, 0.3, "KEY DESIGN DECISIONS", size=11, color=TERRA, bold=True)
    decisions = [
        ("No data leakage", "Imputation, clipping and scaling are fitted on training data only."),
        ("Same pipeline in the app", "The saved model includes its preprocessing, so the app sees "
                                     "data exactly as training did."),
        ("Honest verdicts", "\"Fair\" means within the model's typical error, not an exact match."),
    ]
    for i, (head, body) in enumerate(decisions):
        y = 3.95 + i * 0.92
        badge(s, 7.1, y, str(i + 1), fill=TEAL, size=0.38, font_size=12)
        text(s, 7.65, y - 0.02, 5.0, 0.3, head, size=15, color=NAVY, bold=True)
        text(s, 7.65, y + 0.3, 5.0, 0.55, body, size=13, color=MUTED)
    footer(s, 4)
    s.notes_slide.notes_text_frame.text = (
        "Walk the six stages left to right. Both model families share one preprocessing pipeline, so the "
        "comparison is fair. The best model is chosen on cross-validation error, not on the test set.")


def slide_stack(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    background(s, WHITE)
    title(s, "Tech stack and evaluation plan", "Tools and success criteria")

    stack = [("Language and data", "Python, pandas, NumPy"),
             ("Conventional ML", "scikit-learn, XGBoost"),
             ("Neural network", "TensorFlow, Keras"),
             ("Explainability", "SHAP"),
             ("Visualisation", "Matplotlib, Seaborn"),
             ("App and quality", "Streamlit, pytest, Git")]
    for i, (head, body) in enumerate(stack):
        col, row = i % 2, i // 2
        x, y = 0.6 + col * 3.05, 1.75 + row * 1.35
        card(s, x, y, 2.9, 1.2, name=f"Stack {head}")
        text(s, x + 0.22, y + 0.18, 2.5, 0.3, head, size=12, color=TERRA, bold=True)
        text(s, x + 0.22, y + 0.52, 2.5, 0.6, body, size=15, color=NAVY, bold=True)

    text(s, 6.95, 1.75, 5.8, 0.3, "HOW WE WILL MEASURE SUCCESS", size=11, color=TERRA, bold=True)
    metrics = [("RMSE", "Main selection metric. Punishes large misses."),
               ("MAE", "Typical error in dollars. Sets the fair-price band."),
               ("R²", "Share of price variation the model explains.")]
    for i, (name, body) in enumerate(metrics):
        x = 6.95 + i * 1.98
        card(s, x, 2.1, 1.82, 1.75, name=f"Metric {name}")
        text(s, x + 0.2, 2.25, 1.45, 0.55, name, size=26, color=NAVY, bold=True, font=HEAD)
        text(s, x + 0.2, 2.85, 1.45, 0.95, body, size=12, color=INK)

    text(s, 6.95, 4.15, 5.8, 0.3, "HOW WE WILL PICK THE MODEL", size=11, color=TERRA, bold=True)
    steps = ["Linear Regression sets the baseline every other model has to beat.",
             "All five models are tuned and scored with the same 5-fold cross-validation.",
             "Lowest cross-validation error wins. The held-out test set is used once, for the final score."]
    for i, line in enumerate(steps):
        y = 4.5 + i * 0.62
        badge(s, 6.95, y, str(i + 1), fill=TEAL, size=0.36, font_size=12)
        text(s, 7.5, y + 0.02, 5.25, 0.6, line, size=13, color=INK)
    text(s, 6.95, 6.42, 5.8, 0.45, "Success means clearly beating the baseline, stable scores across folds, "
         "and predictions we can explain.", size=12, color=MUTED, italic=True)
    footer(s, 5)
    s.notes_slide.notes_text_frame.text = (
        "Left: the tools for each part of the pipeline. Right: how we will judge the models. RMSE picks the "
        "winner, MAE becomes the fair-price band in the app, and R² is easy to explain. Every model, including "
        "the neural network, is compared under the same cross-validation, and the test set is touched only once.")


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    for build in (slide_title, slide_problem, slide_dataset, slide_architecture, slide_stack):
        build(prs)
    prs.core_properties.title = "FairPrice - Design Review"
    prs.core_properties.author = f"Team {TEAM_NAME}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
