import streamlit as st
from pathlib import Path
from src.utils.model_loader import load_models_and_columns
from streamlit_folium import st_folium
import folium
import numpy as np
import pandas as pd

# Set up static file path
STATIC_DIR = Path(__file__).parent / "static"

# --- PAGE CONFIG (must be first Streamlit command) ---
st.set_page_config(
    page_title="House Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Link external CSS file
st.markdown(
    "<style>" + Path(STATIC_DIR / "streamlit_styles.css").read_text() + "</style>",
    unsafe_allow_html=True,
)

# Extra CSS to force dark text & visible sidebar toggle
custom_css = """
<style>
/* Make all text dark and readable */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] *,
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
    color: #111827 !important; /* dark slate */
}

/* ---- SIDEBAR TOGGLE BUTTON: MAKE IT BIG & VISIBLE ---- */
[data-testid="collapsedControl"] {
    opacity: 1 !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    width: 42px !important;
    height: 42px !important;
    color: #0f172a !important;      /* dark icon */
    background: #ffffff !important; /* white pill */
    border-radius: 999px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.30) !important;
    border: 1px solid rgba(148, 163, 184, 0.85) !important;
    cursor: pointer !important;
    z-index: 9999 !important;
}

/* Make sure the inner button inherits the styles */
[data-testid="collapsedControl"] > button {
    background: transparent !important;
    border: none !important;
    color: inherit !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- rest of your existing code stays exactly as it is below this line ---


# ---- rest of your existing code stays below here ----


st.markdown(
    "<style>" + Path(STATIC_DIR / "streamlit_styles.css").read_text() + "</style>",
    unsafe_allow_html=True,
)

# Force readable text and visible sidebar toggle
custom_css = """
<style>
/* Make all text dark and readable */
html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] * ,
[data-testid="stSidebar"],
[data-testid="stSidebar"] * {
    color: #111827 !important; /* dark slate */
}

/* ---- SIDEBAR TOGGLE BUTTON: MAKE IT BIG & VISIBLE ---- */
[data-testid="collapsedControl"] {
    opacity: 1 !important;
    display: flex !important;
    align-items: center;
    justify-content: center;
    width: 42px !important;
    height: 42px !important;
    color: #0f172a !important;      /* dark icon */
    background: #ffffff !important; /* white pill */
    border-radius: 999px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.30) !important;
    border: 1px solid rgba(148, 163, 184, 0.85) !important;
    cursor: pointer !important;
    z-index: 9999 !important;
}

/* Make sure the inner button inherits the styles */
[data-testid="collapsedControl"] > button {
    background: transparent !important;
    border: none !important;
    color: inherit !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ------------ HEADER ------------ #
st.markdown(
    """
    <style>
    html, body, [class^="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", system-ui, -system-ui, sans-serif !important;
        background: #f7f7f9 !important;
        font-size: 18px;
    }
    /* Restore dark text for all headers, labels, captions, and normal text */
    body, .stApp {
        color: #111827 !important;
    }
    /* Sidebar toggle visibility fix */
    [data-testid="collapsedControl"] {
        opacity: 1 !important;
        color: #0A0A0A !important;
        background: #f0f0f0 !important;
        padding: 6px !important;
        border-radius: 6px !important;
    }
    .main-card {
        background: #fff;
        border-radius: 18px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.07);
        padding: 2.5rem 2.5rem 2rem 2.5rem;
        margin-top: 2.5rem;
        margin-bottom: 2.5rem;
    }
    .stApp [data-testid="stSidebar"] {
        background: #f0f4f8 !important;
        width: 300px !important;
        min-width: 300px !important;
        max-width: 320px !important;
        padding: 2.5rem 1.5rem 2.5rem 1.5rem !important;
        box-sizing: border-box;
    }
    .sidebar-section {
        margin-bottom: 2.5rem;
    }
    .sidebar-section h2, .sidebar-section h3, .sidebar-section h4 {
        font-weight: 700;
        font-size: 1.25rem;
        margin-bottom: 1.2rem;
        margin-top: 0.5rem;
    }
    .sidebar-section > * {
        margin-bottom: 1.5rem;
    }
    .sidebar-section .stRadio, .sidebar-section .stCaption {
        margin-bottom: 2rem;
    }
    .main-title {
        font-size: 2.7rem;
        font-weight: 700;
        margin-top: 2.5rem;
        margin-bottom: 0.5rem;
        text-align: left;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #7a7a7a;
        margin-bottom: 2.5rem;
        text-align: left;
    }
    .form-row {
        margin-bottom: 1.5rem;
    }
    label {
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        display: block;
    }
    .predict-btn button {
        background: linear-gradient(90deg, #3bb2d7 0%, #1fa2ff 100%);
        color: #fff !important;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(63,178,215,0.12);
        padding: 0.75rem 2.5rem;
        font-size: 1.15rem;
        border: none;
        margin-top: 1.5rem;
        margin-bottom: 2rem;
    }
    .predict-btn button:hover {
        background: linear-gradient(90deg, #1fa2ff 0%, #3bb2d7 100%);
        box-shadow: 0 4px 16px rgba(63,178,215,0.18);
    }
    .stDataFrameContainer {
        margin-top: 2rem;
    }
    .prediction-result {
        margin-top: 2.5rem;
        margin-bottom: 2.5rem;
        font-size: 1.35rem;
        font-weight: 600;
        color: #1fa2ff;
    }
    /* ===== PREMIUM BUTTON STYLES ===== */
    .stButton > button,
    button[kind="secondary"],
    button[type="submit"],
    .stForm button {
        background: linear-gradient(135deg, #2F9E9E 0%, #64C0B5 100%) !important;
        border-radius: 14px !important;
        padding: 12px 24px !important;
        font-weight: 600 !important;
        color: #fff !important;
        border: none !important;
        box-shadow: none !important;
        margin-top: 1.1rem !important;
        margin-bottom: 1.1rem !important;
        font-family: inherit !important;
        transition: box-shadow 0.18s, transform 0.18s;
    }
    .stButton > button:hover,
    button[kind="secondary"]:hover,
    button[type="submit"]:hover,
    .stForm button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 6px 16px rgba(47, 158, 158, 0.25) !important;
        filter: brightness(1.04);
    }
    /* ===== PAGE LAYOUT & TOP BAR FIXES ===== */
    .stAppViewContainer {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    .block-container {
        padding-top: 0rem !important;
        margin-top: 0rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }
    main > div:first-child {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }
    body {
        margin: 0 !important;
    }
    /* Logo alignment and spacing */
    .center-logo {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 0.5rem;
        margin-bottom: 2.2rem;
    }
    /* Remove any extra margin above main content */
    .vmh-title {
        margin-top: 0 !important;
    }
    /* === Premium dark-blue gradient background === */
    /* === Premium dark-blue gradient background === */
body {
    background: linear-gradient(135deg, #0a2342, #1e3b70, #274060);
    background-attachment: fixed;
    color: #f9fafb !important;  /* soft light text OUTSIDE the main card */
}

/* Make the central white content card stand out */
.block-container {
    background: #ffffff !important;        /* solid white card */
    padding: 2rem 3rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    color: #111827 !important;             /* dark text INSIDE the card */
}

/* Text inside the main card should inherit its (dark) color */
.block-container h1,
.block-container h2,
.block-container h3,
.block-container h4,
.block-container p,
.block-container label,
.block-container span {
    color: inherit !important;
}

    }
    /* Make inputs and buttons fit high-end theme */
    .stButton>button {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: #f9fafb !important;
        padding: 0.7rem 2rem;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        transition: 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.35);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------ PAGE CONTENT ------------ #
# ------------ SIDEBAR ------------ #
(
    pipe_lr,
    pipe_xgb,
    feature_cols,
    target_name,
    lr_path,
    xgb_path,
    base_defaults,
) = load_models_and_columns()

with st.sidebar:
    st.sidebar.image("static/logo.svg", width=140)
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)
    section_choice = "Prediction"
    if section_choice == "Prediction":
        st.header("Model Selection")
        model_choice = st.radio(
            "Choose a prediction model:",
            ["Linear Regression", "XGBoost"],
            index=1,
        )
        if lr_path is not None:
            st.caption(f"Linear model: {lr_path.name}")
        else:
            st.caption("Linear model: (not loaded)")
        if xgb_path is not None:
            st.caption(f"XGBoost model: {xgb_path.name}")
        else:
            st.caption("XGBoost model: (not loaded)")


# ------------ MAIN CONTENT ------------ #
# --- MAIN PAGE TOP LOGO ---
st.markdown('<div class="center-logo vm-header-logo">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.image("static/logo.svg", width=340)
st.markdown("</div>", unsafe_allow_html=True)
# --- PAGE TITLE AND INTRO ---
st.markdown(
    '<div class="vmh-title">House Price Prediction Dashboard</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="vmh-subtitle">Welcome! Use this app to estimate the sale price of a house based on its features. Simply fill in the details below and click Predict Price.</div>',
    unsafe_allow_html=True,
)

if section_choice == "Prediction":
    st.markdown(
        '<div class="vmh-section-heading">Enter House Details</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        bedrooms = st.number_input(
            "Bedrooms", min_value=0, max_value=10, value=3, step=1, format="%d"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        bathrooms = st.number_input(
            "Bathrooms (whole number)",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            format="%d",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        floors = st.number_input(
            "Floors", min_value=0, max_value=10, value=1, step=1, format="%d"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        sqft_living = st.number_input("Living Area (sqft)", 300, 10000, 2000)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        sqft_above = st.number_input("Above-ground sqft", 300, 8000, 1800)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        sqft_basement = st.number_input("Basement sqft", 0, 3000, 200)
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        house_age = st.number_input("House Age (years)", 0, 150, 30)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="form-row">', unsafe_allow_html=True)

    zipcode = st.text_input("Zipcode", "98178")
    st.caption(
        "Try one of these valid ZIP codes: 98004, 98033, 98115, 98117, 98052, 98103"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Input validation (after all inputs are defined) ---
    validation_errors = []
    if bedrooms <= 0:
        validation_errors.append("Bedrooms must be greater than 0.")
    if bathrooms <= 0:
        validation_errors.append("Bathrooms must be greater than 0.")
    if floors <= 0:
        validation_errors.append("Floors must be greater than 0.")
    if sqft_living < 200 or sqft_living > 10000:
        validation_errors.append("Living area (sqft) should be between 200 and 10,000.")
    if "sqft_lot" in locals():
        if sqft_lot < 400 or sqft_lot > 100000:
            validation_errors.append(
                "Lot size (sqft) should be between 400 and 100,000."
            )

    # --- Location picker (Folium map) ---
    st.markdown("#### Location")

    # Default to Seattle if nothing is set yet
    DEFAULT_LAT, DEFAULT_LONG = 47.6062, -122.3321

    if "lat" not in st.session_state:
        st.session_state["lat"] = DEFAULT_LAT
    if "long" not in st.session_state:
        st.session_state["long"] = DEFAULT_LONG

    # Try to recenter map if valid zipcode entered
    map_center_lat = st.session_state["lat"]
    map_center_long = st.session_state["long"]
    highlight_circle = None

    col_map, col_coords = st.columns([3, 1])

    with col_map:
        # Build the Folium map centered on the current coordinates or searched zipcode
        m = folium.Map(
            location=[float(map_center_lat), float(map_center_long)],
            zoom_start=11,
        )

        # Marker for the currently selected location
        folium.Marker(
            [float(st.session_state["lat"]), float(st.session_state["long"])],
            tooltip="Selected location",
        ).add_to(m)

        # If zipcode search is active, add a highlight circle
        if highlight_circle:
            folium.Circle(
                location=highlight_circle,
                radius=800,
                color="#1F9BAA",
                fill=True,
                fill_opacity=0.15,
            ).add_to(m)

        # Render map and capture clicks
        map_data = st_folium(m, width=700, height=400)

        # If the user clicks on the map, update session_state
        if map_data and map_data.get("last_clicked"):
            st.session_state["lat"] = float(map_data["last_clicked"]["lat"])
            st.session_state["long"] = float(map_data["last_clicked"]["lng"])

    with col_coords:
        st.markdown("**Selected coordinates**")
        st.write(f"Latitude: {st.session_state['lat']:.5f}")
        st.write(f"Longitude: {st.session_state['long']:.5f}")

    # Build a row from the UI
    user_row = {
        "bedrooms": float(bedrooms),
        "bathrooms": int(bathrooms),
        "sqft_living": float(sqft_living),
        "sqft_above": float(sqft_above),
        "sqft_basement": float(sqft_basement),
        "floors": int(floors),
        "house_age": int(house_age),
        "zipcode": int(zipcode),
        "lat": float(st.session_state["lat"]),
        "long": float(st.session_state["long"]),
    }

    def build_feature_dataframe(
        user_values: dict, required_cols: list[str], defaults: dict
    ) -> pd.DataFrame:
        """
        Create a dataframe with all columns the model expects.

        - Start from typical defaults (medians/modes from training).
        - Override with the values the user provided.
        - Derive yr_built from house_age if the model uses yr_built.
        - Ensure every required column exists and is in the right order.
        """
        row = defaults.copy()
        row.update(user_values)

        # if model has yr_built, compute from house_age (year = 2025 - age)
        if "yr_built" in required_cols and "house_age" in row:
            row["yr_built"] = 2025 - row["house_age"]

        df = pd.DataFrame([row])

        for col in required_cols:
            if col not in df.columns:
                df[col] = 0

        df = df[required_cols]
        return df

    # ------------ PREDICTION ------------ #
    st.markdown(
        '<div class="prediction-header">Prediction</div>', unsafe_allow_html=True
    )

    # --- Prediction history state ---
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []

    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    if st.button("Predict Price"):
        if validation_errors:
            for msg in validation_errors:
                st.error(msg)
            # Do NOT run prediction if there are errors
        else:
            # Pick the active model
            pipe = pipe_xgb if model_choice == "XGBoost" else pipe_lr

            # Use the model's own training columns as source of truth
            if hasattr(pipe, "feature_names_in_"):
                required_cols = list(pipe.feature_names_in_)
            else:
                # Fallback to the list loaded from disk
                required_cols = feature_cols

            # Build feature dataframe that matches the model's expected columns
            df_features = build_feature_dataframe(
                user_values=user_row,
                required_cols=required_cols,
                defaults=base_defaults,
            )

            try:
                pred = float(pipe.predict(df_features)[0])
                pred = max(pred, 0)

                # Confidence interval
                lower = max(0, pred * 0.9)
                upper = pred * 1.1

                formatted_price = f"${pred:,.0f}"
                formatted_lower = f"${lower:,.0f}"
                formatted_upper = f"${upper:,.0f}"

                st.markdown(
                    f"""
<div style='text-align:center; margin-top:30px; margin-bottom:20px;'>
    <div style="
        font-size: 48px;
        font-weight: 800;
        color: #0D47A1;
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 25px 40px;
        border-radius: 20px;
        display: inline-block;
        box-shadow: 0 6px 18px rgba(0,0,0,0.15);
    ">
        Estimated Price<br>
        <span style='font-size: 64px; color:#003c8f;'>
            {formatted_price}
        </span>
    </div>
</div>
""",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<p style='text-align:center; font-size:16px; color:#555;'>"
                    f"Estimated range: {formatted_lower} – {formatted_upper} (±10%)</p>",
                    unsafe_allow_html=True,
                )

                # Save to history
                st.session_state["prediction_history"].append(
                    {
                        "prediction": pred,
                        "bedrooms": user_row["bedrooms"],
                        "bathrooms": user_row["bathrooms"],
                        "sqft_living": user_row["sqft_living"],
                        "zipcode": user_row["zipcode"],
                        "lat": user_row["lat"],
                        "long": user_row["long"],
                    }
                )
            except Exception as exc:
                st.error("Prediction failed. Please check your inputs.")
                st.exception(exc)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Show prediction history chart and table ---
    history = st.session_state["prediction_history"]
    if history:
        hist_df = pd.DataFrame(history)
        st.markdown(
            "<h3 class='prediction-history-title'>Prediction history</h3>",
            unsafe_allow_html=True,
        )
        st.line_chart(hist_df["prediction"])
        st.markdown("##### Last 5 predictions")
        hist_df_display = hist_df.copy()
        hist_df_display["prediction"] = hist_df_display["prediction"].apply(
            lambda x: f"${x:,.0f}"
        )
        st.dataframe(
            hist_df_display[
                [
                    "prediction",
                    "bedrooms",
                    "bathrooms",
                    "sqft_living",
                    "zipcode",
                    "lat",
                    "long",
                ]
            ].tail(5),
            use_container_width=True,
        )

    # --- Nearest Neighbour Feature ---
    import numpy as np

    try:
        # Load dataset
        df_nn = pd.read_csv("data/processed/dataset.csv")
        # Columns to use
        nn_cols = ["bedrooms", "bathrooms", "sqft_living", "lat", "long", "zipcode"]
        # Build user vector
        user_vec = np.array(
            [
                float(bedrooms),
                float(bathrooms),
                float(sqft_living),
                float(st.session_state["lat"]),
                float(st.session_state["long"]),
                int(zipcode),
            ]
        )
        # Compute distances
        df_nn["_dist"] = np.linalg.norm(
            df_nn[nn_cols].astype(float).values - user_vec, axis=1
        )
        # Exclude the home itself (distance==0)
        df_nn = df_nn[df_nn["_dist"] > 0]
        # Get 5 nearest
        neighbors_df = df_nn.nsmallest(5, "_dist")[
            ["price", "sqft_living", "bedrooms", "bathrooms", "zipcode", "lat", "long"]
        ]
        st.subheader("Nearest Comparable Homes")
        st.dataframe(neighbors_df, use_container_width=True)
    except Exception as e:
        st.info("Could not compute nearest neighbours: " + str(e))
else:
    st.markdown("### Model insights")
    st.info("Here we will show feature importance and error analysis.")


# ------------ FOOTER ------------ #
st.markdown("---")
st.caption("© 2025 House Price Predictor. All rights reserved.")
