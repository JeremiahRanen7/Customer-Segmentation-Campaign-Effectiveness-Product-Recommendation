import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Complete Journey - Capstone", layout="wide")


# ---- cluster metadata ----

cluster_names = {
    0: "Core Loyalists",
    1: "One-Time & Full-Price",
    2: "Dormant At-Risk",
}

cluster_taglines = {
    0: "Frequent, high-volume, coupon-engaged.",
    1: "Single transaction, full price, no return.",
    2: "Long gaps, low spend, churn risk.",
}

cluster_descriptions = {
    0: "Heavy hitters. They visit frequently, buy in the largest volumes, "
       "and have shopped very recently. Already engaged - coupons just "
       "amplify what they were going to buy anyway.",
    1: "Single-transaction shoppers. One standard-sized purchase at full "
       "price, no return visit. The goal here is activating a second buy.",
    2: "Previously active customers, now disengaged. Long inter-purchase "
       "gaps, low lifetime spend, minimal response to standard promotions.",
}

cluster_strategies = {
    0: [
        "Retention and reward, not discount-led acquisition.",
        "Volume-based perks (\"Buy 3 get 1\") over flat % off.",
        "VIP loyalty programmes, early access, bundles.",
    ],
    1: [
        "Drive the second purchase with a targeted welcome-back offer.",
        "Personalised first-return discount.",
        "Avoid generic mass campaigns - evidence shows they backfire.",
    ],
    2: [
        "Aggressive win-back with deep re-activation discounts.",
        "'We miss you' direct outreach.",
        "Don't waste standard-campaign spend here - conversion is near zero.",
    ],
}


# ---- features (must match the saved scaler) ----

feature_cols = [
    "total_sales_x", "avg_sales", "total_quantity", "avg_quantity",
    "avg_basket_size", "total_orders", "unique_visits_days",
    "order_frequency", "relative_recency", "avg_days_between_visits",
    "coupon_dependecy_ratio",
]

feature_labels = {
    "total_sales_x":           "Total lifetime sales ($)",
    "avg_sales":               "Average sale per line ($)",
    "total_quantity":          "Total quantity (all items)",
    "avg_quantity":            "Average quantity per line",
    "avg_basket_size":         "Average basket size (qty / basket)",
    "total_orders":            "Total orders (baskets)",
    "unique_visits_days":      "Unique visit days",
    "order_frequency":         "Order frequency (days / order)",
    "relative_recency":        "Relative recency (days since last buy)",
    "avg_days_between_visits": "Avg days between visits",
    "coupon_dependecy_ratio":  "Coupon dependency ratio",
}

feature_help = {
    "total_sales_x":           "Lifetime revenue contributed by the household.",
    "avg_sales":               "Average $ value per individual line item.",
    "total_quantity":          "Total units purchased over the full window.",
    "avg_quantity":            "Average units per individual line item.",
    "avg_basket_size":         "Units per basket on a typical trip.",
    "total_orders":            "Number of distinct baskets / shopping trips.",
    "unique_visits_days":      "Distinct calendar days the household shopped.",
    "order_frequency":         "Active-window length divided by total orders.",
    "relative_recency":        "Days between last purchase and the end of study (712).",
    "avg_days_between_visits": "Mean gap between consecutive visit days.",
    "coupon_dependecy_ratio":  "Share of spend on coupon-discounted transactions.",
}

campaign_type_desc = {
    "TypeA": "Mid-frequency: 5 campaigns. Medium average duration.",
    "TypeB": "High-frequency: 19 campaigns. Short duration.",
    "TypeC": "Low-frequency: 6 campaigns. Longest average run window.",
}


# ---- load artifacts (these come from running the notebooks) ----

@st.cache_data
def load_households():
    return pd.read_parquet("artifacts/households.parquet")

@st.cache_data
def load_similarity():
    data = np.load("artifacts/similarity.npz", allow_pickle=False)
    return data["matrix"], data["index"]

@st.cache_data
def load_household_top_items():
    return pd.read_parquet("artifacts/household_top_items.parquet")

@st.cache_data
def load_segment_top_items():
    return pd.read_parquet("artifacts/segment_top_items.parquet")

@st.cache_data
def load_regression():
    with open("artifacts/regression.json") as f:
        return json.load(f)

@st.cache_data
def load_campaign_overview():
    return pd.read_parquet("artifacts/campaign_overview.parquet")

@st.cache_resource
def load_models():
    scaler = joblib.load("models/scaler.joblib")
    pca    = joblib.load("models/pca.joblib")
    kmeans = joblib.load("models/kmeans_3.joblib")
    return scaler, pca, kmeans


# ---- helpers ----

def predict_cluster(scaler, pca, kmeans, feature_dict):
    row = pd.DataFrame([{c: feature_dict[c] for c in feature_cols}])
    scaled = scaler.transform(row)
    pcs = pca.transform(scaled)
    cluster = int(kmeans.predict(pcs)[0])
    return cluster, pcs[0]


def get_similar_households(hid, sim, idx, n=5):
    pos = int(np.where(idx == hid)[0][0])
    sims = sim[pos]
    order = np.argsort(-sims)
    order = order[order != pos][:n]  # drop self
    return pd.DataFrame({
        "household_key": idx[order].astype(int),
        "similarity": sims[order],
    })


def top5_recommendations(hid, sim, idx, hh_top, n=5):
    nbrs = get_similar_households(hid, sim, idx, n=5)
    pool = hh_top[hh_top["household_key"].isin(nbrs["household_key"])]
    ranked = (pool.groupby("SUB_COMMODITY_DESC")["total_quantity"]
                  .sum().sort_values(ascending=False).head(n))
    return ranked.index.tolist()


# ============================
# tabs
# ============================

tab_overview, tab_seg, tab_eff, tab_rec = st.tabs(
    ["Overview", "Segmentation", "Effectiveness", "Recommendations"]
)


# ---- Tab: Overview ----
with tab_overview:
    st.title("Complete Journey")
    st.write(
        "A retail-analytics capstone built on Dunnhumby's \"Complete Journey\" "
        "dataset: 2,500 households, two years of basket-level transactions, "
        "30 campaigns, 1,600+ sub-commodities. Three problems, one app."
    )

    hh = load_households()
    cov = load_campaign_overview()

    c1, c2, c3 = st.columns(3)
    c1.metric("Households analyzed", f"{len(hh):,}")
    c1.caption("Active over 712 days of trading.")
    c2.metric("Campaigns evaluated", f"{len(cov):,}")
    c2.caption("Three campaign types: A, B, C.")
    c3.metric("Segments derived", "3")
    c3.caption("From RFM features via PCA + KMeans.")

    st.divider()

    st.subheader("From basket-level transactions to a working model")

    p1, p2, p3, p4 = st.columns(4)
    with p1.container(border=True):
        st.caption("01 - Raw")
        st.markdown("**Dunnhumby tables**")
        st.write("Transactions, products, campaigns, coupons, demographics.")
    with p2.container(border=True):
        st.caption("02 - Segment")
        st.markdown("**RFM + PCA + KMeans**")
        st.write("Eleven behavioural features, five PCs, three clusters.")
    with p3.container(border=True):
        st.caption("03 - Effectiveness")
        st.markdown("**Panel regression**")
        st.write("Household x week fixed-effects, cluster x campaign interactions.")
    with p4.container(border=True):
        st.caption("04 - Recommend")
        st.markdown("**Collaborative filter**")
        st.write("Cosine similarity on household x sub-commodity quantities.")

    st.divider()

    st.subheader("Three problems")
    pcol1, pcol2, pcol3 = st.columns(3, gap="large")
    with pcol1:
        st.markdown("**Who are our customers?**")
        st.write(
            "Segmentation on RFM-style features. Eleven behavioural "
            "indicators reduce to five principal components and three "
            "behavioural clusters - Core Loyalists, One-Time Buyers, "
            "Dormant At-Risk."
        )
    with pcol2:
        st.markdown("**Which campaigns actually work?**")
        st.write(
            "A two-way fixed-effects panel model with cluster x campaign-type "
            "interactions, run on a 255,000-row household-week panel."
        )
    with pcol3:
        st.markdown("**What should we recommend next?**")
        st.write(
            "Item-popularity at the segment level and item-item collaborative "
            "filtering at the household level via cosine similarity on the "
            "household x sub-commodity quantity matrix."
        )


# ---- Tab: Segmentation ----
with tab_seg:
    st.title("Predict a customer's segment")
    st.write(
        "Enter household behavioural features below - or load a real "
        "household from the trained set."
    )

    scaler, pca, kmeans = load_models()
    hh = load_households()

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        mode = st.radio(
            "Input mode",
            ["Load a real household", "Manual entry"],
            horizontal=True,
            label_visibility="collapsed",
        )

        # default ranges, taken from the loaded households
        ranges = {c: (float(hh[c].min()), float(hh[c].max()),
                      float(hh[c].median())) for c in feature_cols}

        if mode == "Load a real household":
            hid = st.selectbox(
                "Household key",
                options=sorted(hh["household_key"].tolist()),
                index=0,
                help="Pick any of the 2,500 households in the training data.",
            )
            row = hh.loc[hh["household_key"] == hid, feature_cols].iloc[0]
            feature_values = row.to_dict()

            with st.expander("Inspect this household's features"):
                preview = pd.DataFrame({
                    "Feature": [feature_labels[c] for c in feature_cols],
                    "Value":   [row[c] for c in feature_cols],
                })
                st.dataframe(preview, hide_index=True, use_container_width=True)

        else:
            feature_values = {}

            st.markdown("**Spending**")
            c1, c2 = st.columns(2)
            with c1:
                feature_values["total_sales_x"] = st.number_input(
                    feature_labels["total_sales_x"], min_value=0.0,
                    max_value=ranges["total_sales_x"][1] * 1.5,
                    value=ranges["total_sales_x"][2], step=100.0,
                    help=feature_help["total_sales_x"])
                feature_values["avg_sales"] = st.number_input(
                    feature_labels["avg_sales"], min_value=0.0,
                    max_value=ranges["avg_sales"][1] * 1.5,
                    value=ranges["avg_sales"][2], step=0.1,
                    help=feature_help["avg_sales"])
                feature_values["total_quantity"] = st.number_input(
                    feature_labels["total_quantity"], min_value=0.0,
                    max_value=ranges["total_quantity"][1] * 1.5,
                    value=ranges["total_quantity"][2], step=10.0,
                    help=feature_help["total_quantity"])
            with c2:
                feature_values["avg_quantity"] = st.number_input(
                    feature_labels["avg_quantity"], min_value=0.0,
                    max_value=ranges["avg_quantity"][1] * 1.5,
                    value=ranges["avg_quantity"][2], step=1.0,
                    help=feature_help["avg_quantity"])
                feature_values["avg_basket_size"] = st.number_input(
                    feature_labels["avg_basket_size"], min_value=0.0,
                    max_value=ranges["avg_basket_size"][1] * 1.5,
                    value=ranges["avg_basket_size"][2], step=1.0,
                    help=feature_help["avg_basket_size"])

            st.markdown("**Frequency**")
            c1, c2 = st.columns(2)
            with c1:
                feature_values["total_orders"] = st.number_input(
                    feature_labels["total_orders"], min_value=0.0,
                    max_value=ranges["total_orders"][1] * 1.5,
                    value=ranges["total_orders"][2], step=1.0,
                    help=feature_help["total_orders"])
                feature_values["unique_visits_days"] = st.number_input(
                    feature_labels["unique_visits_days"], min_value=0.0,
                    max_value=712.0, value=ranges["unique_visits_days"][2],
                    step=1.0, help=feature_help["unique_visits_days"])
            with c2:
                feature_values["order_frequency"] = st.number_input(
                    feature_labels["order_frequency"], min_value=0.0,
                    max_value=ranges["order_frequency"][1] * 1.5,
                    value=ranges["order_frequency"][2], step=0.5,
                    help=feature_help["order_frequency"])
                feature_values["avg_days_between_visits"] = st.number_input(
                    feature_labels["avg_days_between_visits"], min_value=0.0,
                    max_value=ranges["avg_days_between_visits"][1] * 1.5,
                    value=ranges["avg_days_between_visits"][2], step=0.5,
                    help=feature_help["avg_days_between_visits"])

            st.markdown("**Recency & promotions**")
            c1, c2 = st.columns(2)
            with c1:
                feature_values["relative_recency"] = st.number_input(
                    feature_labels["relative_recency"], min_value=0.0,
                    max_value=712.0, value=ranges["relative_recency"][2],
                    step=1.0, help=feature_help["relative_recency"])
            with c2:
                feature_values["coupon_dependecy_ratio"] = st.number_input(
                    feature_labels["coupon_dependecy_ratio"], min_value=0.0,
                    max_value=1.0,
                    value=float(ranges["coupon_dependecy_ratio"][2]),
                    step=0.01, format="%.3f",
                    help=feature_help["coupon_dependecy_ratio"])

        st.write("")
        clicked = st.button("Predict segment", type="primary",
                            use_container_width=True)

    # right column: prediction result
    with right:
        if clicked:
            cluster, pcs = predict_cluster(scaler, pca, kmeans, feature_values)
            st.session_state["last_pred"] = {"cluster": cluster, "pcs": pcs}

        if "last_pred" in st.session_state:
            cluster = st.session_state["last_pred"]["cluster"]

            with st.container(border=True):
                st.caption(f"PREDICTED SEGMENT - CLUSTER {cluster}")
                st.subheader(cluster_names[cluster])
                st.write(f"*{cluster_taglines[cluster]}*")
                st.write(cluster_descriptions[cluster])
                st.markdown("**Recommended strategy**")
                for s in cluster_strategies[cluster]:
                    st.write(f"- {s}")
        else:
            with st.container(border=True):
                st.caption("AWAITING INPUT")
                st.write("Choose a household or enter features, then predict.")


# ---- Tab: Effectiveness ----
with tab_eff:
    st.title("Which campaigns actually move sales?")
    st.write(
        "A two-way fixed-effects panel model (household x week) with "
        "cluster x campaign-type interactions, fit on 255,000 "
        "household-week observations."
    )

    reg = load_regression()
    overview = load_campaign_overview()

    # headline finding: strongest positive significant interaction
    sales_int = {k: v for k, v in reg["params"].items() if "_X_segment_" in k}
    best_pos = max(sales_int.items(), key=lambda kv: kv[1]["estimate"])
    st.info(
        f"**Findings:** Significant positive uplift exists only for Core "
        f"Loyalists (segment 0). The strongest single lever is "
        f"**TypeC x Loyalists at +{best_pos[1]['estimate']:.2f} $/household-week** "
        f"(p < {best_pos[1]['p_value']:.4f})."
    )

    sub_overview, sub_effect = st.tabs(["Overview", "Effectiveness by Segment"])

    # ---- sub: Overview ----
    with sub_overview:
        st.subheader("Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total campaigns", str(len(overview)))
        c1.caption("Three campaign types in play.")

        type_counts = " - ".join(
            str(int((overview["campaign_type"] == t).sum()))
            for t in ["TypeA", "TypeB", "TypeC"]
        )
        c2.metric("TypeA - TypeB - TypeC", type_counts)
        c2.caption("Count by type.")

        c3.metric("Households reached",
                  f"{int(overview['households_reached'].sum()):,}")
        c3.caption("Sum across campaigns (with overlap).")

        c4.metric("Avg duration", f"{overview['duration_days'].mean():.0f} d")
        c4.caption("Mean campaign length.")

        st.divider()

        cleft, cright = st.columns([1.1, 1], gap="large")
        with cleft:
            st.markdown("**Duration by campaign type**")
            by_type = (overview.groupby("campaign_type")["duration_days"]
                                .mean().reset_index()
                                .sort_values("duration_days", ascending=True))
            longest = by_type["campaign_type"].iloc[-1]
            # color the longest one in sky, others in two grey shades so
            # the middle and smallest bars are distinguishable
            non_focal = ["#94A3B8", "#374151"]
            colors, j = [], 0
            for t in by_type["campaign_type"]:
                if t == longest:
                    colors.append("#4A90B8")
                else:
                    colors.append(non_focal[j]); j += 1

            fig = go.Figure(go.Bar(
                y=by_type["campaign_type"], x=by_type["duration_days"],
                orientation="h", marker_color=colors,
                text=[f"{d:.0f} d" for d in by_type["duration_days"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=300, xaxis_title="Average days", yaxis_title="",
                showlegend=False,
                margin=dict(l=20, r=60, t=20, b=40),
                xaxis_range=[0, by_type["duration_days"].max() * 1.18],
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Type C runs the longest, on average - but is also the "
                       "rarest, with only six deployments over the two-year window.")

        with cright:
            st.markdown("**Campaign-type reference**")
            for t in ["TypeA", "TypeB", "TypeC"]:
                st.markdown(f"**{t}.** {campaign_type_desc[t]}")

    # ---- sub: Effectiveness by Segment ----
    with sub_effect:
        st.subheader("Where does the uplift land?")
        st.write(
            "Each bar is one interaction coefficient from the panel model. "
            "Positive = sales uplift on that segment during weeks lagged "
            "four periods after a campaign of that type. Sky bars are "
            "statistically significant positive effects (p < 0.05); terracotta "
            "are significant negative; grey are non-significant."
        )

        # collect interaction coefficients into a frame
        seg_map = {"segment_0": "Loyalists",
                   "segment_1": "One-Time",
                   "segment_2": "Dormant"}
        type_map = {"TypeA": "Type A", "TypeB": "Type B", "TypeC": "Type C"}

        rows = []
        for k, v in reg["params"].items():
            if "_X_segment_" not in k:
                continue
            parts = k.split("_X_")
            camp = parts[0].replace("campaign_type_", "").replace("_lag4", "")
            seg = parts[1]
            rows.append({
                "camp": type_map[camp],
                "seg":  seg_map[seg],
                "label": f"{type_map[camp]} x {seg_map[seg]}",
                "estimate": v["estimate"],
                "p_value": v["p_value"],
                "ci_low": v["ci_low"],
                "ci_high": v["ci_high"],
            })
        ints = pd.DataFrame(rows).sort_values("estimate")

        def bar_color(row):
            if row["p_value"] >= 0.05:
                return "#94A3B8"  # non-significant grey
            return "#4A90B8" if row["estimate"] > 0 else "#C97064"

        ints["color"] = ints.apply(bar_color, axis=1)
        ints["sig_label"] = ints["p_value"].apply(
            lambda p: "p<0.001" if p < 0.001 else f"p={p:.3f}")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=ints["label"], x=ints["estimate"], orientation="h",
            marker_color=ints["color"],
            error_x=dict(
                type="data",
                arrayminus=ints["estimate"] - ints["ci_low"],
                array=ints["ci_high"] - ints["estimate"],
                thickness=1.0,
            ),
            text=[f"{e:+.2f} ({p})" for e, p in
                  zip(ints["estimate"], ints["sig_label"])],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_width=1)
        fig.update_layout(
            height=440, xaxis_title="Effect on weekly sales ($)",
            yaxis_title="", showlegend=False,
            margin=dict(l=20, r=80, t=20, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Reading the chart**")
        # pull the headline number out of the frame
        try:
            typec_loy = float(
                ints.loc[(ints["seg"] == "Loyalists") &
                         (ints["camp"] == "Type C"), "estimate"].iloc[0])
        except (IndexError, KeyError):
            typec_loy = 3.31
        st.write(
            f"- **All three campaign types lift Loyalists significantly.** "
            f"TypeC x Loyalists is the strongest single lever at "
            f"+{typec_loy:.2f} $ per household per week."
        )
        st.write(
            "- **One-Time buyers do not respond - and TypeB actively backfires.** "
            "The TypeB x One-Time coefficient is significantly negative."
        )
        st.write(
            "- **Dormant households are unresponsive to all three types.** "
            "Standard campaigns are the wrong approach here."
        )

        st.write("")
        st.markdown("**Full parameter table**")
        st.dataframe(
            pd.DataFrame([
                {"Parameter": p,
                 "Estimate": v["estimate"], "Std. Err.": v["std_err"],
                 "t-stat": v["t_stat"], "p-value": v["p_value"],
                 "CI low": v["ci_low"], "CI high": v["ci_high"]}
                for p, v in reg["params"].items()
            ]).round(4),
            hide_index=True, use_container_width=True,
        )


# ---- Tab: Recommendations ----
with tab_rec:
    st.title("What should we recommend next?")
    st.write(
        "A two-step approach. At the segment level - for popularity based "
        "suggestions. At the household level - for personalisation - "
        "cosine-similarity collaborative filtering on the household x "
        "sub-commodity quantity matrix."
    )

    seg_top = load_segment_top_items()
    hh = load_households()
    hh_top = load_household_top_items()
    sim, sim_index = load_similarity()

    # --- segment-level top 5 ---
    st.subheader("Top 5 sub-commodities, per segment")
    st.caption(
        "Top items by quantity within each cluster, with common items "
        "(gasoline, fluid milk, white bread, bananas, shredded cheese, "
        "condensed soup, and the two largest soft-drink categories) excluded "
        "so segment-distinctive items can be recommended."
    )
    st.write("")

    cols = st.columns(3, gap="medium")
    for col, c in zip(cols, [0, 1, 2]):
        items = seg_top[seg_top["cluster"] == c]["SUB_COMMODITY_DESC"].tolist()
        with col.container(border=True):
            st.caption(f"Cluster {c}")
            st.markdown(f"**{cluster_names[c]}**")
            st.write(f"*{cluster_taglines[c]}*")
            for i, name in enumerate(items, start=1):
                st.write(f"{i:02d}. {name}")

    st.divider()

    # --- household-level recommendations ---
    st.subheader("Top 5 recommendations for a specific household")
    st.caption(
        "Pick a household and the engine finds its five most similar "
        "households by cosine similarity on the purchase matrix, then "
        "aggregates the top items those neighbours actually buy."
    )
    st.write("")

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        hid = st.selectbox(
            "Household key",
            options=sorted(hh["household_key"].astype(int).tolist()),
            index=0,
            help="Any of the 2,500 households in the training set.",
        )
        cluster_of_hid = int(hh.loc[hh["household_key"] == hid,
                                    "cluster"].iloc[0])
        with st.container(border=True):
            st.caption("This household sits in")
            st.markdown(f"**{cluster_names[cluster_of_hid]}**")
            st.caption(
                f"Cluster {cluster_of_hid} - {cluster_taglines[cluster_of_hid]}"
            )

        # nearest neighbours
        nbrs = get_similar_households(hid, sim, sim_index, n=5)
        nbrs = nbrs.merge(hh[["household_key", "cluster"]],
                          on="household_key", how="left")
        nbrs["segment"] = nbrs["cluster"].map(cluster_names)

        st.write("")
        st.markdown("**Nearest 5 neighbours**")
        st.dataframe(
            nbrs[["household_key", "segment", "similarity"]]
                .rename(columns={"household_key": "Household",
                                 "segment": "Segment",
                                 "similarity": "Cosine similarity"})
                .round({"Cosine similarity": 4}),
            hide_index=True, use_container_width=True,
        )

    with right:
        recs = top5_recommendations(hid, sim, sim_index, hh_top, n=5)
        with st.container(border=True):
            st.caption("Recommended next")
            st.markdown(f"**Top 5 sub-commodities for household {hid}**")
            st.write("*From neighbours' purchase patterns.*")
            for i, name in enumerate(recs, start=1):
                st.write(f"{i:02d}. {name}")

        # context: this household's own top items
        own = (hh_top[hh_top["household_key"] == hid]
                      .sort_values("total_quantity", ascending=False)
                      .head(5))
        if len(own):
            st.write("")
            st.markdown("**For context - this household's own top items**")
            own_disp = own[["SUB_COMMODITY_DESC", "total_quantity",
                            "total_sales"]].rename(columns={
                "SUB_COMMODITY_DESC": "Sub-commodity",
                "total_quantity": "Quantity",
                "total_sales": "Sales ($)",
            }).round({"Sales ($)": 2})
            st.dataframe(own_disp, hide_index=True, use_container_width=True)
